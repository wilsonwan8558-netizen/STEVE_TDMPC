"""Frozen G9.6 paths, profile net, inputs, and audited source contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g95", "generic_solver_g9", "generic_solver_g6a",
               "generic_solver_g5", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from g95_common import F_LIM, TAUS, SEEDS, cuda_gate, json_new, sha, source_gate  # noqa: E402
from g95_confirmation_data import verify_freeze as verify_g95_freeze  # noqa: E402
from g9_risk import data as g9_data, matrix as g9_matrix  # noqa: E402

G9 = RL / "logs/generic_solver_phase_g9_temporal_anticipatory_safety"
G95 = RL / "logs/generic_solver_phase_g95_planner_conditioned_risk"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline"
OUT = RL / "logs/generic_solver_phase_g96_stepwise_h2_force_profile"
CONFIRM = tuple(range(104000, 104030))
GROUPS = ("train", "validation", "development_test")


class StepwiseNet(nn.Module):
    """Exact G9 64-64 trunk with two independent noncrossing quantile heads."""

    def __init__(self, dim=298):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Linear(64, 8)

    def forward(self, x):
        raw = self.head(self.trunk(x)).reshape(-1, 2, 4)
        return torch.cumsum(F.softplus(raw), dim=-1)


def profile_pinball(pred, target):
    if pred.ndim != 3 or pred.shape[1:] != (2, 4) or target.shape != pred.shape[:2]:
        raise RuntimeError("G9.6 profile loss shape changed")
    levels = torch.as_tensor(TAUS, dtype=pred.dtype, device=pred.device)
    error = target[:, :, None] - pred
    individual = torch.maximum(levels * error, (levels - 1) * error)
    return individual[:, 0].mean() + individual[:, 1].mean()


def predict_profile(net, x, mean, scale):
    net.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            normalized = (x[start:start+4096] - mean) / scale
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device="cuda:0")
            parts.append(net(tensor).cpu().numpy())
    p = np.concatenate(parts).astype(np.float64)
    if (p.shape != (len(x), 2, 4) or not np.all(np.isfinite(p)) or
            np.any(np.diff(p, axis=2) < 0) or np.any(p < 0)):
        raise RuntimeError("G9.6 stepwise output crossing/nonfinite/nonnegative invariant failed")
    return p


def frozen_sources():
    cuda_gate()
    source = source_gate()
    frozen = verify_g95_freeze()
    manifest = json.loads((G95 / "model_training_manifest.json").read_text())
    if (manifest["training_seeds"] != list(SEEDS) or
            manifest["G95_planner_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz") or
            manifest["G9_behavior_temporal_sha256"] != sha(G9 / "development_temporal_behavior.npz") or
            manifest["G9_risk_normalization_sha256"] != sha(G9 / "models/risk/E1/H2/normalization.json")):
        raise RuntimeError("G9.5 training dataset/normalization contract changed")
    baseline = {}
    for seed in SEEDS:
        key = manifest["models"]["M1_planner_conditioned"][str(seed)]
        path = G95 / "models/M1_planner_conditioned" / f"seed_{seed}.pt"
        selection = G95 / "models/M1_planner_conditioned" / f"seed_{seed}_selection.json"
        if sha(path) != key["checkpoint_sha256"] or sha(selection) != key["selection_sha256"]:
            raise RuntimeError(f"G9.5 scalar baseline changed {seed}")
        baseline[str(seed)] = {"checkpoint_path": str(path),
                               "checkpoint_sha256": sha(path),
                               "selection_sha256": sha(selection),
                               "Q95_delta_N": key["validation_Q95_delta_N"]}
    return {"g95_pre_confirmation_freeze_sha256": sha(G95 / "pre_confirmation_freeze.json"),
            "g95_development_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
            "g95_confirmation_dataset_sha256": sha(G95 / "confirmation_planner_dataset.npz"),
            "g9_behavior_dataset_sha256": sha(G9 / "development_temporal_behavior.npz"),
            "g9_E1_estimated_state_sha256": sha(G9 / "estimated_current_physical.npz"),
            "g9_input_normalization_sha256": sha(G9 / "models/risk/E1/H2/normalization.json"),
            "planner_source_sha256": source["planner_source_sha256"],
            "controller_checkpoint_sha256": source["checkpoint_sha256"],
            "g95_frozen_status": frozen["status"],
            "scalar_baseline": baseline}


def training_arrays():
    """Exactly M1's behavior+planner rows, replacing only scalar y with F1/F2."""
    d, estimates, split = g9_data()
    bx, by = g9_matrix(d, estimates["E1"], 2)
    behavior_y = d["future_force_sequence"][:, :2].astype(np.float64)
    if bx.shape != (len(behavior_y), 298) or not np.array_equal(
            by.astype(np.float32), np.max(behavior_y, axis=1).astype(np.float32)):
        raise RuntimeError("G9.6 behavior profile/scalar target alignment failed")
    with np.load(G95 / "development_planner_dataset.npz") as z:
        p = {key: z[key].copy() for key in z.files}
    planner_y = np.stack((p["F1_true_N"], p["F2_true_N"]), axis=1).astype(np.float64)
    if (p["x"].shape != (len(planner_y), 298) or
            not np.array_equal(np.max(planner_y, axis=1).astype(np.float32), p["R2_true_N"])):
        raise RuntimeError("G9.6 planner profile/scalar target alignment failed")
    masks = {group: np.isin(d["seed"], split[group]) for group in GROUPS}
    pmasks = {group: p["group_index"] == index for index, group in enumerate(GROUPS)}
    norm = json.loads((G9 / "models/risk/E1/H2/normalization.json").read_text())
    mean, scale = np.asarray(norm["mean"], dtype=np.float64), np.asarray(norm["std"], dtype=np.float64)
    if mean.shape != (298,) or scale.shape != (298,) or np.any(scale <= 0):
        raise RuntimeError("G9.6 risk input normalization changed")
    return bx, behavior_y, masks, p, planner_y, pmasks, mean, scale


def group_data(group):
    bx, by, bm, p, py, pm, mean, scale = training_arrays()
    x = np.concatenate((bx[bm[group]], p["x"][pm[group]].astype(np.float64)))
    y = np.concatenate((by[bm[group]], py[pm[group]]))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise RuntimeError(f"G9.6 {group} training/validation data invalid")
    return x, y, mean, scale, int(np.sum(bm[group])), int(np.sum(pm[group]))


def assert_prepared():
    source = frozen_sources()
    metadata = json.loads((OUT / "metadata.json").read_text())
    if source != metadata["frozen_G95_sources"] or metadata["confirmation_seeds_unopened"] != list(CONFIRM):
        raise RuntimeError("G9.6 prepared frozen source changed")
    audit = json.loads((OUT / "g95_data_audit.json").read_text())
    if audit["status"] != "PASS_EXACT_G95_F1_F2_ALIGNMENT":
        raise RuntimeError("G9.6 G9.5 data alignment audit failed")
    return source
