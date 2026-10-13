"""Frozen G10.2 local ensemble and force-constrained translation search."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g102_prepare import OUT, SEEDS, SEARCH_AXIS, F_LIM, verify_source  # noqa: E402
from g102_train import DeltaNet, predict  # noqa: E402
from g10_repair import FrozenRepair  # noqa: E402
from g95_common import sha  # noqa: E402


def key(value):
    return float(np.float32(value))


def search_points(v_nom):
    nominal = np.float32(v_nom)
    values = {key(v) for v in SEARCH_AXIS}
    values.add(key(nominal))
    for delta in (.05, .1, .2):
        values.add(key(np.clip(nominal-np.float32(delta), -1, 1)))
        values.add(key(np.clip(nominal+np.float32(delta), -1, 1)))
    return np.asarray(sorted(values), dtype=np.float32)


class LocalEnsemble:
    def __init__(self):
        verify_source()
        manifest = json.loads((OUT / "model_training_manifest.json").read_text())
        rule = json.loads((OUT / "validation_ensemble_rule.json").read_text())
        if (manifest["training_seeds"] != list(SEEDS) or manifest["M_LOCAL_ABS_trained"] or
                sha(OUT / "normalization.json") != manifest["normalization_sha256"] or
                sha(OUT / "validation_ensemble_rule.json") != manifest["validation_ensemble_rule_sha256"]):
            raise RuntimeError("G10.2 frozen M_DELTA manifest changed")
        norm = json.loads((OUT / "normalization.json").read_text())
        self.mean = np.asarray(norm["mean"], dtype=np.float64)
        self.std = np.asarray(norm["std"], dtype=np.float64)
        if self.mean.shape != (296,) or self.std.shape != (296,) or np.any(self.std <= 0):
            raise RuntimeError("G10.2 input normalization invalid")
        self.nets = []
        self.corrections = []
        for replica_index, seed in enumerate(SEEDS, start=1):
            info = manifest["models"][str(seed)]
            path = OUT / "models" / f"delta_seed_{replica_index}" / "model.pt"
            selection = OUT / "models" / f"delta_seed_{replica_index}" / "selection.json"
            if sha(path) != info["checkpoint_sha256"] or sha(selection) != info["selection_sha256"]:
                raise RuntimeError(f"G10.2 frozen local replica changed {seed}")
            net = DeltaNet().to("cuda:0")
            net.load_state_dict(torch.load(path, map_location="cuda:0", weights_only=True))
            net.eval()
            self.nets.append(net)
            self.corrections.append(info["validation_only_Q95_delta_correction_N"])
        self.corrections = np.asarray(self.corrections, dtype=np.float64)
        self.spread_threshold = rule["Q95_corrected_replica_spread_threshold_N"]
        self.nominal_model = FrozenRepair()

    def score(self, x294, pair, v_values):
        x294 = np.asarray(x294, dtype=np.float32)
        pair = np.asarray(pair, dtype=np.float32)
        v_values = np.asarray(v_values, dtype=np.float32)
        if x294.shape != (294,) or pair.shape != (2, 2) or v_values.ndim != 1:
            raise RuntimeError("G10.2 local score input shape invalid")
        nominal, second = pair
        if np.any(np.abs(v_values) > 1+1e-6):
            raise RuntimeError("G10.2 normalized translation out of range")
        delta = (v_values-nominal[0]).astype(np.float32)
        matrix = np.c_[np.broadcast_to(x294, (len(delta), 294)),
                       np.full(len(delta), nominal[0], dtype=np.float32), delta]
        q = np.stack([predict(net, matrix, self.mean, self.std)
                      for net in self.nets], axis=1).astype(np.float64)
        q[:, :, 2] += self.corrections[None, :]
        nominal_mask = v_values == nominal[0]
        q[nominal_mask] = 0
        nominal_q95 = self.nominal_model.scores(x294, nominal[None, :], second)[0]
        base = float(np.max(nominal_q95))
        conservative_delta = np.max(q[:, :, 2], axis=1)
        spread = np.max(q[:, :, 2], axis=1)-np.min(q[:, :, 2], axis=1)
        absolute = base+conservative_delta
        force_safe = absolute <= F_LIM
        uncertainty_ok = spread <= self.spread_threshold
        predicted_safe = force_safe & uncertainty_ok
        return {"nominal_Q95_replicas_N": nominal_q95,
            "nominal_conservative_Q95_N": base,
            "delta_quantiles_replicas_N": q,
            "predicted_absolute_R2_N": absolute,
            "corrected_Q95_spread_N": spread,
            "force_constraint_pass": force_safe,
            "uncertainty_constraint_pass": uncertainty_ok,
            "predicted_safe": predicted_safe}


def closest_safe(vs, mask, v_nom):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    return int(min(indices, key=lambda j: (abs(float(vs[j])-float(v_nom)), float(vs[j]))))
