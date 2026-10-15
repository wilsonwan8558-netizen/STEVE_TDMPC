"""Matched held-out G12 versus G13 one-step force-head evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402
from g12_force_head import ForceHead  # noqa: E402
from g13_model import G13Agent  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G12 = RL / "logs/world_model_force_aware_planning_g12"
F_LIM = 3.685331040756903


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(arm: str, region: str, truth: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - truth
    high = truth > F_LIM
    low = ~high
    return {
        "arm": arm, "region": region, "count": len(truth),
        "true_above_F_lim_count": int(high.sum()),
        "MAE_N": float(np.mean(np.abs(error))),
        "RMSE_N": float(np.sqrt(np.mean(error**2))),
        "mean_error_pred_minus_true_N": float(np.mean(error)),
        "false_safe_count": int(np.count_nonzero((prediction <= F_LIM) & high)),
        "false_safe_rate_given_true_high":
            float(np.mean(prediction[high] <= F_LIM)) if high.any() else "",
        "above_limit_recall":
            float(np.mean(prediction[high] > F_LIM)) if high.any() else "",
        "false_warning_count": int(np.count_nonzero((prediction > F_LIM) & low)),
        "false_warning_rate_given_true_safe":
            float(np.mean(prediction[low] > F_LIM)) if low.any() else "",
    }


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 head evaluation requires CUDA_VISIBLE_DEVICES=0")
    selected = json.loads((OUT / "checkpoint_selection.json").read_text())
    selected_path = Path(selected["selected_checkpoint"])
    if sha256(selected_path) != selected["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected G13 checkpoint identity changed")
    manifest = json.loads((OUT / "g13_heldout_geometry_manifest.json").read_text())
    data_path = OUT / "g13_heldout_geometry_transitions.npz"
    if sha256(data_path) != manifest["dataset_sha256"]:
        raise RuntimeError("G13 held-out geometry dataset identity changed")
    with np.load(data_path, allow_pickle=False) as data:
        new_data = {key: data[key] for key in data.files}
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as data:
        old_data = {key: data[key] for key in data.files}
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    mask = np.isin(old_data["seed"], split["heldout_test_seeds"])
    for key in ("seed", "decision_index", "action"):
        if not np.array_equal(new_data[key], old_data[key][mask]):
            raise RuntimeError(f"Matched held-out G12/G13 transition alignment failed: {key}")
    if not np.array_equal(new_data["force_next_max_N"],
                          old_data["force_next_max_N"][mask].astype(np.float32)):
        raise RuntimeError("Matched held-out force labels differ beyond declared float32 storage rounding")
    checkpoint = load_torch_checkpoint(selected_path, map_location="cpu")
    config = checkpoint["config"]
    new_agent = G13Agent(22, 2, build_agent_config(config),
                         episode_length=int(config["environment"]["max_episode_steps"]),
                         device=torch.device("cuda:0"))
    new_agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    new_agent.model.eval()
    del checkpoint
    state = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    old_head = ForceHead(state["state_dict"]["input_mean"],
                         state["state_dict"]["input_std"]).to("cuda:0")
    old_head.load_state_dict(state["state_dict"])
    old_head.eval()
    new_pred_parts = []
    old_pred_parts = []
    old_latent = old_data["latent"][mask]
    actions = new_data["action"]
    with torch.inference_mode():
        for start in range(0, len(actions), 512):
            end = min(len(actions), start + 512)
            o = torch.from_numpy(new_data["observation_geo"][start:end]).to("cuda:0")
            a = torch.from_numpy(actions[start:end]).to("cuda:0")
            z_next = new_agent.model.next(new_agent.model.encode(o), a)
            new_pred_parts.append(new_agent.model.force(z_next).squeeze(-1).cpu().numpy())
            z_old = torch.from_numpy(old_latent[start:end]).to("cuda:0")
            old_pred_parts.append(old_head(z_old, a).squeeze(-1).cpu().numpy())
    # G12 archived nominal-N labels retain float64 CSV precision. The G13
    # geometry cache stored exactly their float32 casts; use the source
    # float64 values for both arms' metrics after the exact cast audit above.
    truth = old_data["force_next_max_N"][mask].astype(np.float64)
    new_pred = np.concatenate(new_pred_parts).astype(np.float64)
    old_pred = np.concatenate(old_pred_parts).astype(np.float64)
    regions = {
        "all_heldout_steps": np.ones(len(truth), dtype=bool),
        "no_contact_F_eq_0": truth == 0,
        "low_0_lt_F_lt_0p5F_lim": (truth > 0) & (truth < 0.5 * F_LIM),
        "near_0p5F_lim_to_F_lim": (truth >= 0.5 * F_LIM) & (truth <= F_LIM),
        "near_boundary_0p8_to_1p2F_lim": (truth >= 0.8 * F_LIM) & (truth <= 1.2 * F_LIM),
        "above_F_lim": truth > F_LIM,
    }
    rows = [metrics(arm, name, truth[mask_region], prediction[mask_region])
            for arm, prediction in (("old_G12_posthoc", old_pred), ("G13_joint", new_pred))
            for name, mask_region in regions.items()]
    old_all = next(row for row in rows if row["arm"] == "old_G12_posthoc" and row["region"] == "all_heldout_steps")
    if (old_all["false_safe_count"] != 45 or old_all["true_above_F_lim_count"] != 105
            or abs(old_all["MAE_N"] - 0.455) > 0.002):
        raise RuntimeError(f"Old G12 head reference not reproduced: {old_all}")
    metrics_path = OUT / "force_head_metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_step_path = OUT / "force_head_heldout_predictions.csv"
    with per_step_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "decision_index", "true_force_N", "old_G12_pred_N", "G13_pred_N"))
        for seed, decision, actual, old, new in zip(new_data["seed"], new_data["decision_index"],
                                                     truth, old_pred, new_pred):
            writer.writerow((int(seed), int(decision), actual, old, new))
    new_all = next(row for row in rows if row["arm"] == "G13_joint" and row["region"] == "all_heldout_steps")
    comparison = {
        "status": "MATCHED_G12_HELDOUT_COHORT_ONE_STEP_EVALUATED",
        "heldout_seed_count": len(set(new_data["seed"].tolist())),
        "heldout_transition_count": len(truth),
        "F_lim_N": F_LIM,
        "old_G12": old_all, "G13_joint": new_all,
        "near_limit_old_G12": next(row for row in rows if row["arm"] == "old_G12_posthoc" and row["region"] == "near_0p5F_lim_to_F_lim"),
        "near_limit_G13": next(row for row in rows if row["arm"] == "G13_joint" and row["region"] == "near_0p5F_lim_to_F_lim"),
        "near_boundary_old_G12": next(row for row in rows if row["arm"] == "old_G12_posthoc" and row["region"] == "near_boundary_0p8_to_1p2F_lim"),
        "near_boundary_G13": next(row for row in rows if row["arm"] == "G13_joint" and row["region"] == "near_boundary_0p8_to_1p2F_lim"),
        "comparison_semantics": "same 30 held-out G12 seeds and same true G2 transitions, old cached latents versus G13 geometry observation and predicted post-transition latent",
        "force_label_storage_audit": "G13 geometry cache labels equal exact float32 casts of G12 float64 archived labels; maximum storage-rounding difference below 5e-7 N; shared float64 G12 source labels used for both metric arms",
        "selected_G13_checkpoint_sha256": selected["selected_checkpoint_sha256"],
        "old_G12_force_head_sha256": sha256(G12 / "force_head.pt"),
    }
    (OUT / "old_vs_g13_force_head.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_HELDOUT_HEAD_METRICS", old_all, new_all, flush=True)


if __name__ == "__main__":
    main()
