"""Held-out one-step and actual-world-model rollout-depth force diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from g12_force_head import ForceHead
from g12_train import OUT, CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, sha, new_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402

HORIZON = 3


def describe(name: str, depth: int, actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) != len(predicted) or not len(actual):
        raise RuntimeError(f"G12 empty/misaligned metric group {name}/{depth}")
    error = predicted - actual
    high = actual > F_LIM
    low = ~high
    return {
        "group": name, "rollout_depth": depth, "count": len(actual),
        "true_above_F_lim_count": int(high.sum()),
        "true_safe_count": int(low.sum()),
        "MAE_N": float(np.mean(np.abs(error))),
        "RMSE_N": float(np.sqrt(np.mean(error**2))),
        "mean_error_pred_minus_true_N": float(np.mean(error)),
        "false_safe_count": int(np.count_nonzero((predicted <= F_LIM) & high)),
        "false_safe_rate_given_true_high":
            float(np.mean(predicted[high] <= F_LIM)) if high.any() else None,
        "false_warning_count": int(np.count_nonzero((predicted > F_LIM) & low)),
        "false_warning_rate_given_true_safe":
            float(np.mean(predicted[low] > F_LIM)) if low.any() else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12 head evaluation requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    manifest = json.loads((OUT / "force_dataset_manifest.json").read_text())
    split = json.loads((OUT / "force_dataset_split.json").read_text())
    definition = json.loads((OUT / "force_head_definition.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(OUT / "force_transitions.npz") != manifest["joined_dataset_sha256"] or
            sha(OUT / "force_head.pt") != definition["force_head_checkpoint_sha256"]):
        raise RuntimeError("G12 frozen checkpoint/data/head identity changed")
    with np.load(OUT / "force_transitions.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    mask = np.isin(arrays["seed"], split["heldout_test_seeds"])
    if len(set(arrays["seed"][mask].tolist())) != 30:
        raise RuntimeError("G12 held-out episode count changed")
    force_state = torch.load(OUT / "force_head.pt", map_location="cpu", weights_only=True)
    if force_state["tdmpc2_checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("G12 force head was not fit to the frozen checkpoint")
    device = torch.device("cuda:0")
    state = force_state["state_dict"]
    head = ForceHead(state["input_mean"], state["input_std"]).to(device)
    head.load_state_dict(state)
    head.eval()
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=device)
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    agent.model.eval()
    if any(p.requires_grad for p in head.parameters()) or any(p.requires_grad for p in agent.model.parameters()):
        # Evaluation does not use backward; explicit freezing makes accidental
        # optimization in later G12 diagnostics impossible.
        head.requires_grad_(False)
        agent.model.requires_grad_(False)

    with torch.inference_mode():
        z = torch.from_numpy(arrays["latent"][mask]).to(device)
        action = torch.from_numpy(arrays["action"][mask]).to(device)
        one_step = head(z, action).squeeze(-1).cpu().numpy().astype(np.float64)
    truth = arrays["force_next_max_N"][mask].astype(np.float64)
    low_boundary = 0.5 * F_LIM
    regions = {
        "all_heldout_steps": np.ones(len(truth), dtype=bool),
        "no_contact_F_eq_0": truth == 0,
        "low_0_lt_F_lt_0p5F_lim": (truth > 0) & (truth < low_boundary),
        "near_0p5F_lim_to_F_lim": (truth >= low_boundary) & (truth <= F_LIM),
        "above_F_lim": truth > F_LIM,
    }
    rows = [describe(name, 1, truth[region], one_step[region])
            for name, region in regions.items()]
    if sum(len(truth[regions[name]]) for name in list(regions)[1:]) != len(truth):
        raise RuntimeError("G12 force regions do not partition held-out rows")

    # The actual planner dynamics, not a surrogate autoregressive model, rolls
    # forward the same latent z and executed action sequence. Only z_0 comes
    # from a real pre-action observation; no later observation is encoded.
    start_indices = np.flatnonzero(mask)
    windows = [i for i in start_indices if i + HORIZON <= len(arrays["seed"])
               and np.all(mask[i:i+HORIZON])
               and np.all(arrays["seed"][i:i+HORIZON] == arrays["seed"][i])
               and np.array_equal(arrays["decision_index"][i:i+HORIZON],
                                  np.arange(arrays["decision_index"][i],
                                            arrays["decision_index"][i] + HORIZON))]
    if not windows:
        raise RuntimeError("G12 no held-out three-step rollout windows")
    predicted = [[] for _ in range(HORIZON)]
    actual = [[] for _ in range(HORIZON)]
    batch_size = 512
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = np.asarray(windows[start:start+batch_size], dtype=np.int64)
            latent = torch.from_numpy(arrays["latent"][batch]).to(device)
            for k in range(HORIZON):
                a = torch.from_numpy(arrays["action"][batch+k]).to(device)
                predicted[k].append(head(latent, a).squeeze(-1).cpu().numpy().astype(np.float64))
                actual[k].append(arrays["force_next_max_N"][batch+k].astype(np.float64))
                latent = agent.model.next(latent, a)
    depth_rows = [describe("heldout_behavior_action_rollout", k+1,
                           np.concatenate(actual[k]), np.concatenate(predicted[k]))
                  for k in range(HORIZON)]
    if not np.allclose(depth_rows[0]["MAE_N"], describe(
            "windows_depth_1", 1,
            arrays["force_next_max_N"][windows].astype(np.float64),
            np.concatenate(predicted[0]))["MAE_N"], atol=0, rtol=0):
        raise RuntimeError("G12 first-depth indexing changed")
    write_csv(OUT / "force_head_metrics.csv", rows)
    write_csv(OUT / "rollout_depth_metrics.csv", depth_rows)
    new_json(OUT / "force_head_evaluation_manifest.json", {
        "status": "HELDOUT_ONE_STEP_AND_FROZEN_WORLD_MODEL_ROLLOUT_EVALUATED",
        "heldout_episode_count": 30,
        "heldout_step_count": len(truth),
        "three_step_window_count": len(windows),
        "one_step_prediction": "head(z_t,a_t) on cached single-observation CUDA latents",
        "multi_step_prediction": "head(z_hat_(t+k),a_(t+k)); z_hat_(t+k+1)=frozen WorldModel.next(z_hat_(t+k),a_(t+k))",
        "future_action_source": "actual held-out behavior action sequence; no future observation re-encoding",
        "region_boundaries": "no contact F=0; low 0<F<0.5 F_lim; near 0.5 F_lim<=F<=F_lim; above F>F_lim",
        "force_head_checkpoint_sha256": sha(OUT / "force_head.pt"),
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    })
    print("G12_HELDOUT_FORCE_HEAD_EVALUATED", rows[0], depth_rows, flush=True)


if __name__ == "__main__":
    main()
