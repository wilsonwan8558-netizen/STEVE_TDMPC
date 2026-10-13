"""Independent G8.5 event lead-time diagnostics using archived behavior actions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from g9_estimate import Mapper, Transition, ukf_episode  # noqa: E402
from g9_risk import OUT, ESTIMATORS, HORIZONS, F_LIM, cuda_gate, load_model, predict, sha  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402
from prepare import G1B, CONFIG, EXPECTED_CHECKPOINT_SHA  # noqa: E402

G8 = RL / "logs/generic_solver_phase_g8_offline_quantile_shield"
G85 = RL / "logs/generic_solver_phase_g85_continuous_oracle_feasibility"


def csv_new(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archived():
    audit = json.loads((G85 / "search_parent_audit.json").read_text())
    if audit["status"] != "PASS" or audit["parent_steps_compared"] != 1848:
        raise RuntimeError("G8.5 replay source invalid")
    by_seed = {}
    with (G8 / "confirmation_behavior_steps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            by_seed.setdefault(int(row["seed"]), []).append(row)
    if sorted(by_seed) != list(range(101000, 101030)) or sum(map(len, by_seed.values())) != 1848:
        raise RuntimeError("G8 behavior source incomplete")
    for seed, episode in by_seed.items():
        if [int(r["step_index"]) for r in episode] != list(range(len(episode))):
            raise RuntimeError(f"G8 decision index changed {seed}")
    manifest = json.loads((OUT / "difficult_event_manifest.json").read_text())
    events = []
    for state_id in manifest["state_ids"]:
        state = json.loads((G8 / "confirmation_candidates" / f"state_{state_id:04d}.json").read_text())
        events.append(state)
    return by_seed, events


def encoder():
    path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(path) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen TD-MPC2 checkpoint changed")
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    config = checkpoint["config"]
    if config != yaml.safe_load(CONFIG.read_text()):
        raise RuntimeError("Frozen Generic config changed")
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                       episode_length=int(config["environment"]["max_episode_steps"]),
                       device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    return agent


def map_physical(name, features):
    folder = OUT / "models" / name
    norm = json.loads((folder / "normalization.json").read_text())
    x = (np.asarray(features, dtype=np.float64) - norm["input_mean"]) / norm["input_std"]
    model = Mapper(x.shape[1]).to("cuda:0")
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cuda:0", weights_only=True))
    model.eval()
    with torch.inference_mode():
        normalized = model(torch.as_tensor(x, device="cuda:0", dtype=torch.float32)).cpu().numpy()
    return (normalized * np.asarray(norm["target_std"]) + np.asarray(norm["target_mean"])).astype(np.float32)


def main():
    cuda_gate()
    if (OUT / "lead_time_analysis.csv").exists():
        raise FileExistsError(OUT / "lead_time_analysis.csv")
    selection = json.loads((OUT / "validation_horizon_selection.json").read_text())
    if selection["status"] != "VALIDATION_ONLY_SELECTION_BEFORE_DEVELOPMENT_TEST_AND_CONFIRMATION":
        raise RuntimeError("Validation selection missing")
    by_seed, events = archived()
    agent = encoder()
    transition = Transition()
    transition.load_state_dict(torch.load(OUT / "models/transition.pt", map_location="cpu", weights_only=True))
    transition.eval()
    filter_def = json.loads((OUT / "filter_definition.json").read_text())
    Q = np.asarray(filter_def["process_covariance_diagonal"])
    R = float(filter_def["measurement_covariance_diagonal"][0])
    # Encode only archived pre-action observations. UKF uses measurements/actions
    # through the current decision, resetting independently for every episode.
    cache = {}
    for seed, episode in by_seed.items():
        obs = np.asarray([json.loads(r["decision_observation_json"]) for r in episode], dtype=np.float32)
        act = np.asarray([json.loads(r["selected_action_json"]) for r in episode], dtype=np.float32)
        force = np.asarray([float(r["distal_contact_force_max_N"]) for r in episode])
        with torch.inference_mode():
            z = np.asarray([agent.model.encode(agent._tensor_observation(o))[0].cpu().numpy() for o in obs])
        ukf, _ = ukf_episode(transition, obs, act, Q, R)
        history = np.empty((len(obs), 46), dtype=np.float32)
        history[:] = np.nan
        for d in range(2, len(obs)):
            history[d] = np.r_[obs[d-2:d+1].reshape(-1), act[d-2:d].reshape(-1)]
        views = {"E0": obs, "E1": history[2:], "E2": ukf}
        estimates = {"E0": map_physical("E0", views["E0"]),
                     "E1": np.full((len(obs), 38), np.nan, dtype=np.float32),
                     "E2": map_physical("E2", views["E2"])}
        estimates["E1"][2:] = map_physical("E1", views["E1"])
        cache[seed] = {"observation": obs, "action": act, "force": force, "latent": z,
                       "estimate": estimates}
    for state in events:
        seed, d = int(state["seed"]), int(state["decision_index"])
        if not np.array_equal(cache[seed]["latent"][d], np.asarray(state["latent"], dtype=np.float32)):
            raise RuntimeError(f"G8 frozen latent mismatch {seed}/{d}")
        if abs(cache[seed]["force"][d] - float(state["candidates"][0]["force_next_max_N"])) > 1e-8:
            raise RuntimeError(f"G8 event force mismatch {seed}/{d}")
    np.savez_compressed(OUT / "g8_archived_current_features.npz",
                        seed=np.concatenate([np.full(len(cache[s]["force"]), s, dtype=np.int32) for s in sorted(cache)]),
                        decision_index=np.concatenate([np.arange(len(cache[s]["force"]), dtype=np.int32) for s in sorted(cache)]),
                        latent=np.concatenate([cache[s]["latent"] for s in sorted(cache)]),
                        E0=np.concatenate([cache[s]["estimate"]["E0"] for s in sorted(cache)]),
                        E1=np.concatenate([cache[s]["estimate"]["E1"] for s in sorted(cache)]),
                        E2=np.concatenate([cache[s]["estimate"]["E2"] for s in sorted(cache)]))
    output = []
    for state in events:
        sid, seed, onset = int(state["state_id"]), int(state["seed"]), int(state["decision_index"])
        episode = cache[seed]
        for name in ESTIMATORS:
            for horizon in HORIZONS:
                net, mean, scale, delta = load_model(name, horizon)
                entries = []
                for d in range(max(2, onset-5), onset+1):
                    if d + horizon > len(episode["force"]):
                        continue
                    sequence = episode["action"][d:d+horizon].reshape(-1)
                    x = np.r_[episode["latent"][d], episode["estimate"][name][d], sequence][None, :]
                    quantiles = predict(net, x, mean, scale)[0]
                    truth = float(np.max(episode["force"][d:d+horizon]))
                    entries.append({"decision_index": d, "lookback_steps": onset-d,
                                    "true_R_H_N": truth, "raw_Q95_N": float(quantiles[2]),
                                    "corrected_Q95_N": float(quantiles[2]+delta),
                                    "true_high": bool(truth > F_LIM),
                                    "warned_correctly": bool(truth > F_LIM and quantiles[2]+delta > F_LIM)})
                prior = [e for e in entries if e["lookback_steps"] > 0 and e["warned_correctly"]]
                earliest = max((e["lookback_steps"] for e in prior), default=None)
                output.append({"state_id": sid, "episode_seed": seed, "event_decision_index": onset,
                               "estimator": name, "horizon": horizon,
                               "event_nominal_force_N": float(episode["force"][onset]),
                               "true_high_prior_count": sum(e["lookback_steps"] > 0 and e["true_high"] for e in entries),
                               "correct_early_warning_count": len(prior),
                               "earliest_correct_warning_lead_steps": earliest,
                               "earliest_correct_warning_lead_s": None if earliest is None else earliest*.132,
                               "no_warning": int(earliest is None),
                               "prior_detail_json": json.dumps([e for e in entries if e["lookback_steps"] > 0], separators=(",", ":")),
                               "warning_protocol": "RETROSPECTIVE_REALIZED_BEHAVIOR_ACTIONS_NOT_PROSPECTIVE_PLAN"})
    csv_new(OUT / "lead_time_analysis.csv", output)
    print(f"G9 G8.5 event lead-time analysis: {len(events)} events, {len(output)} estimator-horizon rows", flush=True)


if __name__ == "__main__":
    main()
