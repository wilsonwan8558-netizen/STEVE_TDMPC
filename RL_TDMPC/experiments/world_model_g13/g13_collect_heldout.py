"""Replay G2 once to attach current geometry to G12's held-out transitions."""

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
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
from tdmpc2.common import set_seed  # noqa: E402
from collect import check_scene  # noqa: E402
from g13_env import make_g13_env  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G12 = RL / "logs/world_model_force_aware_planning_g12"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"
SEEDS = tuple(range(90000, 90200))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 held-out replay requires CUDA_VISIBLE_DEVICES=0")
    source = json.loads((G12 / "force_dataset_manifest.json").read_text())
    if sha256(G2) != source["source_sha256"][str(G2.resolve())]:
        raise RuntimeError("G2 archived transition source SHA changed")
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    heldout = set(int(seed) for seed in split["heldout_test_seeds"])
    if len(heldout) != 30 or not heldout.issubset(SEEDS):
        raise RuntimeError("G12 held-out seed set changed")
    by_seed = {seed: [] for seed in SEEDS}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in by_seed:
                by_seed[seed].append(row)
    if any(not by_seed[seed] for seed in SEEDS):
        raise RuntimeError("Archived G2 seed rows incomplete")
    dataset_path = OUT / "g13_heldout_geometry_transitions.npz"
    manifest_path = OUT / "g13_heldout_geometry_manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError("Held-out geometry dataset already collected")
    set_seed(SEEDS[0])
    env = make_g13_env()
    fields = {key: [] for key in ("seed", "decision_index", "observation_geo",
                                  "action", "force_next_max_N", "reward",
                                  "terminated", "truncated")}
    exact_steps = 0
    max_force_diff = 0.0
    try:
        for seed in SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env.env)
            for decision, ref in enumerate(by_seed[seed]):
                expected_obs = np.asarray(json.loads(ref["decision_observation_json"]), dtype=np.float32)
                if not np.array_equal(observation[:14], expected_obs):
                    raise RuntimeError(f"G13 held-out parent observation mismatch {seed}/{decision}")
                action = np.asarray(json.loads(ref["selected_action_json"]), dtype=np.float32)
                if seed in heldout:
                    fields["seed"].append(seed)
                    fields["decision_index"].append(decision)
                    fields["observation_geo"].append(observation.copy())
                    fields["action"].append(action.copy())
                next_observation, reward, terminated, truncated, info = env.step(action)
                true_force = float(info["distal_contact_force_max_N"])
                expected_force = float(ref["distal_contact_force_max_N"])
                max_force_diff = max(max_force_diff, abs(true_force - expected_force))
                if (not np.array_equal(next_observation[:14],
                                       np.asarray(json.loads(ref["next_observation_json"]), dtype=np.float32))
                        or abs(float(reward) - float(ref["reward"])) > 1e-9
                        or true_force != expected_force
                        or bool(terminated) != (ref["terminated"] == "True")
                        or bool(truncated) != (ref["truncated"] == "True")
                        or not np.array_equal(np.asarray(info["raw_action"], dtype=np.float32),
                                              np.asarray(json.loads(ref["applied_action_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G13 held-out replay transition mismatch {seed}/{decision}")
                if seed in heldout:
                    fields["force_next_max_N"].append(true_force)
                    fields["reward"].append(float(reward))
                    fields["terminated"].append(bool(terminated))
                    fields["truncated"].append(bool(truncated))
                exact_steps += 1
                observation = next_observation
            print("G13_HELDOUT_REPLAY", seed, len(by_seed[seed]), flush=True)
    finally:
        env.close()
    arrays = {
        "seed": np.asarray(fields["seed"], dtype=np.int32),
        "decision_index": np.asarray(fields["decision_index"], dtype=np.int16),
        "observation_geo": np.asarray(fields["observation_geo"], dtype=np.float32),
        "action": np.asarray(fields["action"], dtype=np.float32),
        "force_next_max_N": np.asarray(fields["force_next_max_N"], dtype=np.float32),
        "reward": np.asarray(fields["reward"], dtype=np.float32),
        "terminated": np.asarray(fields["terminated"], dtype=np.bool_),
        "truncated": np.asarray(fields["truncated"], dtype=np.bool_),
    }
    if arrays["observation_geo"].shape != (len(arrays["seed"]), 22):
        raise RuntimeError("G13 held-out geometry output shape invalid")
    np.savez_compressed(dataset_path, **arrays)
    manifest = {
        "status": "PASS_EXACT_G2_REPLAY_AND_MATCHED_G12_HELDOUT_COHORT",
        "source_G2": str(G2.resolve()), "source_G2_sha256": sha256(G2),
        "source_G12_split": str((G12 / "force_dataset_split.json").resolve()),
        "source_G12_split_sha256": sha256(G12 / "force_dataset_split.json"),
        "heldout_seeds": sorted(heldout), "heldout_rows": len(arrays["seed"]),
        "replayed_seeds": [SEEDS[0], SEEDS[-1]], "exact_replayed_steps": exact_steps,
        "max_true_force_difference_N": max_force_diff,
        "dataset": str(dataset_path.resolve()), "dataset_sha256": sha256(dataset_path),
        "scene_lifecycle": "one scene, ascending seed resets, all 90000-90199 preceding histories replayed",
        "label": "next RL interval max-substep distal force, nominal N",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_HELDOUT_GEOMETRY_COMPLETE", len(arrays["seed"]), exact_steps, flush=True)


if __name__ == "__main__":
    main()
