"""Seal local model/protocol after development and before opening 108000 seeds."""

from __future__ import annotations

import json
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g102_prepare import OUT, SEEDS, F_LIM, verify_source  # noqa: E402
from g95_common import json_new, sha  # noqa: E402


def freeze():
    verify_source()
    path = OUT / "pre_confirmation_freeze.json"
    if path.exists():
        raise FileExistsError(path)
    confirm = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if confirm["status"] != "SEALED_NOT_OPENED_BEFORE_PRE_CONFIRMATION_FREEZE":
        raise RuntimeError("G10.2 confirmation already opened or seed protocol changed")
    model = json.loads((OUT / "model_training_manifest.json").read_text())
    if model["training_seeds"] != list(SEEDS) or model["M_LOCAL_ABS_trained"]:
        raise RuntimeError("G10.2 model/replica contract changed")
    if model["training_code_sha256"] != sha(HERE / "g102_train.py"):
        raise RuntimeError("G10.2 local model training implementation changed")
    for name in ("development_response_metrics.csv", "development_boundary_metrics.csv",
                 "development_repair_results.csv", "false_infeasible_revisit.csv",
                 "difficult_event_results.csv", "training_seed_robustness.json",
                 "G101_vs_G102_comparison.json"):
        if not (OUT / name).is_file():
            raise RuntimeError(f"G10.2 development incomplete: {name}")
    files = ("metadata.json", "local_dataset_manifest.json", "local_dataset_split.json",
        "local_training_range.json", "model_definition.json", "abstention_rule.json",
        "local_dataset_arrays.npz", "local_dataset_summary.json", "normalization.json",
        "model_training_manifest.json", "validation_ensemble_rule.json",
        "expansion_parent_replay_audit.json", "development_event_feature_replay_audit.json",
        "confirmation_seed_manifest.json", "development_response_metrics.csv",
        "development_boundary_metrics.csv", "development_repair_results.csv",
        "false_infeasible_revisit.csv", "difficult_event_results.csv",
        "training_seed_robustness.json", "G101_vs_G102_comparison.json")
    hashes = {name: sha(OUT / name) for name in files}
    event_feature_hashes = {path.name: sha(path) for path in sorted(
        (OUT / "development_event_features").glob("state_*.json"))}
    if len(event_feature_hashes) != 22:
        raise RuntimeError("G10.2 22 exact event feature artifacts required before freeze")
    code_hashes = {name: sha(HERE / name) for name in (
        "g102_prepare.py", "g102_train.py", "g102_eval_common.py",
        "g102_development.py", "g102_confirm.py", "g102_confirm_analysis.py")}
    for replica_index, seed in enumerate(SEEDS, start=1):
        for name in ("model.pt", "selection.json"):
            rel = f"models/delta_seed_{replica_index}/{name}"
            hashes[rel] = sha(OUT / rel)
    json_new(path, {"status": "FROZEN_BEFORE_UNTOUCHED_108000_CONFIRMATION",
        "frozen_output_file_sha256": hashes,
        "development_event_feature_sha256": event_feature_hashes,
        "frozen_code_sha256": code_hashes,
        "training_seeds": list(SEEDS), "model_checkpoint_sha256": {
            str(seed): hashes[f"models/delta_seed_{replica_index}/model.pt"]
            for replica_index, seed in enumerate(SEEDS, start=1)},
        "normalization_sha256": hashes["normalization.json"],
        "uncertainty_abstention_rule_sha256": hashes["validation_ensemble_rule.json"],
        "local_training_range_sha256": hashes["local_training_range.json"],
        "model_definition_sha256": hashes["model_definition.json"],
        "confirmation_seed_manifest_sha256": hashes["confirmation_seed_manifest.json"],
        "F_lim_N": F_LIM, "H": 2,
        "rotation": "exact nominal optimized-mean w_nom, never optimized",
        "second_action": "exact fixed nominal optimized-mean second action",
        "search_grid": "frozen 101 uniform float32 points plus nominal and clipped +/-0.05,0.10,0.20",
        "M_LOCAL_ABS_trained": False,
        "confirmation_seeds_opened_before_freeze": False,
        "107000_107199_online_seeds_opened": False})
    print("G10.2 pre-confirmation freeze complete; 108000-series may now be opened", flush=True)


if __name__ == "__main__":
    freeze()
