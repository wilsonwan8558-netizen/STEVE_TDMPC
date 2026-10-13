"""Predeclare G10 offline action-repair protocol, without opening online seeds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g95", "generic_solver_g96", "generic_solver_g9", "generic_solver_g8"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g95_common import OUT as G95, G9, SEEDS, F_LIM, source_gate, json_new, sha  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g10_force_constrained_minimal_repair"
G8 = RL / "logs/generic_solver_phase_g8_offline_quantile_shield"
G85 = RL / "logs/generic_solver_phase_g85_continuous_oracle_feasibility"
G96 = RL / "logs/generic_solver_phase_g96_stepwise_h2_force_profile"
G10_CONFIRM = tuple(range(107000, 107200))
GRID_AXIS = np.linspace(-1, 1, 41, dtype=np.float32)
REFINE_RADII = (0.025, 0.005, 0.001)
ORACLE_GRID_AXIS = np.linspace(-1, 1, 21, dtype=np.float32)


def gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or
            not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G10 frozen RTX 5090 CUDA_VISIBLE_DEVICES=0 required; no CPU fallback")
    source = source_gate()
    manifest = json.loads((G95 / "model_training_manifest.json").read_text())
    if (manifest["status"] != "THREE_FIXED_REPLICAS_PER_MODEL_BEFORE_DEVELOPMENT_TEST" or
            manifest["training_seeds"] != list(SEEDS) or
            manifest["G95_planner_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz") or
            manifest["G9_risk_normalization_sha256"] != sha(G9 / "models/risk/E1/H2/normalization.json")):
        raise RuntimeError("G10 frozen G9.5 scalar planner-conditioned predictor contract changed")
    prior = json.loads((G96 / "pre_confirmation_freeze_v2.json").read_text())
    if (prior["source_sha256"]["g96_protocol_v2.py"] !=
            sha(RL / "experiments/generic_solver_g96/g96_protocol_v2.py") or
            prior["controller_checkpoint_sha256"] != source["checkpoint_sha256"] or
            prior["G95_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz")):
        raise RuntimeError("G10 G9.6-v2 executable optimized-mean protocol changed")
    return source, manifest


def offline_states():
    with np.load(G95 / "development_planner_dataset.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    mask = ((d["group_index"] == 2) & (d["candidate_id"] == 8) &
            (d["R2_true_N"] > F_LIM))
    rows = []
    for index in np.flatnonzero(mask):
        group_sid = int(d["state_id_within_group"][index])
        path = G95 / "development_states" / f"state_development_test_{group_sid:04d}.json"
        item = json.loads(path.read_text())
        pair = np.asarray(item["plans"][8]["actions"], dtype=np.float32)
        if (pair.shape != (2, 2) or not np.array_equal(pair, d["action_pair"][index]) or
                not np.array_equal(pair[0], np.asarray(item["planner_mean_action"], dtype=np.float32)) or
                (item["seed"], item["decision_index"]) !=
                (int(d["seed"][index]), int(d["decision_index"][index]))):
            raise RuntimeError("G10 archived G9.5 optimized mean/action state alignment changed")
        rows.append({"cohort": "heldout_development_unsafe_nominal", "state_id": group_sid,
            "seed": int(d["seed"][index]), "decision_index": int(d["decision_index"][index]),
            "nominal_pair": pair.tolist(), "archived_nominal_F1_N": float(d["F1_true_N"][index]),
            "archived_nominal_F2_N": float(d["F2_true_N"][index]),
            "archived_nominal_R2_N": float(d["R2_true_N"][index]),
            "source_state_sha256": sha(path), "G95_planner_row": int(index)})
    if len(rows) != 18 or len({(r["seed"], r["decision_index"]) for r in rows}) != 18:
        raise RuntimeError("G10 expected exact 18 held-out unsafe G9.5 nominal states")
    difficult = json.loads((G9 / "difficult_event_manifest.json").read_text())["state_ids"]
    old = json.loads((G85 / "recovered_g8_state_manifest.json").read_text())
    by_id = {int(row["state_id"]): row for row in old["states"]}
    if len(difficult) != 11 or any(sid not in by_id for sid in difficult):
        raise RuntimeError("G10 difficult G8.5 event identity changed")
    for sid in difficult:
        state = by_id[sid]
        for lookback in (1, 2):
            rows.append({"cohort": "difficult_event_early_intervention", "state_id": int(sid),
                "seed": int(state["episode_seed"]),
                "decision_index": int(state["decision_index"])-lookback,
                "event_decision_index": int(state["decision_index"]),
                "lookback_steps": lookback,
                "nominal_pair": None,
                "archived_nominal_F1_N": None,
                "archived_nominal_F2_N": None,
                "archived_nominal_R2_N": None,
                "source_state_sha256": state["archived_state_sha256"]})
    return rows


def prepare():
    source, manifest = gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    states = offline_states()
    OUT.mkdir(parents=True)
    json_new(OUT / "metadata.json", {"phase": "G10", "branch": "feature/generic-contact-force-safety",
        "status": "OFFLINE_PROTOCOL_PREDECLARED;_107000_SERIES_UNOPENED",
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0),
        "controller_checkpoint_sha256": source["checkpoint_sha256"],
        "G95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "G96_executable_mean_source_sha256": sha(RL / "experiments/generic_solver_g96/g96_protocol_v2.py"),
        "F_lim_N": F_LIM,
        "F_lim_interpretation": "Generic-simulation statistical high-force reference, not a clinical injury threshold",
        "online_confirmation_seeds_unopened": list(G10_CONFIRM)})
    model = manifest["models"]["M1_planner_conditioned"]
    json_new(OUT / "frozen_predictor_definition.json", {
        "family": "G9.5_M1_planner_conditioned", "training_seeds": list(SEEDS),
        "replica_checkpoint_sha256": {s: model[str(s)]["checkpoint_sha256"] for s in SEEDS},
        "replica_validation_only_Q95_delta_N": {s: model[str(s)]["validation_Q95_delta_N"] for s in SEEDS},
        "normalization_sha256": sha(G9 / "models/risk/E1/H2/normalization.json"),
        "E1_mapper_sha256": source["g9_E1_mapper_sha256"],
        "E1_mapper_normalization_sha256": source["g9_E1_mapper_normalization_sha256"],
        "input": "256D frozen TD-MPC2 current-observation latent + 38D E1 estimated pre-action physical state + candidate (a_t,a_nom_t_plus_1), each 2D normalized",
        "input_dimension": 298, "quantiles": [.5,.9,.95,.99],
        "network": "64x64 ReLU cumulative-softplus noncrossing quantile net",
        "target": "R2=max(F_t_plus_1,F_t_plus_2), true Generic-SOFA two complete RL intervals",
        "output": "each replica's Q95(R2) plus its frozen nonnegative validation-only correction in nominal N",
        "multi_replica_rule": "predicted-safe iff all three frozen corrected Q95 values <= F_lim; no best-seed selection",
        "training_or_recalibration_in_G10": False})
    json_new(OUT / "repair_problem_definition.json", {
        "objective": "minimize squared Euclidean distance ||a-a_nom||_2^2 in normalized action space",
        "constraint": "all three frozen corrected Q95(R2 | x_t,a,a_nom(t+1)) <= F_lim",
        "action_bounds": [-1.,1.], "action_dimension": 2, "H": 2, "F_lim_N": F_LIM,
        "second_action": "frozen clipped executable TD-MPC2 optimized-mean second action; not optimized or replanned inside fork",
        "only_current_action_repaired": True,
        "no_safe_found": "explicit INFEASIBLE; no risky substitute is silently returned",
        "deviation_report": "Euclidean normalized-action distance and separate translation/rotation changes"})
    json_new(OUT / "action_search_protocol.json", {
        "status": "FROZEN_BEFORE_OFFLINE_TRUE_SOFA_LABELS",
        "stage_A": "nominal exactly, then 41x41 float32 uniform grid over [-1,1]^2; all three corrected Q95 replicas evaluated in batches",
        "grid_axis": GRID_AXIS.tolist(), "grid_action_count": len(GRID_AXIS)**2,
        "stage_B": "if grid has a predicted-safe point, refine around current nearest-safe action with 5x5 offsets at radii 0.025, 0.005, 0.001; each radius evaluates 25 clipped float32 actions, updates nearest-safe point; duplicate realized actions omitted",
        "refinement_radii": list(REFINE_RADII), "local_offsets_per_radius": [-1.,-.5,0.,.5,1.],
        "distance_tie_break": "Euclidean distance then float32 translation then float32 rotation",
        "stopping": "nominal already predicted safe => no repair; otherwise after three fixed refinement scales; no global grid-safe point => INFEASIBLE",
        "nominal_unsafe_but_predicted_safe": "not-triggered false-safe nominal pass-through, tracked separately",
        "predictor_evaluations_max_per_state": 1+len(GRID_AXIS)**2+3*25,
        "true_SOFA_oracle": "21x21 uniform float32 grid, then 5x5 local refinement around closest true-safe grid point at same three radii; finite-search closest-safe benchmark, not proof of global optimum/infeasibility",
        "oracle_grid_axis": ORACLE_GRID_AXIS.tolist(),
        "oracle_rule": "R2<=F_lim only if both intervals complete; censoring not called safe",
        "offline_cohorts": "all 18 held-out G9.5 development-test truly unsafe optimized-mean plans plus all 11 G8.5 difficult events at t*-1 and t*-2",
        "offline_gate": "judge true-SOFA rescue, false-safe repairs, oracle-recoverable misses, and action deviation together; no numeric pass threshold",
        "no_online_seeds_opened": True})
    json_new(OUT / "fallback_definition.json", {
        "action": [0.,0.], "semantics": "zero normalized translation and zero normalized rotation",
        "use": "predeclared conservative candidate only if an offline GO permits online testing and separate true-SOFA fallback-force evaluation passes",
        "not_executed_online_until_offline_gate": True,
        "not_assumed_safe": True})
    json_new(OUT / "offline_state_manifest.json", {"status": "SELECTED_BEFORE_G10_REPAIR_ORACLE_OUTCOMES",
        "states": states, "heldout_unsafe_nominal_count": 18,
        "difficult_event_parent_state_count": 22,
        "development_state_selection_uses_archived_nominal_truth_only": True,
        "confirmation_107000_series_accessed": False})
    print("G10 prepared 18 held-out unsafe nominal states and 22 difficult-event prior states; 107000 unopened", flush=True)


if __name__ == "__main__":
    prepare()
