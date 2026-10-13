"""Predeclare planner-conditioned G9.5 protocol before any SOFA plan labels."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from g95_common import (OUT, G2, G3, G9, RL, CONFIRM, F_LIM, H, TAUS,
                        PLANS_PER_STATE, SAMPLED_PER_STATE, SEEDS, source_gate,
                        sha, json_new)


def prepare():
    source = source_gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    split = json.loads((G3 / "development_split.json").read_text())
    states = json.loads((G3 / "development_state_selection.json").read_text())["states"]
    groups = ("train", "validation", "development_test")
    counts = {g: len(states[g]) for g in groups}
    if counts != {"train": 560, "validation": 120, "development_test": 120}:
        raise RuntimeError("G3 four-per-episode planner-state selection changed")
    if set(map(int, split["train"])) & set(map(int, split["validation"])) or set(map(int, split["train"])) & set(map(int, split["development_test"])) or set(map(int, split["validation"])) & set(map(int, split["development_test"])):
        raise RuntimeError("G9.5 episode groups overlap")
    for group in groups:
        selected = states[group]
        if len({(r["seed"], r["decision_index"]) for r in selected}) != counts[group]:
            raise RuntimeError(f"G9.5 {group} states duplicate")
        if sorted({r["seed"] for r in selected}) != sorted(split[group]):
            raise RuntimeError(f"G9.5 {group} seed split changed")
        if any(r["decision_index"] < 2 or r["decision_index"]+2 > r["episode_length"] for r in selected):
            raise RuntimeError(f"G9.5 {group} E1 history or R2 horizon undefined")
    if set(CONFIRM) & set(sum((split[g] for g in groups), [])):
        raise RuntimeError("G9.5 untouched confirmation overlaps development")
    g9_risk_norm = json.loads((G9 / "models/risk/E1/H2/normalization.json").read_text())
    g9_mapper_norm = json.loads((G9 / "models/E1/normalization.json").read_text())
    g9_sel = json.loads((G9 / "models/risk/E1/H2/selection.json").read_text())
    if (len(g9_risk_norm["mean"]) != 298 or len(g9_mapper_norm["input_mean"]) != 46 or
            len(g9_mapper_norm["target_mean"]) != 38 or
            int(g9_sel["training_seed"]) != SEEDS[0]):
        raise RuntimeError("Frozen G9 E1/H2 dimensions or training seed changed")
    OUT.mkdir(parents=True)
    (OUT / "development_states").mkdir()
    (OUT / "confirmation_states").mkdir()
    (OUT / "models/M0_behavior").mkdir(parents=True)
    (OUT / "models/M1_planner_conditioned").mkdir(parents=True)
    json_new(OUT / "metadata.json", {"phase": "G9.5", "branch": "feature/generic-contact-force-safety",
        "question": "Does exact planner-distribution supervision improve prospective H2 force-risk evaluation?",
        "frozen_sources": source, "F_lim_N": F_LIM, "horizon": H,
        "confirm_seeds_declared_but_unopened": CONFIRM,
        "training_seeds": SEEDS, "no_controller_solver_or_force_change": True,
        "g95_prepare_source_sha256": sha(__file__)})
    json_new(OUT / "g9_model_audit.json", {"status": "G9_E1_H2_EXACT_RECOVERY",
        "E1_current_state_input": "obs[d-2],obs[d-1],obs[d] (3x14) plus actions[d-2],actions[d-1] (2x2), in this order",
        "E1_mapper": "46-128-ReLU-128-ReLU-38, G9 TRAIN-only input/target normalization frozen",
        "risk_input": "frozen 256D current TD-MPC2 latent + 38D E1 estimated current physical state + original candidate [a_d,a_(d+1)] flattened in time order",
        "risk_input_dimension": 298, "risk_target": "R2=max(F1_true,F2_true) for two complete true Generic-SOFA RL intervals",
        "quantile_levels": TAUS, "model": "298-64-ReLU-64-ReLU-4 with cumulative softplus increments, noncrossing nonnegative quantiles",
        "optimizer": "Adam lr=0.001, batch=512, max 60 epochs, early-stop patience 8",
        "loss": "equal-weight four-quantile pinball",
        "normalization": "exact G9 E1/H2 risk input TRAIN-behavior mean/std, frozen for M0 and M1",
        "checkpoint_rule": "strict minimum VALIDATION four-quantile pinball improvement >=1e-5",
        "Q95_correction_rule": "nonnegative additive 95th percentile of VALIDATION residual y-Q95 using higher order statistic; no development-test or confirmation recalibration",
        "G9_original_seed": SEEDS[0], "G9_original_validation_Q95_delta_N": g9_sel["validation_only_Q95_delta_N"],
        "G9_original_training_code_sha256": sha(RL / "experiments/generic_solver_g9/g9_risk.py"),
        "G9_original_risk_checkpoint_sha256": source["g9_model_sha256"],
        "G9_original_mapper_checkpoint_sha256": source["g9_E1_mapper_sha256"]})
    json_new(OUT / "planner_candidate_protocol.json", {"status": "PREDECLARED_BEFORE_ANY_G95_PLAN_FORK",
        "planner": "exact frozen TD-MPC2 agent.act(eval_mode=True)",
        "planner_source_sha256": source["planner_source_sha256"],
        "horizon": 3, "sampled_sequences_per_CEM_iteration": 512,
        "policy_generated_prefix_sequences": 24, "Gaussian_sequences": 488,
        "CEM_iterations": 6, "elites_per_iteration": 64,
        "mean_update": "softmax weights exp(0.5*(elite value-max elite value)); weighted elite mean and variance; std clipped [0.05,2.0]",
        "initial_mean": "zero at episode first step, otherwise previous optimized mean shifted one action; last step zero",
        "initial_std": 2.0,
        "random_actions": "mean+std*torch.randn on cuda:0, componentwise clipped [-1,1]",
        "policy_prefix": "24 policy-generated length-3 model rollout action sequences, fixed within this planning call",
        "stochastic": True, "seed_control": "set_seed(seed), env.action_space.seed(seed), env.reset(seed=seed) at each ordered reused-scene episode; no extra planner RNG calls",
        "capture": "passive wrapper of _estimate_value, returns original value tensor unchanged; final sixth 3x512x2 actions and 512 model values copied after valuation; optimized mean read after agent.act",
        "subsample_without_force_labels": "two highest model-value, two nearest median rank, two lowest, two greedy max-min diverse in flattened first-two-action 4D normalized space; ties by original population index; plus exact optimized mean plan",
        "candidate_count_per_state": PLANS_PER_STATE,
        "sampled_final_population_count_per_state": SAMPLED_PER_STATE,
        "original_action_sequence": "extract only actions[0:2] from exact captured length-3 sample or optimized mean; preserve float32 values and order",
        "fork_execution": "open-loop action0 then original action1; no replan between forked intervals",
        "censoring": "one-step terminal/truncated forks have no F2 and no R2; retain as censored records, exclude from complete-target metrics",
        "behavior_action_pair": "audit-only, not substituted for planner candidate",
        "no_planner_mutation": True})
    json_new(OUT / "dataset_split.json", {"development_groups": {g: split[g] for g in groups},
        "selected_development_states": {g: states[g] for g in groups},
        "expected_state_count": counts, "episode_disjoint": True,
        "candidate_rows_from_same_state_stay_together": True,
        "untouched_confirmation_seeds": CONFIRM,
        "confirmation_selection_rule": "exact G3 choose_states four-per-episode progress/prior-force protocol, evaluated only after frozen confirmation behavior is collected",
        "confirmation_unopened": True})
    json_new(OUT / "dataset_manifest.json", {"status": "PREDECLARED_NO_G95_PLAN_FORCE_LABELS_YET",
        "development_state_selection_source": "G3 frozen development_state_selection.json",
        "selection_uses_candidate_force_labels": False,
        "selection_uses_future_behavior_force": False,
        "train_states": counts["train"], "validation_states": counts["validation"],
        "development_test_states": counts["development_test"],
        "total_development_states": sum(counts.values()),
        "planned_candidates_per_state": PLANS_PER_STATE,
        "planned_development_candidate_forks": sum(counts.values())*PLANS_PER_STATE,
        "planned_confirmation_states": 4*len(CONFIRM),
        "planned_confirmation_candidate_forks": 4*len(CONFIRM)*PLANS_PER_STATE,
        "all_training_labels_true_Generic_SOFA": True,
        "one_step_terminal_censored_not_redefined_as_R2": True})
    print(f"G9.5 prepared {sum(counts.values())} G3 planner states; 103000-103029 unopened", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare",), required=True)
    if parser.parse_args().stage == "prepare":
        prepare()


if __name__ == "__main__":
    main()
