"""Assemble the audited G3 artifacts into a human-reviewed final report."""

from __future__ import annotations

import argparse
import csv
import json

from prepare import OUT


def rows(name):
    with (OUT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def data(name):
    return json.loads((OUT / name).read_text())


def f(value, digits=3):
    if value in (None, "", "None"):
        return "undefined"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", choices=("A", "B", "C"), required=True)
    parser.add_argument("--interpretation", required=True, help="Human-reviewed qualitative interpretation; no arbitrary numerical gate")
    args = parser.parse_args()
    path = OUT / "PHASE_G3_REPORT.md"
    if path.exists():
        raise FileExistsError(path)
    checkpoint = data("checkpoint_audit.json")
    split = data("development_split.json")
    target = data("force_target_definition.json")
    restore = data("counterfactual_restore_audit.json")
    manifest = data("data_manifest.json")
    training = data("training_manifest.json")
    frozen = data("frozen_model_manifest.json")
    confirmation_seed = data("confirmation_seed_manifest.json")
    confirmation = data("confirmation_summary.json")
    replay = data("confirmation_replay_audit.json")
    val = rows("validation_metrics.csv")
    behavior = rows("development_behavior_metrics.csv")
    tails = rows("development_behavior_tail_metrics.csv")
    candidate = rows("development_candidate_summary.csv")
    boundary = rows("development_boundary_metrics.csv")
    boot = data("bootstrap_summary.json")["metrics"]
    q = data("metadata.json")["quantiles_nominal_N"]
    short = {"M0_behavior": "M0", "M1_counterfactual": "M1"}
    classification = {"A": "PREDICTOR READY FOR SAFETY STUDY", "B": "CONDITIONAL PREDICTOR READY", "C": "PREDICTOR NOT READY"}[args.classification]
    lines = ["# Phase G3 — Generic one-step force predictor",
             "", f"**Final classification: {args.classification}. {classification}.**", "",
             "Offline predictor validation only. No shield, CBF, controller update, encoder fine-tuning, or clinical force threshold was used.", "",
             "## Frozen data and protocol", "",
             f"- Navigation checkpoint: `{checkpoint['path']}`; SHA256 `{checkpoint['sha256']}`; format v{checkpoint['format_version']}; 280,000 environment steps; CUDA `cuda:0` on RTX 5090.",
             f"- Canonical row: pre-action `o_d` → frozen 256-D `z_d`; normalized 2-D `a_d`; execute one `env.step(a_d)`; `force_next_max_N` is the max 3 mm distal force over its 22 SOFA substeps. G2 source field: `{target['g2_source_field']}`. Force uses vector-sum-before-norm and `J_contact^T lambda / 0.006 s`.",
             f"- G2 development dataset: 200 ordered reused-scene episodes, 15,500 behavior rows. Episode split: {len(split['train'])} train / {len(split['validation'])} validation / {len(split['development_test'])} development-test. Exact seeds: `development_split.json`; no step-level random split.",
             f"- Fork audit: `{restore['status']}` on {restore['state_count']} states across {len(restore['seeds'])} seeds; maximum checked parent/fork/G2 differences `{restore['max_abs_difference']}`. Development parent replay compared {manifest['parent_steps_compared_to_G2']} steps with G2.",
             "- Candidate set: behavior action plus clipped ±0.2/±0.4 translation/rotation axis offsets, 0.2/0.4 signed diagonals, and deterministic seeded broad fills to 21 unique bounded float32 actions. Exact rule and RNG offsets: `candidate_action_protocol.json`.",
             f"- Behavior TRAIN rows: {training['behavior_train_samples']}; TRAIN counterfactual states/labels: {manifest['development_candidate_counts']['train']['states']}/{manifest['development_candidate_counts']['train']['labels']}; VALIDATION states/labels: {manifest['development_candidate_counts']['validation']['states']}/{manifest['development_candidate_counts']['validation']['labels']}; DEVELOPMENT-TEST states/labels: {manifest['development_candidate_counts']['development_test']['states']}/{manifest['development_candidate_counts']['development_test']['labels']}; confirmation states/labels: {confirmation['state_count']}/{confirmation['candidate_count']}.",
             f"- SOFA one-step candidate evaluations: {manifest['development_one_step_candidate_evaluations']} development + {replay['candidate_evaluations']} confirmation; parent trajectories were separately executed and audited. Confirmation parent replay: {replay['parent_steps_compared']} steps, maximum differences `{replay['max_abs_difference']}`.",
             f"- Exact Generic G2 all-step statistical references (nominal N, not clinical): Q90 {q['q90']:.9f}, Q95 {q['q95']:.9f}, Q97.5 {q['q97.5']:.9f}, Q99 {q['q99']:.9f}.",
             "", "## Matched M0/M1 training and freeze", "",
             "Both models are newly initialized 258–64–64–1 ReLU MLPs with raw-force MSE, Adam 1e-3, batch 512, shared TRAIN-behavior-only normalization, and matched sample/optimizer-update budget per validation round. M0 uses TRAIN behavior; M1 additionally draws from TRAIN counterfactual labels. Validation behavior MSE selects each checkpoint; only validation candidate ranking selects the primary model. No LCP weights, dev-test labels, or confirmation labels enter fitting or selection.",
             f"Primary frozen predictor: **{short[frozen['primary_selected_model']]}**. The comparison checkpoints and normalization files were frozen before confirmation; hashes are in `frozen_model_manifest.json`. Confirmation seed manifest matches the predeclared 92000–92029 range and records the freeze manifest hash.",
             f"Model checkpoint SHA256: M0 `{frozen['checkpoint_sha256']['M0_behavior']}`; M1 `{frozen['checkpoint_sha256']['M1_counterfactual']}`. The shared normalization file SHA256 is `{frozen['normalization_sha256']['M0_behavior']}`.",
             "The recorded inference timings are single-call diagnostics, not a fair speed comparison: the first model call includes CUDA warm-up. Both heads have the same architecture.",
             "", "| Model | Val candidate RMSE | Val median state Spearman | Val pairwise accuracy | Selected epoch | Training wall time (s) |", "|---|---:|---:|---:|---:|---:|"]
    for row in val:
        name = row["model"]
        config = data(f"models/{name}/configuration.json")
        lines.append(f"| {short[name]} | {f(row['candidate_RMSE_N'])} | {f(row['median_within_state_Spearman'])} | {f(row['mean_pairwise_accuracy'])} | {config['selected_epoch']} | {f(config['train_wall_seconds'],1)} |")
    lines += ["", "## Development-test behavior prediction", "",
              "| Model | RMSE N | MAE N | R² | Pearson | Spearman |", "|---|---:|---:|---:|---:|---:|"]
    for row in behavior:
        lines.append(f"| {short[row['model']]} | {f(row['RMSE_N'])} | {f(row['MAE_N'])} | {f(row['R2'])} | {f(row['Pearson'])} | {f(row['Spearman'])} |")
    lines += ["", "Above-reference behavior MAE, mean/median underprediction (`true − predicted`), and false-safe counts at all four Generic references are in `development_behavior_tail_metrics.csv`.",
              "", "## Development-test candidate-action prediction", "",
              "| Model | RMSE N | Median state Spearman | Mean pairwise accuracy | Mean safe-choice regret N | Top-3 high/low containment |", "|---|---:|---:|---:|---:|---:|"]
    for row in candidate:
        lines.append(f"| {short[row['model']]} | {f(row['candidate_RMSE_N'])} | {f(row['median_within_state_Spearman'])} | {f(row['mean_pairwise_accuracy'])} | {f(row['mean_safe_choice_regret_N'])} | {f(row['mean_top3_high_containment'])} / {f(row['mean_top3_low_containment'])} |")
    lines += ["", "Per-state profile RMSE/MAE, predicted/true ranges, ranks of true force extrema, and individual regrets are in `development_candidate_metrics.csv`. Safe-choice regret is the true-force gap between the predicted-lowest candidate and the true-lowest candidate; no action was executed by a shield.",
              "", "### Development-test boundary fidelity", "",
              "Predicted exceedance means `F_hat > q`; true exceedance means `F_true > q`. False-safe rate is `P(predicted safe | true unsafe)`.", "",
              "| Model | Reference | Positive count | False-safe count/rate | AUROC | AUPRC | Mean missed severity N |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in boundary:
        lines.append(f"| {short[row['model']]} | {row['reference'].upper()} | {row['true_exceedances']} | {row['FN_false_safe']} / {f(row['false_safe_rate_P_pred_safe_given_true_unsafe'])} | {f(row['AUROC'])} | {f(row['AUPRC'])} | {f(row['false_safe_severity_mean_N'])} |")
    lines += ["", "`development_boundary_metrics.csv` also gives recall, precision, specificity, NPV, false-unsafe count/rate, and full missed-exceedance severity. Q99 must be read with its positive count.",
              "", "### Paired state-level bootstrap", "",
              "1,000 fixed-seed replicates resample states, keeping each state's 21 actions together. Four states per episode mean these intervals do not fully account for episode clustering.", "",
              "| M1 − M0 metric | Paired point difference | 95% percentile CI |", "|---|---:|---:|"]
    for key in ("within_state_Spearman", "pairwise_accuracy", "safe_choice_regret_N", "Q95_false_safe_rate", "Q95_AUPRC"):
        result = boot[key]
        lo, hi = result["CI95_percentile"] if result["CI95_percentile"] else (None, None)
        lines.append(f"| {key.replace('_',' ')} | {f(result['point_estimate_M1_minus_M0'])} | [{f(lo)}, {f(hi)}] |")
    lines += ["", "## Untouched confirmation: seeds 92000–92029", "",
              f"The frozen model/protocol existed before these 30 episodes were opened: **{confirmation['models_frozen_before_confirmation']}**. Exactly {confirmation['state_count']} states and {confirmation['candidate_count']} fresh candidate labels were evaluated, with no retraining or recalibration.", "",
              "| Model | Candidate RMSE N | Median state Spearman | Mean pairwise accuracy | Mean safe-choice regret N | Q95 false-safe rate | Q95 AUROC | Q95 AUPRC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("M0_behavior", "M1_counterfactual"):
        result = confirmation["models"][name]
        q95 = result["boundaries"]["q95"]
        lines.append(f"| {short[name]} | {f(result['candidate_RMSE_N'])} | {f(result['median_within_state_Spearman'])} | {f(result['mean_pairwise_accuracy'])} | {f(result['mean_safe_choice_regret_N'])} | {f(q95['false_safe_rate_P_pred_safe_given_true_unsafe'])} ({q95['FN_false_safe']}/{q95['true_exceedances']}) | {f(q95['AUROC'])} | {f(q95['AUPRC'])} |")
    lines += ["", "| Model | Q97.5 positives | Q97.5 false-safe count/rate | Q97.5 AUROC | Q97.5 AUPRC | Top-3 high/low containment |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("M0_behavior", "M1_counterfactual"):
        result = confirmation["models"][name]
        tail = result["boundaries"]["q97.5"]
        lines.append(f"| {short[name]} | {tail['true_exceedances']} | {tail['FN_false_safe']} / {f(tail['false_safe_rate_P_pred_safe_given_true_unsafe'])} | {f(tail['AUROC'])} | {f(tail['AUPRC'])} | {f(result['mean_top3_high_containment'])} / {f(result['mean_top3_low_containment'])} |")
    lines += ["", "Q99 has only 65 positive confirmation candidates; both models miss 38/65 at the raw Q99 reference. These extreme-tail estimates are descriptive only. Full Q90/Q95/Q97.5/Q99 diagnostics, candidate predictions, and state metrics are saved separately.",
              "", "## Interpretation and limitations", "", args.interpretation, "",
              "Historical LCP Phase 2D found weak behavior-only within-state candidate ranking and improved ranking with counterfactual supervision, but residual false-safe/tail and action-selection errors blocked a direct safety claim. The Generic qualitative pattern is compared here only as a methodological contrast; LCP and Generic nominal-N references are not numerically interchangeable.",
              "", "This G3 study establishes offline one-step prediction only. It does not establish closed-loop safety, task success under intervention, calibrated physical force, or clinical safety boundaries. No shield/CBF or training of TD-MPC2 was performed.", ""]
    path.write_text("\n".join(lines))
    print(path)


if __name__ == "__main__":
    main()
