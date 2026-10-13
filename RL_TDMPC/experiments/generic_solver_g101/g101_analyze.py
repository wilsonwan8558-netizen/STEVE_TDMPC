"""Analyze the predeclared G10.1 finite translation sweeps, without online data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g101"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g10"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g95"))

from g101_prepare import OUT, G10, PRED_AXIS, TRUE_AXIS, F_LIM, gate  # noqa: E402
from g101_collect import vkey  # noqa: E402
from g95_common import json_new, sha  # noqa: E402


def csv_new(path: Path, rows: list[dict]):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def get(items: dict, value: float):
    return items[vkey(value)]


def sign(value: float):
    return int(np.sign(value)) if abs(value) > 1e-6 else 0


def correlation(x, y, kind="pearson"):
    if len(x) < 3 or np.ptp(x) <= 1e-6 or np.ptp(y) <= 1e-6:
        return None
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def median_or_none(values):
    return float(np.median(values)) if values else None


def summary_counts(items):
    return {"states": len(items),
        "true_forks": sum(i["true_fork_count"] for i in items),
        "censored_one_interval_forks": sum(sum(not r["complete_two_intervals"] for r in i["true_curve"]) for i in items),
        "translation_recoverable": sum(i["true_safe_translation_count"] > 0 for i in items),
        "no_safe_translation_on_finite_sweep": sum(i["true_safe_translation_count"] == 0 for i in items),
        "zero_translation_true_safe": sum(get({vkey(r["v"]): r for r in i["true_curve"]}, 0)["true_safe"] for i in items),
        "withdrawal_minus_1_true_safe": sum(get({vkey(r["v"]): r for r in i["true_curve"]}, -1)["true_safe"] for i in items),
        "withdrawal_minus_0_5_true_safe": sum(get({vkey(r["v"]): r for r in i["true_curve"]}, -.5)["true_safe"] for i in items),
        "any_negative_translation_true_safe": sum(any(r["v"] < 0 and r["true_safe"] for r in i["true_curve"]) for i in items),
        "predicted_safe_translation_found": sum(i["predicted_closest_safe_v"] is not None for i in items),
        "predicted_repair_true_safe": sum(i["predicted_closest_safe_v"] is not None and get({vkey(r["v"]): r for r in i["true_curve"]}, i["predicted_closest_safe_v"])["true_safe"] for i in items),
        "false_safe_predicted_repair": sum(i["predicted_closest_safe_v"] is not None and not get({vkey(r["v"]): r for r in i["true_curve"]}, i["predicted_closest_safe_v"])["true_safe"] for i in items),
        "false_infeasible": sum(i["predicted_closest_safe_v"] is None and i["true_safe_translation_count"] > 0 for i in items)}


def load():
    gate()
    manifest = json.loads((OUT / "state_manifest.json").read_text())
    if len(manifest["states"]) != 40:
        raise RuntimeError("G10.1 exact 40-state manifest required")
    for cohort in ("heldout", "events"):
        audit = json.loads((OUT / f"{cohort}_parent_replay_audit.json").read_text())
        if audit["status"] != "PASS_EXACT_FROZEN_PARENT_AND_G10_NOMINAL":
            raise RuntimeError(f"G10.1 {cohort} parent replay audit not PASS")
    items = []
    for record in manifest["states"]:
        cohort = "heldout" if record["cohort"] == "heldout_development_unsafe_nominal" else "events"
        name = (f"state_{record['state_id']:04d}.json" if cohort == "heldout" else
                f"state_{record['state_id']:04d}_minus_{record['lookback_steps']}.json")
        item = json.loads((OUT / f"{cohort}_state_curves" / name).read_text())
        if (item["seed"], item["decision_index"], item["G10_result_sha256"]) != (
                record["seed"], record["decision_index"], record["G10_result_sha256"]):
            raise RuntimeError("G10.1 state curve does not match manifest")
        item["G10_status"] = record["G10_status"]
        items.append(item)
    if len(items) != 40:
        raise RuntimeError("G10.1 incomplete curves")
    return items


def analyze():
    items = load()
    sweep, recovery, directions, slopes, monotonicity = [], [], [], [], []
    boundaries, zeros, withdrawals, events = [], [], [], []
    p_cube = np.full((40, len(PRED_AXIS), 3), np.nan, dtype=np.float32)
    t_cube = np.full((40, len(TRUE_AXIS), 3), np.nan, dtype=np.float32)
    t_complete = np.zeros((40, len(TRUE_AXIS)), dtype=bool)
    t_safe = np.zeros((40, len(TRUE_AXIS)), dtype=bool)
    p_safe = np.zeros((40, len(PRED_AXIS)), dtype=bool)
    state_keys = []
    for index, item in enumerate(items):
        key = {k: item[k] for k in ("cohort", "state_id", "seed", "decision_index", "lookback_steps")}
        state_keys.append(f"{item['cohort']}:{item['state_id']}:{item['lookback_steps']}")
        v_nom = item["v_nom"]
        p = {vkey(r["v"]): r for r in item["predicted_curve"]}
        t = {vkey(r["v"]): r for r in item["true_curve"]}
        for j, v in enumerate(PRED_AXIS):
            row = get(p, v)
            p_cube[index, j] = row["Q95_corrected_replicas_N"]
            p_safe[index, j] = row["predicted_safe"]
        for j, v in enumerate(TRUE_AXIS):
            row = get(t, v)
            t_cube[index, j] = [row["F1_true_N"], row["F2_true_N"], row["R2_true_N"]]
            t_complete[index, j] = row["complete_two_intervals"]
            t_safe[index, j] = row["true_safe"]
        for row in item["true_curve"]:
            predicted = get(p, row["v"])
            sweep.append({**key, "v": row["v"], "w_nom_fixed": item["w_nom_fixed"],
                "fixed_second_action": json.dumps(item["second_action_fixed"]),
                "F1_true_N": row["F1_true_N"], "F2_true_N": row["F2_true_N"],
                "R2_true_N": row["R2_true_N"], "complete_two_intervals": row["complete_two_intervals"],
                "true_safe": row["true_safe"], "censor_reason": row["censor_reason"],
                "predicted_max_corrected_Q95_N": predicted["max_corrected_Q95_N"],
                "predicted_safe": predicted["predicted_safe"]})
        pred_v, true_v = item["predicted_closest_safe_v"], item["true_closest_tested_safe_v"]
        pred_row = get(t, pred_v) if pred_v is not None else None
        false_safe = pred_row is not None and not pred_row["true_safe"]
        false_infeasible = pred_v is None and true_v is not None
        recovery.append({**key, "v_nom": v_nom, "w_nom_fixed": item["w_nom_fixed"],
            "G10_2D_repair_status": item["G10_status"],
            "nominal_true_R2_N": item["nominal_true_R2_N"],
            "true_safe_count_tested": item["true_safe_translation_count"],
            "translation_recoverable_on_finite_sweep": true_v is not None,
            "v_true_closest_safe": true_v, "delta_v_true": true_v-v_nom if true_v is not None else None,
            "abs_delta_v_true": abs(true_v-v_nom) if true_v is not None else None,
            "predicted_safe_count_tested": item["predicted_safe_translation_count"],
            "v_pred_closest_safe": pred_v, "delta_v_pred": pred_v-v_nom if pred_v is not None else None,
            "predicted_repair_true_R2_N": pred_row["R2_true_N"] if pred_row else None,
            "predicted_repair_two_intervals_complete": pred_row["complete_two_intervals"] if pred_row else None,
            "predicted_repair_true_safe": pred_row["true_safe"] if pred_row else None,
            "false_safe": false_safe, "false_infeasible": false_infeasible,
            "finite_sweep_not_impossibility_proof": True})
        if true_v is not None and pred_v is not None:
            error = abs(pred_v-true_v)
            true_side, pred_side = sign(true_v-v_nom), sign(pred_v-v_nom)
            boundaries.append({**key, "v_nom": v_nom, "v_true_closest_safe": true_v,
                "v_pred_closest_safe": pred_v, "abs_boundary_error_v": error,
                "true_side_of_nominal": true_side, "predicted_side_of_nominal": pred_side,
                "same_side_or_nominal": true_side == pred_side,
                "predicted_candidate_true_safe": pred_row["true_safe"]})
        zero = get(t, 0)
        zeros.append({**key, "w_nom_fixed": item["w_nom_fixed"],
            "zero_translation_fixed_rotation_true_R2_N": zero["R2_true_N"],
            "zero_translation_fixed_rotation_complete": zero["complete_two_intervals"],
            "zero_translation_fixed_rotation_true_safe": zero["true_safe"],
            "zero_translation_fixed_rotation_predicted_safe": get(p, 0)["predicted_safe"],
            "G10_zero_translation_zero_rotation_true_R2_N": item["G10_zero_translation_zero_rotation_R2_N"],
            "G10_zero_translation_zero_rotation_true_safe": item["G10_zero_translation_zero_rotation_true_safe"],
            "zero_rotation_also_required_for_this_witness":
                item["G10_zero_translation_zero_rotation_true_safe"] and not zero["true_safe"]})
        withdrawals.append({**key, "v_nom": v_nom,
            **{f"v_{label}_true_R2_N": get(t, v)["R2_true_N"] for label, v in
               (("minus_1", -1), ("minus_0_5", -.5), ("zero", 0))},
            **{f"v_{label}_true_safe": get(t, v)["true_safe"] for label, v in
               (("minus_1", -1), ("minus_0_5", -.5), ("zero", 0))},
            "stronger_withdrawal_reduces_R2_minus_1_vs_minus_0_5":
                get(t, -1)["complete_two_intervals"] and get(t, -.5)["complete_two_intervals"] and
                get(t, -1)["R2_true_N"] < get(t, -.5)["R2_true_N"]-1e-6})
        nominal_t, nominal_p = get(t, v_nom), get(p, v_nom)
        local_direction_rows = []
        for delta in (.05, .1, .2):
            for direction in (-1, 1):
                v = vkey(np.clip(np.float32(v_nom)+np.float32(direction*delta), -1, 1))
                tr, pr = get(t, v), get(p, v)
                valid = (v != vkey(v_nom) and nominal_t["complete_two_intervals"] and
                         tr["complete_two_intervals"])
                ts = sign(tr["R2_true_N"]-nominal_t["R2_true_N"]) if valid else None
                ps = sign(pr["max_corrected_Q95_N"]-nominal_p["max_corrected_Q95_N"]) if valid else None
                row = {**key, "v_nom": v_nom, "probe_delta": delta, "probe_direction": direction,
                    "probe_v": v, "within_bounds_nonzero_change": v != vkey(v_nom),
                    "two_intervals_complete": valid, "true_delta_R2_N":
                    tr["R2_true_N"]-nominal_t["R2_true_N"] if valid else None,
                    "predicted_delta_max_Q95_N": pr["max_corrected_Q95_N"]-nominal_p["max_corrected_Q95_N"] if valid else None,
                    "true_sign": ts, "predicted_sign": ps,
                    "direction_agrees": ts == ps if valid else None,
                    "true_force_decreases": ts == -1 if valid else None}
                directions.append(row)
                local_direction_rows.append(row)
        minus_v = vkey(np.clip(np.float32(v_nom)-np.float32(.05), -1, 1))
        plus_v = vkey(np.clip(np.float32(v_nom)+np.float32(.05), -1, 1))
        minus, plus = get(t, minus_v), get(t, plus_v)
        if plus_v != vkey(v_nom) and minus_v != vkey(v_nom) and plus["complete_two_intervals"] and minus["complete_two_intervals"]:
            a, b, method = minus_v, plus_v, "central"
        elif plus_v != vkey(v_nom) and plus["complete_two_intervals"] and nominal_t["complete_two_intervals"]:
            a, b, method = v_nom, plus_v, "forward"
        elif minus_v != vkey(v_nom) and minus["complete_two_intervals"] and nominal_t["complete_two_intervals"]:
            a, b, method = minus_v, v_nom, "backward"
        else:
            a = b = method = None
        true_slope = (get(t, b)["R2_true_N"]-get(t, a)["R2_true_N"])/(b-a) if a is not None else None
        pred_slope = (get(p, b)["max_corrected_Q95_N"]-get(p, a)["max_corrected_Q95_N"])/(b-a) if a is not None else None
        slopes.append({**key, "method": method, "v_low": a, "v_high": b,
            "true_slope_N_per_normalized_v": true_slope,
            "predicted_slope_N_per_normalized_v": pred_slope,
            "true_slope_sign": sign(true_slope) if true_slope is not None else None,
            "predicted_slope_sign": sign(pred_slope) if pred_slope is not None else None,
            "slope_sign_agrees": sign(true_slope) == sign(pred_slope) if true_slope is not None else None})
        uniform = [get(t, v) for v in TRUE_AXIS]
        def mono(rows):
            valid = [(r["v"], r["R2_true_N"]) for r in rows if r["complete_two_intervals"]]
            rho = correlation([v for v, _ in valid], [f for _, f in valid], "spearman")
            pairs = [(left, right) for left, right in zip(rows[:-1], rows[1:])
                     if left["complete_two_intervals"] and right["complete_two_intervals"]]
            violations = sum(right["R2_true_N"] < left["R2_true_N"]-1e-6 for left, right in pairs)
            frac = violations/len(pairs) if pairs else None
            return len(valid), len(pairs), violations, frac, rho
        global_m = mono(uniform)
        local_m = mono([r for r in uniform if abs(r["v"]-v_nom) <= .2+1e-6])
        monotonicity.append({**key, "global_complete_grid_points": global_m[0],
            "global_valid_adjacent_pairs": global_m[1], "global_adjacent_decrease_count": global_m[2],
            "global_adjacent_decrease_fraction": global_m[3], "global_Spearman_v_R2": global_m[4],
            "global_approximately_nondecreasing": global_m[3] <= .05 if global_m[3] is not None else None,
            "local_complete_grid_points": local_m[0], "local_valid_adjacent_pairs": local_m[1],
            "local_adjacent_decrease_count": local_m[2], "local_adjacent_decrease_fraction": local_m[3],
            "local_Spearman_v_R2": local_m[4],
            "local_approximately_nondecreasing": local_m[3] <= .05 if local_m[3] is not None else None})
        if item["cohort"] == "difficult_event_early_intervention":
            valid_dirs = [r for r in local_direction_rows if r["two_intervals_complete"]]
            events.append({**key, "v_nom": v_nom, "w_nom_fixed": item["w_nom_fixed"],
                "G10_2D_repair_status": item["G10_status"],
                "v_true_closest_safe": true_v, "v_pred_closest_safe": pred_v,
                "translation_recoverable": true_v is not None,
                "predicted_repair_true_safe": pred_row["true_safe"] if pred_row else None,
                "false_infeasible": false_infeasible, "false_safe": false_safe,
                "zero_translation_true_safe": zero["true_safe"],
                "withdrawal_minus_1_true_safe": get(t, -1)["true_safe"],
                "withdrawal_minus_0_5_true_safe": get(t, -.5)["true_safe"],
                "safe_direction_agreement_count": sum(r["direction_agrees"] for r in valid_dirs),
                "safe_direction_valid_probes": len(valid_dirs),
                "true_curve_file": str(OUT / "events_state_curves" /
                    f"state_{item['state_id']:04d}_minus_{item['lookback_steps']}.json")})
    if len(sweep) != sum(len(item["true_curve"]) for item in items):
        raise RuntimeError("G10.1 sweep aggregation incomplete")
    csv_new(OUT / "translation_sweep_results.csv", sweep)
    np.savez_compressed(OUT / "true_force_curves.npz", state_keys=np.asarray(state_keys),
        uniform_v=TRUE_AXIS, F1_F2_R2_N=t_cube, complete=t_complete, safe=t_safe)
    np.savez_compressed(OUT / "predicted_force_curves.npz", state_keys=np.asarray(state_keys),
        uniform_v=PRED_AXIS, corrected_Q95_replicas_N=p_cube, safe=p_safe)
    for name, rows in (("recoverability_results", recovery), ("direction_agreement_results", directions),
                       ("slope_results", slopes), ("monotonicity_results", monotonicity),
                       ("safe_boundary_results", boundaries), ("zero_translation_witness_results", zeros),
                       ("withdrawal_witness_results", withdrawals), ("difficult_event_results", events)):
        csv_new(OUT / f"{name}.csv", rows)
    cohorts = {"heldout": [i for i in items if i["cohort"] == "heldout_development_unsafe_nominal"],
        "t_star_minus_1": [i for i in items if i["lookback_steps"] == 1],
        "t_star_minus_2": [i for i in items if i["lookback_steps"] == 2]}
    counts = {name: summary_counts(rows) for name, rows in cohorts.items()}
    counts["combined_descriptive_nonexchangeable"] = summary_counts(items)
    valid_dir = [r for r in directions if r["two_intervals_complete"]]
    direction_summary = {name: {"agreement": sum(r["direction_agrees"] for r in valid_dir if predicate(r)),
        "valid": sum(predicate(r) for r in valid_dir)} for name, predicate in (
            ("overall", lambda r: True), ("increasing_translation", lambda r: r["probe_direction"] == 1),
            ("decreasing_translation", lambda r: r["probe_direction"] == -1),
            ("withdrawal_probe", lambda r: r["probe_v"] < 0 and r["probe_direction"] == -1))}
    for value in direction_summary.values():
        value["rate"] = value["agreement"]/value["valid"] if value["valid"] else None
    decreasing = [r for r in valid_dir if r["probe_direction"] == -1]
    direction_summary["reducing_translation_lowers_true_R2"] = {
        "count": sum(r["true_force_decreases"] for r in decreasing),
        "valid": len(decreasing),
        "rate": sum(r["true_force_decreases"] for r in decreasing)/len(decreasing) if decreasing else None}
    valid_slopes = [r for r in slopes if r["true_slope_N_per_normalized_v"] is not None]
    sx = [r["true_slope_N_per_normalized_v"] for r in valid_slopes]
    sy = [r["predicted_slope_N_per_normalized_v"] for r in valid_slopes]
    slope_summary = {"valid": len(valid_slopes), "sign_agreement": sum(r["slope_sign_agrees"] for r in valid_slopes),
        "sign_agreement_rate": np.mean([r["slope_sign_agrees"] for r in valid_slopes]).item() if valid_slopes else None,
        "signed_slope_pearson": correlation(sx, sy),
        "absolute_slope_magnitude_pearson_diagnostic_only": correlation(np.abs(sx), np.abs(sy))}
    boundary_errors = [r["abs_boundary_error_v"] for r in boundaries]
    boundary_summary = {"both_safe_candidates_count": len(boundaries),
        "mean_abs_error_v": float(np.mean(boundary_errors)) if boundary_errors else None,
        "median_abs_error_v": float(np.median(boundary_errors)) if boundary_errors else None,
        "max_abs_error_v": max(boundary_errors) if boundary_errors else None,
        "same_side_count": sum(r["same_side_or_nominal"] for r in boundaries)}
    mono_summary = {"global_approximately_nondecreasing_count": sum(r["global_approximately_nondecreasing"] is True for r in monotonicity),
        "local_approximately_nondecreasing_count": sum(r["local_approximately_nondecreasing"] is True for r in monotonicity),
        "global_Spearman_median": median_or_none([r["global_Spearman_v_R2"] for r in monotonicity if r["global_Spearman_v_R2"] is not None]),
        "local_Spearman_median": median_or_none([r["local_Spearman_v_R2"] for r in monotonicity if r["local_Spearman_v_R2"] is not None]),
        "global_adjacent_decrease_fraction_median": median_or_none([r["global_adjacent_decrease_fraction"] for r in monotonicity if r["global_adjacent_decrease_fraction"] is not None]),
        "local_adjacent_decrease_fraction_median": median_or_none([r["local_adjacent_decrease_fraction"] for r in monotonicity if r["local_adjacent_decrease_fraction"] is not None])}
    g10_summary = json.loads((G10 / "offline_summary.json").read_text())
    comparison = {"G10_classification": "C_FORCE_CONSTRAINED_REPAIR_FAILS",
        "G10_offline_summary_sha256": sha(G10 / "offline_summary.json"),
        "G10_heldout": g10_summary["heldout_unsafe_nominal"],
        "G101_translation_only_counts": counts,
        "maximum_nominal_CUDA_batch_vs_single_difference_N":
            max(i["nominal_CUDA_batch_vs_single_max_abs_difference_N"] for i in items),
        "predicted_tested_points_within_0_0001_N_of_F_lim":
            sum(abs(r["max_corrected_Q95_N"]-F_LIM) <= 1e-4
                for i in items for r in i["predicted_curve"]),
        "direction_summary": direction_summary, "slope_summary": slope_summary,
        "monotonicity_summary": mono_summary, "boundary_summary": boundary_summary,
        "zero_rotation_required_for_G10_zero_safe_witness_count":
            sum(r["zero_rotation_also_required_for_this_witness"] for r in zeros),
        "positive_nominal_translation_states": sum(r["v_nom"] > 0 for r in recovery),
        "positive_nominal_translation_states_zero_translation_true_safe":
            sum(r["v_nom"] > 0 and z["zero_translation_fixed_rotation_true_safe"]
                for r, z in zip(recovery, zeros)),
        "stronger_withdrawal_minus_1_reduces_R2_vs_minus_0_5_count":
            sum(r["stronger_withdrawal_reduces_R2_minus_1_vs_minus_0_5"] for r in withdrawals),
        "all_relevant_finite_sweep_no_mathematical_impossibility_claim": True,
        "online_107000_107199_opened": False}
    json_new(OUT / "G10_vs_G101_comparison.json", comparison)
    return comparison


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2), flush=True)
