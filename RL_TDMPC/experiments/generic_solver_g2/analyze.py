"""G2 navigation regression and descriptive Generic distal-force characterization."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


SEEDS = tuple(range(90000, 90200))
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_REPLICATES = 1000
SOFA_DT_S = 0.006
CONTROL_SUBSTEPS = 22


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n",
                    encoding="utf-8")


def quantiles(values, levels=(.5, .75, .9, .95, .975, .99)) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise RuntimeError("Empty/nonfinite numeric distribution")
    result = {"count": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
              "max": float(np.max(x))}
    for level in levels:
        result[f"q{100*level:g}"] = float(np.quantile(x, level))
    return result


def runs(flags: list[bool]) -> list[int]:
    result = []
    length = 0
    for flag in flags:
        if flag:
            length += 1
        elif length:
            result.append(length)
            length = 0
    if length:
        result.append(length)
    return result


def run_summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"run_count": 0, "median_length": None,
                "q95_length": None, "max_length": None, "total_steps": 0}
    return {"run_count": len(lengths), "median_length": float(np.median(lengths)),
            "q95_length": float(np.quantile(lengths, .95)),
            "max_length": max(lengths), "total_steps": sum(lengths)}


def correlation(x, y) -> dict:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise RuntimeError("Invalid correlation pairs")
    if np.std(a) == 0 or np.std(b) == 0:
        return {"pairs": len(a), "pearson": None, "spearman": None,
                "note": "Undefined because at least one variable is constant."}
    return {"pairs": len(a), "pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic)}


def validate_and_group(root: Path):
    steps = read(root / "step_level_force.csv")
    episodes = read(root / "episode_level_force.csv")
    reference = read(root.parent / "generic_solver_phase_g1b_checkpoint_sweep" /
                     "confirmation_200_episode_results.csv")
    if (len(episodes) != 200 or len(reference) != 200 or
            [int(row["seed"]) for row in episodes] != list(SEEDS) or
            [int(row["seed"]) for row in reference] != list(SEEDS)):
        raise RuntimeError("Missing, duplicated, or unordered 200-seed episode set")
    by_seed = defaultdict(list)
    for row in steps:
        by_seed[int(row["seed"])].append(row)
    if set(by_seed) != set(SEEDS):
        raise RuntimeError("Step rows have a wrong seed set")
    reward_diffs = []
    trajectory_mismatches = success_mismatches = length_mismatches = 0
    for index, (episode, expected) in enumerate(zip(episodes, reference)):
        seed = SEEDS[index]
        trace = by_seed[seed]
        length = int(episode["episode_length"])
        if ([int(row["step_index"]) for row in trace] != list(range(length)) or
                len(trace) != length or
                any(int(row["episode_index"]) != index for row in trace)):
            raise RuntimeError(f"Step identity/timing mismatch at seed {seed}")
        if any(row["episode_success"] != episode["success"] for row in trace):
            raise RuntimeError(f"Step episode outcome mismatch at seed {seed}")
        success_mismatches += episode["success"] != expected["success"]
        length_mismatches += length != int(expected["episode_length"])
        reward_diffs.append(abs(float(episode["episode_reward"])-
                                float(expected["episode_reward"])))
        trajectory_mismatches += episode["trajectory_sha256"] != expected["trajectory_sha256"]
        if (episode["checkpoint_sha256"] != expected["checkpoint_sha256"] or
                episode["checkpoint_step"] != expected["checkpoint_step"] or
                episode["simulation_error"] != "False"):
            raise RuntimeError(f"Frozen checkpoint/simulator mismatch at seed {seed}")
        force = np.asarray([float(row["distal_contact_force_max_N"]) for row in trace])
        mean_force = np.asarray([float(row["distal_contact_force_mean_N"]) for row in trace])
        final = np.asarray([float(row["distal_contact_force_final_N"]) for row in trace])
        fti = np.asarray([float(row["distal_force_time_integral_Ns"]) for row in trace])
        contact = np.asarray([int(row["distal_contact_count"]) > 0 for row in trace])
        if (not all(np.all(np.isfinite(x)) and np.all(x >= 0) for x in
                    (force, mean_force, final, fti)) or
                np.any(force + 1e-12 < mean_force) or np.any(force + 1e-12 < final) or
                not np.allclose(fti, mean_force*CONTROL_SUBSTEPS*SOFA_DT_S,
                                atol=1e-10, rtol=1e-10)):
            raise RuntimeError(f"Invalid force diagnostics at seed {seed}")
        if any(int(row["sofa_substeps"]) != CONTROL_SUBSTEPS or
               not (0 <= int(row["distal_contact_substeps"]) <= CONTROL_SUBSTEPS)
               for row in trace):
            raise RuntimeError(f"Missing SOFA substeps at seed {seed}")
        if any(not contact[j] and force[j] > 1e-10 for j in range(length)):
            raise RuntimeError(f"Force without active distal contact at seed {seed}")
        if (not np.isclose(float(episode["episode_peak_force_N"]), np.max(force)) or
                not np.isclose(float(episode["episode_mean_force_N"]), np.mean(force)) or
                not np.isclose(float(episode["episode_total_fti_Ns"]), np.sum(fti)) or
                int(episode["contact_steps"]) != int(np.sum(contact)) or
                not np.isclose(float(episode["contact_fraction"]), np.mean(contact)) or
                int(episode["longest_contact_run"]) != max(runs(contact.tolist()), default=0)):
            raise RuntimeError(f"Episode force summary mismatch at seed {seed}")
        for row in trace:
            if (len(json.loads(row["decision_observation_json"])) != 14 or
                    len(json.loads(row["next_observation_json"])) != 14 or
                    len(json.loads(row["selected_action_json"])) != 2 or
                    len(json.loads(row["applied_action_json"])) != 2):
                raise RuntimeError(f"Incomplete decision/action state at seed {seed}")
    if success_mismatches or length_mismatches or trajectory_mismatches or max(reward_diffs) > 1e-9:
        raise RuntimeError("G1B navigation baseline did not reproduce exactly")
    regression = {
        "status": "PASS", "seeds": [90000, 90199], "episodes": 200,
        "total_rl_steps": len(steps),
        "success_count": sum(row["success"] == "True" for row in episodes),
        "mean_episode_length": float(np.mean([int(row["episode_length"]) for row in episodes])),
        "median_episode_length": float(np.median([int(row["episode_length"]) for row in episodes])),
        "mean_episode_reward": float(np.mean([float(row["episode_reward"]) for row in episodes])),
        "success_mismatches": success_mismatches,
        "length_mismatches": length_mismatches,
        "trajectory_hash_mismatches": trajectory_mismatches,
        "maximum_reward_absolute_difference": max(reward_diffs),
        "reference": "G1B confirmation_200_episode_results.csv",
    }
    if (regression["success_count"] != 172 or
            regression["mean_episode_length"] != 77.5 or
            regression["median_episode_length"] != 58 or
            abs(regression["mean_episode_reward"]-3.4300794214612553) > 1e-9):
        raise RuntimeError("G1B aggregate task behavior changed")
    return steps, episodes, by_seed, regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    steps, episodes, by_seed, regression = validate_and_group(root)
    save(root / "task_regression.json", regression)
    force = np.asarray([float(row["distal_contact_force_max_N"]) for row in steps])
    endpoint = np.asarray([float(row["distal_contact_force_final_N"]) for row in steps])
    contact = np.asarray([int(row["distal_contact_count"]) > 0 for row in steps])
    if not np.any(contact):
        raise RuntimeError("No distal contact steps; cannot characterize contact distribution")
    all_dist = quantiles(force)
    contact_dist = quantiles(force[contact], (.5, .9, .95, .975, .99))
    contact_dist["fraction_of_all_steps"] = float(np.mean(contact))
    episode_measures = {
        "episode_peak_force_N": [float(row["episode_peak_force_N"]) for row in episodes],
        "episode_mean_force_N": [float(row["episode_mean_force_N"]) for row in episodes],
        "episode_total_fti_Ns": [float(row["episode_total_fti_Ns"]) for row in episodes],
        "contact_fraction": [float(row["contact_fraction"]) for row in episodes],
        "contact_steps": [int(row["contact_steps"]) for row in episodes],
        "longest_contact_run": [int(row["longest_contact_run"]) for row in episodes],
    }
    episode_dists = {key: quantiles(value, (.5, .9, .95, .99))
                     for key, value in episode_measures.items()}
    groups = {}
    for label, expected_success in (("successful", True), ("failed", False)):
        group = [row for row in episodes if (row["success"] == "True") == expected_success]
        groups[label] = {
            "episodes": len(group),
            "episode_peak_force_N": quantiles([float(row["episode_peak_force_N"]) for row in group],
                                             (.5, .9, .95)),
            "episode_total_fti_Ns": quantiles([float(row["episode_total_fti_Ns"]) for row in group],
                                             (.5, .9, .95)),
            "contact_fraction": quantiles([float(row["contact_fraction"]) for row in group],
                                          (.5, .9, .95)),
        }
    distribution = {
        "status": "COMPLETE", "primary_signal": "distal_contact_force_max_N",
        "units": "nominal simulation N", "distal_region_mm": 3.0,
        "contact_step_definition": "At least one active distal normal-contact group in any of the 22 SOFA substeps; force may theoretically cancel vectorially.",
        "positive_force_step_count": int(np.sum(force > 1e-12)),
        "positive_force_step_fraction": float(np.mean(force > 1e-12)),
        "active_contact_but_zero_resultant_steps": int(np.sum(contact & (force <= 1e-12))),
        "all_rl_steps": all_dist, "contact_only_steps": contact_dist,
        "episode_distributions": episode_dists,
        "success_failure_descriptive": groups,
        "total_rl_steps": len(steps),
    }
    save(root / "force_distribution_summary.json", distribution)

    thresholds = {label: float(np.quantile(force, level))
                  for label, level in (("q75", .75), ("q90", .9), ("q95", .95),
                                       ("q97.5", .975), ("q99", .99))}
    exceedance = {}
    for label in ("q90", "q95", "q97.5"):
        threshold = thresholds[label]
        per_episode = [[float(row["distal_contact_force_max_N"]) >= threshold
                        for row in by_seed[seed]] for seed in SEEDS]
        lengths = [length for flags in per_episode for length in runs(flags)]
        exceedance[label] = {
            "threshold_N": threshold, "comparison": ">=",
            "episodes_with_any": sum(any(flags) for flags in per_episode),
            **run_summary(lengths),
        }
    contact_runs = [length for seed in SEEDS for length in runs(
        [int(row["distal_contact_count"]) > 0 for row in by_seed[seed]])]
    progress = {}
    for label, lower, upper in (("early", 0, 1/3), ("middle", 1/3, 2/3),
                                ("late", 2/3, 1)):
        segment = [row for seed in SEEDS for row in by_seed[seed]
                   if lower <= (int(row["step_index"])+.5)/len(by_seed[seed]) < upper]
        x = np.asarray([float(row["distal_contact_force_max_N"]) for row in segment])
        progress[label] = {"steps": len(segment), "mean_force_N": float(np.mean(x)),
                           "contact_fraction": float(np.mean(
                               [int(row["distal_contact_count"]) > 0 for row in segment])),
                           "q95_exceedance_fraction": float(np.mean(x >= thresholds["q95"]))}
    temporal = {
        "status": "COMPLETE", "threshold_source": "G2 Generic all-step F_t distribution",
        "thresholds_nominal_N": thresholds,
        "contact_run_lengths": run_summary(contact_runs),
        "high_force_exceedance_runs": exceedance,
        "episode_progress_thirds": progress,
    }
    save(root / "temporal_structure.json", temporal)

    action = {}
    for label, transform in (("abs_translation", lambda x: abs(x[0])),
                             ("abs_rotation", lambda x: abs(x[1])),
                             ("action_norm", lambda x: float(np.linalg.norm(x)))):
        values = [transform(np.asarray(json.loads(row["selected_action_json"])))
                  for row in steps]
        same = correlation(values, force)
        lag_x, lag_y = [], []
        for seed in SEEDS:
            trace = by_seed[seed]
            for index in range(len(trace)-1):
                lag_x.append(transform(np.asarray(json.loads(
                    trace[index]["selected_action_json"]))))
                lag_y.append(float(trace[index+1]["distal_contact_force_max_N"]))
        action[label] = {"same_control_interval_force": same,
                         "following_control_interval_force": correlation(lag_x, lag_y)}
    relationship = {
        "status": "COMPLETE", "action_components": "absolute normalized translation and rotation; L2 norm",
        "same_interval_timing": "a_t is selected from decision observation s_t, then force is measured over env.step(a_t); CSV target is F_(t+1) from decision-state indexing.",
        "following_interval_timing": "a_t versus force measured during the next control interval, within the same episode only.",
        "correlations": action,
        "interpretation": "Descriptive association only; correlation does not establish controllability or causal force effects.",
    }
    save(root / "action_force_relationship.json", relationship)

    precursors = {}
    for event_label in ("q95", "q99"):
        event_threshold = thresholds[event_label]
        events = [(seed, index) for seed in SEEDS for index, row in enumerate(by_seed[seed])
                  if float(row["distal_contact_force_max_N"]) >= event_threshold]
        lags = {}
        for lag in (1, 2, 3):
            eligible = [(seed, index) for seed, index in events if index >= lag]
            prior = [float(by_seed[seed][index-lag]["distal_contact_force_max_N"])
                     for seed, index in eligible]
            lags[str(lag)] = {
                "eligible_events": len(eligible),
                "prior_nonzero_force_fraction": float(np.mean(np.asarray(prior) > 1e-12)) if prior else None,
                "prior_elevated_q75_fraction": float(np.mean(np.asarray(prior) >= thresholds["q75"])) if prior else None,
                "prior_tail_q90_fraction": float(np.mean(np.asarray(prior) >= thresholds["q90"])) if prior else None,
            }
        precursors[event_label] = {"threshold_N": event_threshold,
                                   "event_steps": len(events), "prior_lags": lags}
    temporal["high_force_event_precursors"] = precursors
    temporal["precursor_interpretation"] = "Retrospective description only; repeated adjacent event steps are retained, and this is not prospective prediction evidence."
    save(root / "temporal_structure.json", temporal)

    difference = force-endpoint
    ratios = force[(endpoint > 1e-9) & (force > 0)] / endpoint[(endpoint > 1e-9) & (force > 0)]
    misses = {}
    for label in ("q90", "q95", "q99"):
        threshold = thresholds[label]
        events = force >= threshold
        missed = events & (endpoint < threshold)
        misses[label] = {"threshold_N": threshold,
                         "max_substep_event_steps": int(np.sum(events)),
                         "endpoint_missed_steps": int(np.sum(missed)),
                         "fraction_of_max_events_missed": float(np.sum(missed)/np.sum(events)),
                         "fraction_of_all_steps_missed": float(np.mean(missed))}
    endpoint_summary = {
        "status": "COMPLETE", "steps": len(steps),
        "difference_max_minus_final_N": quantiles(difference, (.5, .9, .95)),
        "ratio_max_over_final_when_final_gt_1e_minus_9_N": quantiles(ratios, (.5, .9, .95, .99)),
        "ratio_eligible_steps": len(ratios),
        "ratio_excluded_steps_due_to_small_endpoint": int(len(force)-len(ratios)),
        "positive_peak_near_zero_endpoint_steps": int(np.sum((force > 1e-12) &
                                                             (endpoint <= 1e-9))),
        "positive_peak_near_zero_endpoint_fraction_of_contact_steps": float(
            np.sum((force > 1e-12) & (endpoint <= 1e-9))/np.sum(contact)),
        "endpoint_missed_events": misses,
        "interpretation": "Endpoint-only sampling can miss within-interval peaks; Generic-derived quantile levels are statistical references, not clinical safety thresholds.",
    }
    save(root / "substep_vs_endpoint.json", endpoint_summary)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(episodes), size=(BOOTSTRAP_REPLICATES, len(episodes)))
    bootstrap = {}
    for key in ("contact_fraction", "episode_peak_force_N", "episode_total_fti_Ns"):
        values = np.asarray(episode_measures[key], dtype=float)
        mean_samples = np.mean(values[indices], axis=1)
        median_samples = np.median(values[indices], axis=1)
        bootstrap[key] = {
            "observed_episode_mean": float(np.mean(values)),
            "episode_mean_ci95": [float(x) for x in np.quantile(mean_samples, (.025, .975))],
            "observed_episode_median": float(np.median(values)),
            "episode_median_ci95": [float(x) for x in np.quantile(median_samples, (.025, .975))],
        }
    save(root / "bootstrap_summary.json", {
        "status": "COMPLETE", "unit": "whole episode", "episodes": 200,
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "method": "Nonparametric episode resampling with replacement; common 200-index draw across statistics; percentile 95% intervals. Correlated RL steps are not resampled independently.",
        "statistics": bootstrap,
    })

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(force, bins=80, color="steelblue")
    axes[0].set(xlabel="3 mm distal max-substep force (nominal N)", ylabel="RL steps",
                title="All-step force distribution")
    axes[1].hist(np.log10(force[force > 0]), bins=80, color="darkorange")
    axes[1].set(xlabel="log10(force / nominal N)", ylabel="contact-force steps",
                title="Positive-force tail")
    fig.tight_layout()
    fig.savefig(plots / "force_distribution.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(endpoint, force, s=3, alpha=.15)
    extent = max(float(np.max(endpoint)), float(np.max(force)))
    ax.plot([0, extent], [0, extent], color="black", lw=1)
    ax.set(xlabel="Final-substep distal force (nominal N)",
           ylabel="Max-substep distal force (nominal N)",
           title="Within-control peaks versus endpoint")
    fig.tight_layout()
    fig.savefig(plots / "substep_vs_endpoint.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    success_peaks = [float(row["episode_peak_force_N"]) for row in episodes
                     if row["success"] == "True"]
    fail_peaks = [float(row["episode_peak_force_N"]) for row in episodes
                  if row["success"] == "False"]
    ax.boxplot([success_peaks, fail_peaks], tick_labels=["Success", "Failure"],
               showfliers=True)
    ax.set(ylabel="Episode peak max-substep force (nominal N)",
           title="Episode force by navigation outcome")
    fig.tight_layout()
    fig.savefig(plots / "episode_peak_by_outcome.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "total_rl_steps": len(steps),
                      "contact_fraction": contact_dist["fraction_of_all_steps"],
                      "force_q95_N": thresholds["q95"],
                      "episode_peak_q95_N": episode_dists["episode_peak_force_N"]["q95"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
