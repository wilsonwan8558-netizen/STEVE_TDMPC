"""Descriptive, non-pooled Generic-versus-accepted-LCP force context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    generic = json.loads((root / "force_distribution_summary.json").read_text())
    lcp = json.loads((root / "historical_lcp_reference.json").read_text())
    regression = json.loads((root / "task_regression.json").read_text())
    if (generic["status"] != "COMPLETE" or regression["status"] != "PASS" or
            regression["episodes"] != 200 or
            lcp["status"] != "ACCEPTED_HISTORICAL_METHOD_REFERENCE_ONLY"):
        raise RuntimeError("Generic full baseline or historical provenance incomplete")
    values = {
        "contact_step_fraction": (
            generic["contact_only_steps"]["fraction_of_all_steps"],
            lcp["contact_step_fraction"]),
        "positive_force_step_fraction": (
            generic["positive_force_step_fraction"], lcp["contact_step_fraction"]),
        "all_step_force_median_nominal_N": (
            generic["all_rl_steps"]["median"], lcp["all_step_force_median_nominal_N"]),
        "all_step_force_q95_nominal_N": (
            generic["all_rl_steps"]["q95"], lcp["all_step_force_q95_nominal_N"]),
        "contact_only_force_median_nominal_N": (
            generic["contact_only_steps"]["median"], lcp["contact_only_force_median_nominal_N"]),
        "contact_only_force_q95_nominal_N": (
            generic["contact_only_steps"]["q95"], lcp["contact_only_force_q95_nominal_N"]),
        "episode_peak_median_nominal_N": (
            generic["episode_distributions"]["episode_peak_force_N"]["median"],
            lcp["episode_peak_median_nominal_N"]),
        "episode_peak_q95_nominal_N": (
            generic["episode_distributions"]["episode_peak_force_N"]["q95"],
            lcp["episode_peak_q95_nominal_N"]),
        "episode_total_fti_median_nominal_Ns": (
            generic["episode_distributions"]["episode_total_fti_Ns"]["median"],
            lcp["episode_total_fti_median_nominal_Ns"]),
        "episode_total_fti_q95_nominal_Ns": (
            generic["episode_distributions"]["episode_total_fti_Ns"]["q95"],
            lcp["episode_total_fti_q95_nominal_Ns"]),
    }
    comparison = {
        key: {"generic": new, "historical_lcp": old,
              "generic_minus_historical_lcp": new-old,
              "generic_divided_by_historical_lcp": new/old if old else None}
        for key, (new, old) in values.items()
    }
    result = {
        "status": "DESCRIPTIVE_ONLY", "generic_episodes": 200,
        "generic_total_rl_steps": regression["total_rl_steps"],
        "historical_lcp_episodes": lcp["episodes"],
        "historical_lcp_total_rl_steps": lcp["total_rl_steps"],
        "historical_lcp_source_branch_commit": lcp["source_branch_commit"],
        "historical_lcp_source_sha256": lcp["source_sha256"],
        "metrics": comparison,
        "contact_fraction_caveat": "Generic primary contact step is any active distal contact group during the RL interval; the archived LCP count used positive peak force. Generic positive-force fraction is shown separately for closer alignment.",
        "interpretation_limit": "Different solvers change simulated contact dynamics, and the frozen TD-MPC2 checkpoints differ. This is not a controlled solver-only or algorithm ablation and says nothing about clinical correctness. Generic thresholds are computed solely from Generic data.",
    }
    (root / "historical_lcp_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
