"""Held-out G5 diagnostics, contact strata, and paired state bootstrap."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from g5_data import OUT, PRIMARY, behavior, candidates, minimal_groups, source
from g5_prepare import SEEDS
from g5_metrics import load_predictor, predict, evaluate_candidate, regression, classification, quantiles
from g4_data import candidate_states
from g4_train import new_json

MODELS = PRIMARY + ("Z1", "P1", "Pmin1", "Pmin", "ZPmin")
ADDITIONS = {"K": ("Z", "ZK"), "D": ("Z", "ZD"), "C": ("Z", "ZC")}
REMOVALS = {"K": ("ZP", "ZP_noK"), "D": ("ZP", "ZP_noD"), "C": ("ZP", "ZP_noC")}
FIELDS = ("Q95_false_safe_rate", "candidate_RMSE_N", "mean_pairwise_accuracy", "mean_safe_choice_regret_N")


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise RuntimeError(f"No records for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_vector(y, p, records, q95):
    n = len(records)
    return {"sse": np.asarray([np.sum((p[i*21:(i+1)*21]-y[i*21:(i+1)*21])**2) for i in range(n)]),
            "pair": np.asarray([r["pairwise_accuracy"] if r["pairwise_accuracy"] is not None else np.nan for r in records], dtype=float),
            "regret": np.asarray([r["safe_choice_regret_N"] for r in records], dtype=float),
            "unsafe": np.asarray([int(np.sum(y[i*21:(i+1)*21]>q95)) for i in range(n)]),
            "false_safe": np.asarray([int(np.sum((y[i*21:(i+1)*21]>q95)&(p[i*21:(i+1)*21]<=q95))) for i in range(n)])}


def sampled_metrics(v, ix):
    positive = int(np.sum(v["unsafe"][ix]))
    return {"Q95_false_safe_rate": float(np.sum(v["false_safe"][ix])/positive) if positive else np.nan,
            "candidate_RMSE_N": float(np.sqrt(np.mean(v["sse"][ix])/21)),
            "mean_pairwise_accuracy": float(np.nanmean(v["pair"][ix])),
            "mean_safe_choice_regret_N": float(np.mean(v["regret"][ix]))}


def bootstrap(vectors, n):
    rng = np.random.default_rng(20261008)
    if n != 120:
        raise RuntimeError("G5 development bootstrap requires 120 paired states")
    report = {}
    for kind, comparisons in (("addition", ADDITIONS), ("removal", REMOVALS)):
        for group, (left, right) in comparisons.items():
            key = f"{kind}_{group}_{right}_minus_{left}"
            by_seed = {seed: {field: sampled_metrics(vectors[right][seed], np.arange(n))[field] -
                                      sampled_metrics(vectors[left][seed], np.arange(n))[field] for field in FIELDS} for seed in SEEDS}
            draws = {field: [] for field in FIELDS}
            for _ in range(1000):
                ix = rng.integers(0, n, n)
                differences = {field: [] for field in FIELDS}
                for seed in SEEDS:
                    a, b = sampled_metrics(vectors[left][seed], ix), sampled_metrics(vectors[right][seed], ix)
                    for field in FIELDS:
                        if np.isfinite(a[field]) and np.isfinite(b[field]):
                            differences[field].append(b[field]-a[field])
                for field in FIELDS:
                    if differences[field]:
                        draws[field].append(float(np.mean(differences[field])))
            report[key] = {field: {"individual_seed_differences": {str(seed): by_seed[seed][field] for seed in SEEDS},
                                  "training_seed_mean_difference": float(np.mean([by_seed[seed][field] for seed in SEEDS])),
                                  "training_seed_sd_difference": float(np.std([by_seed[seed][field] for seed in SEEDS], ddof=1)),
                                  "between_state_CI95_percentile": [float(np.quantile(draws[field], .025)), float(np.quantile(draws[field], .975))],
                                  "bootstrap_replicates": len(draws[field])} for field in FIELDS}
    return {"method": "1000 fixed-seed paired bootstrap resamples of 120 states; all 21 actions retained; same state draw across models and three training seeds; no prediction ensembling",
            "bootstrap_seed": 20261008, "effect_sign": "addition=Z+group minus Z; removal=full ZP without group minus full ZP; lower false-safe/RMSE/regret and higher pairwise are favorable",
            "comparisons": report}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen CUDA device required")
    if not (OUT / "secondary_training_complete.json").exists():
        raise RuntimeError("All models must be frozen before development evaluation")
    groups = minimal_groups()
    arrays, split, index, physical = source()
    states = candidate_states("development_test", split)
    q = quantiles()
    if len(states) != 120:
        raise RuntimeError("Held-out development state count changed")
    device = torch.device("cuda:0")
    behavior_rows, candidate_rows, boundary_rows, state_rows, strata_rows, emergence_rows = [], [], [], [], [], []
    predictions, vectors = {}, {name: {} for name in MODELS}
    contact_reference = None
    for name in MODELS:
        xbeh, ybeh, _ = behavior(name, "development_test", arrays, split, index, physical, groups)
        xcan, ycan, contact = candidates(name, states, arrays, index, physical, groups)
        if len(ybeh) != 2111 or len(ycan) != 2520:
            raise RuntimeError(f"Held-out G4 counts changed for {name}")
        if contact_reference is None:
            contact_reference = contact
        elif not np.array_equal(contact, contact_reference):
            raise RuntimeError("Contact strata differ between models")
        predictions[name] = {}
        for seed in SEEDS:
            predictor = load_predictor(name, seed, device, groups)
            pbeh = predict(predictor, xbeh, device)
            pcan = predict(predictor, xcan, device)
            records, aggregate, boundaries = evaluate_candidate(states, ycan, pcan, q)
            label = {"model": name, "training_seed": seed}
            behavior_rows.append({**label, **regression(ybeh, pbeh),
                "Q95_false_safe_rate": classification(ybeh,pbeh,q["q95"])["false_safe_rate_P_pred_safe_given_true_unsafe"]})
            candidate_rows.append({**label, **aggregate, "Q95_false_safe_rate": boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"]})
            boundary_rows.extend({**label, "reference": reference, "reference_N": q[reference], **values} for reference, values in boundaries.items())
            state_rows.extend({**label, **record} for record in records)
            vectors[name][seed] = metric_vector(ycan, pcan, records, q["q95"])
            predictions[name][seed] = {"truth": ycan, "prediction": pcan}
            for stratum, mask in (("no_current_contact", ~contact), ("current_contact", contact)):
                indices = np.flatnonzero(mask[::21])
                if len(indices) == 0:
                    raise RuntimeError(f"Empty contact stratum: {stratum}")
                selected = np.concatenate([np.arange(i*21,(i+1)*21) for i in indices])
                b = classification(ycan[selected], pcan[selected], q["q95"])
                strata_rows.append({**label, "stratum": stratum, "state_count": len(indices),
                                    **regression(ycan[selected],pcan[selected]),
                                    "mean_pairwise_accuracy": float(np.nanmean([records[i]["pairwise_accuracy"] if records[i]["pairwise_accuracy"] is not None else np.nan for i in indices])),
                                    "mean_safe_choice_regret_N": float(np.mean([records[i]["safe_choice_regret_N"] for i in indices])),
                                    "Q95_true_exceedances": b["true_exceedances"], "Q95_false_safe_count": b["FN_false_safe"],
                                    "Q95_false_safe_rate": b["false_safe_rate_P_pred_safe_given_true_unsafe"],
                                    "Q95_AUROC": b["AUROC"], "Q95_AUPRC": b["AUPRC"]})
            no_contact = ~contact
            b = classification(ycan[no_contact], pcan[no_contact], q["q95"])
            emergence_rows.append({**label, "definition": "pre-action P[34]==0 and next-22-substep force>G2 Q95 nominal N",
                                   "no_contact_states": int(np.sum(no_contact[::21])), "no_contact_candidate_count": int(np.sum(no_contact)),
                                   "new_high_force_candidates": b["true_exceedances"], "predicted_high_force_candidates": b["predicted_exceedances"],
                                   "false_safe_count": b["FN_false_safe"], "false_safe_rate": b["false_safe_rate_P_pred_safe_given_true_unsafe"],
                                   "unsafe_recall": b["unsafe_recall"], "AUROC": b["AUROC"], "AUPRC": b["AUPRC"]})
    write_csv(OUT / "development_behavior_metrics.csv", behavior_rows)
    write_csv(OUT / "development_candidate_metrics.csv", candidate_rows)
    write_csv(OUT / "development_boundary_metrics.csv", boundary_rows)
    write_csv(OUT / "development_state_metrics.csv", state_rows)
    write_csv(OUT / "contact_conditioned_metrics.csv", strata_rows)
    write_csv(OUT / "no_contact_emergence_metrics.csv", emergence_rows)
    write_csv(OUT / "counterfactual_secondary_metrics.csv", [row for row in candidate_rows if row["model"] in ("Z", "P", "Pmin", "Z1", "P1", "Pmin1")])
    for filename, comparisons in (("single_group_addition_metrics.csv", ADDITIONS), ("leave_one_group_out_metrics.csv", REMOVALS)):
        rows = []
        for group, (left, right) in comparisons.items():
            for seed in SEEDS:
                a, b = sampled_metrics(vectors[left][seed], np.arange(120)), sampled_metrics(vectors[right][seed], np.arange(120))
                rows.append({"group": group, "left": left, "right": right, "training_seed": seed,
                             **{f"{field}_left": a[field] for field in FIELDS},
                             **{f"{field}_right": b[field] for field in FIELDS},
                             **{f"{field}_right_minus_left": b[field]-a[field] for field in FIELDS}})
        write_csv(OUT / filename, rows)
    new_json(OUT / "feature_group_bootstrap.json", bootstrap(vectors, len(states)))
    new_json(OUT / "development_evaluation_summary.json", {"status": "COMPLETE_G5_ALL_PRIMARY_SECONDARY_AND_ALIASES",
        "models": list(MODELS), "development_behavior_rows": 2111, "development_candidate_states": 120,
        "development_candidate_labels": 2520, "preaction_contact_states": int(np.sum(contact_reference[::21])),
        "preaction_no_contact_states": int(np.sum(~contact_reference[::21])), "P_min_groups": list(groups),
        "quantiles_nominal_N": q, "confirmation_opened_before_development_freeze": False})
    print("G5 development metrics and 1000 paired-state bootstrap complete; confirmation untouched", flush=True)


if __name__ == "__main__":
    main()
