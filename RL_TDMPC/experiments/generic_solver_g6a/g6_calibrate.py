"""Select one exact-breakpoint cutoff using worst of three frozen replicas."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g6_prepare import OUT, SEEDS, verify_source, sha, new_json
from g6_metrics import load_rows, metrics, availability, contact_emergence


def three_seed_summary(per_seed, fields):
    result={}
    for field in fields:
        entries={str(seed):per_seed[str(seed)][field] for seed in SEEDS}
        values=np.asarray([v for v in entries.values() if v is not None],dtype=float)
        result[field]={"per_seed":entries,
                       "mean":float(np.mean(values)) if len(values) else None,
                       "sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                       "minimum":float(np.min(values)) if len(values) else None,
                       "maximum":float(np.max(values)) if len(values) else None}
    return result


def cutoff_metrics(data,cutoff,limit):
    return {str(seed):metrics(data["truth"],data["prediction"][seed],cutoff,limit) for seed in SEEDS}


def flat_row(cutoff,seed,m):
    s=m["false_safe_severity"]
    ci=m["false_safe_CI95_exact"]
    return {"c_pred_N":cutoff,"training_seed":seed,"true_unsafe_count":m["true_unsafe_count"],
        "false_safe_count":m["false_safe_count"],"false_safe_rate":m["false_safe_rate"],
        "false_safe_CI95_low":ci[0] if ci else None,"false_safe_CI95_high":ci[1] if ci else None,
        "false_unsafe_count":m["false_unsafe_count"],"false_unsafe_rate":m["false_unsafe_rate"],
        "accepted_count":m["accepted_count"],"acceptance_rate":m["acceptance_rate"],
        "rejected_count":m["rejected_count"],"rejection_rate":m["rejection_rate"],
        "unsafe_recall":m["unsafe_recall"],"precision":m["precision"],"specificity":m["specificity"],"NPV":m["NPV"],
        "false_safe_severity_mean_N":s["mean_N"],"false_safe_severity_median_N":s["median_N"],
        "false_safe_severity_q90_N":s["q90_N"],"false_safe_severity_max_N":s["max_N"],
        "maximum_true_force_among_false_safe_N":s["max_true_force_N"]}


def main():
    if (OUT/"selected_cutoff.json").exists() or (OUT/"pre_confirmation_freeze.json").exists():
        raise FileExistsError("G6A cutoff already selected/frozen")
    q=verify_source()
    limit=q["q95"]
    audit=json.loads((OUT/"calibration_scoring_audit.json").read_text())
    if audit["status"]!="PASS_THREE_FROZEN_REPLICA_PREDICTIONS" or audit["candidate_rows"]!=2520:
        raise RuntimeError("Calibration scoring incomplete")
    data=load_rows("calibration")
    if any(int(r["true_Q95_high_force"])!=int(y>limit) for r,y in zip(data["rows"],data["truth"])):
        raise RuntimeError("Frozen G2 Q95 truth semantics changed")
    positives=int(np.sum(data["truth"]>limit))
    if positives==0:
        raise RuntimeError("No Q95-positive calibration actions; cannot select cutoff")
    predictions=np.concatenate([data["prediction"][seed] for seed in SEEDS])
    finite=np.unique(predictions[predictions<=limit])
    if not len(finite):
        cutoffs=np.unique(np.asarray([np.nextafter(float(np.min(predictions)),-np.inf),limit],dtype=np.float64))
    else:
        sentinel=np.nextafter(float(np.min(finite)),-np.inf)
        cutoffs=np.unique(np.r_[sentinel,finite,limit]).astype(np.float64)
    # Full exact-event sweep; every candidate prediction at/below F_lim is a breakpoint.
    sweep=[]
    passing=[]
    for cutoff in cutoffs:
        per_seed=cutoff_metrics(data,float(cutoff),limit)
        worst=max(per_seed[str(seed)]["false_safe_rate"] for seed in SEEDS)
        passing.append(worst<=.05+1e-15)
        across={}
        for field in ("false_safe_rate","false_unsafe_rate","acceptance_rate","rejection_rate",
                      "unsafe_recall","precision","specificity","NPV"):
            values=np.asarray([per_seed[str(seed)][field] for seed in SEEDS if per_seed[str(seed)][field] is not None],dtype=float)
            across.update({f"replica_{field}_mean":float(np.mean(values)) if len(values) else None,
                           f"replica_{field}_sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                           f"replica_{field}_minimum":float(np.min(values)) if len(values) else None,
                           f"replica_{field}_maximum":float(np.max(values)) if len(values) else None})
        for seed in SEEDS:
            sweep.append({**flat_row(float(cutoff),seed,per_seed[str(seed)]),"worst_replica_false_safe_rate":worst,
                          "all_replicas_meet_5pct":int(worst<=.05+1e-15),**across})
    if not any(passing):
        raise RuntimeError("Even all-rejected sentinel failed robust FSR rule: inconsistent metric implementation")
    index=max(i for i,ok in enumerate(passing) if ok)
    if index==len(cutoffs)-1:
        selected=float(limit)
        selection_detail="raw G2 Q95 cutoff itself passed"
    else:
        first_failing=float(cutoffs[index+1])
        selected=float(np.nextafter(first_failing,-np.inf))
        selection_detail="largest float64 below first failing exact prediction breakpoint"
    if selected>limit or not np.isfinite(selected):
        raise RuntimeError("Selected cutoff out of scope")
    selected_metrics=cutoff_metrics(data,selected,limit)
    raw_metrics=cutoff_metrics(data,limit,limit)
    worst=max(selected_metrics[str(seed)]["false_safe_rate"] for seed in SEEDS)
    if worst>.05+1e-15:
        raise RuntimeError("Robust shared cutoff selection failed")
    for seed in SEEDS:
        if selected_metrics[str(seed)]["false_safe_count"]!=sweep[index*3+list(SEEDS).index(seed)]["false_safe_count"]:
            raise RuntimeError("Predecessor cutoff changed classification unexpectedly")
    availability_by_seed={str(seed):availability(data["prediction"][seed],selected) for seed in SEEDS}
    worst_accept=min(selected_metrics[str(seed)]["acceptance_rate"] for seed in SEEDS)
    worst_state_availability=min(availability_by_seed[str(seed)]["fraction_states_with_at_least_one_accepted"] for seed in SEEDS)
    selected_obj={"status":"SHARED_WORST_REPLICA_Q95_CUTOFF_SELECTED_ON_CALIBRATION_ONLY",
        "F_lim_N":limit,"c_pred_N":selected,"Delta_N":limit-selected,
        "selection_rule":"largest float64 cutoff <=F_lim with every frozen replica empirical Q95 false-safe rate <=5%",
        "exact_event_breakpoint_count":len(cutoffs),"selection_detail":selection_detail,
        "last_passing_event_cutoff_N":float(cutoffs[index]),
        "next_failing_prediction_N":float(cutoffs[index+1]) if index<len(cutoffs)-1 else None,
        "calibration_true_unsafe_count":positives,"calibration_metrics_by_replica":selected_metrics,
        "worst_replica_false_safe_rate":worst,"worst_replica_acceptance_rate":worst_accept,
        "availability_by_replica":availability_by_seed,"worst_replica_fraction_states_with_accepted_action":worst_state_availability,
        "all_replicas_meet_empirical_5pct":True,
        "degenerate_all_candidates_rejected_any_replica":any(selected_metrics[str(seed)]["accepted_count"]==0 for seed in SEEDS),
        "confirmation_seeds_opened_before_selection":False,
        "calibration_candidate_results_sha256":sha(OUT/"calibration_candidate_results.csv")}
    path=OUT/"cutoff_sweep.csv"
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    new_json(OUT/"selected_cutoff.json",selected_obj)
    fields=("false_safe_rate","false_unsafe_rate","acceptance_rate","rejection_rate","unsafe_recall","specificity","NPV")
    new_json(OUT/"calibration_tradeoff.json",{"status":"EXACT_EVENT_SWEEP_COMPLETE",
        "cutoff_count":len(cutoffs),"sweep_rows":len(sweep),"selection_primary":"max per-replica FSR <=5%; no mean/pooled selection",
        "raw_cutoff_N":limit,"selected_cutoff_N":selected,
        "raw_by_replica":raw_metrics,"selected_by_replica":selected_metrics,
        "raw_three_seed_summary":three_seed_summary(raw_metrics,fields),
        "selected_three_seed_summary":three_seed_summary(selected_metrics,fields),
        "selected_worst_replica_FSR":worst,"selected_worst_replica_acceptance_rate":worst_accept,
        "sweep_sha256":sha(path)})
    new_json(OUT/"calibration_false_safe_severity.json",{"F_lim_N":limit,"raw_cutoff_N":limit,"selected_cutoff_N":selected,
        "raw_by_replica":{str(seed):raw_metrics[str(seed)]["false_safe_severity"] for seed in SEEDS},
        "selected_by_replica":{str(seed):selected_metrics[str(seed)]["false_safe_severity"] for seed in SEEDS}})
    new_json(OUT/"calibration_contact_emergence.json",{"F_lim_N":limit,"raw_cutoff_N":limit,"selected_cutoff_N":selected,
        "raw_by_replica":{str(seed):contact_emergence(data["truth"],data["prediction"][seed],data["current_contact"],limit,limit) for seed in SEEDS},
        "selected_by_replica":{str(seed):contact_emergence(data["truth"],data["prediction"][seed],data["current_contact"],selected,limit) for seed in SEEDS}})
    new_json(OUT/"calibration_reference_diagnostics.json",{"purpose":"descriptive only; Q95 cutoff not retuned",
        "selected_cutoff_N":selected,"by_reference":{reference:{"true_reference_N":value,
            "by_replica":{str(seed):metrics(data["truth"],data["prediction"][seed],selected,value) for seed in SEEDS}}
            for reference,value in q.items() if reference!="q95"}})
    new_json(OUT/"calibration_action_availability.json",{"selected_cutoff_N":selected,"by_replica":availability_by_seed,
        "worst_replica_fraction_states_with_accepted_action":worst_state_availability,
        "worst_replica_acceptance_rate":worst_accept})
    plots=OUT/"plots"
    plots.mkdir(exist_ok=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for seed in SEEDS:
        rows=[r for r in sweep if r["training_seed"]==seed]
        ax.plot([r["acceptance_rate"] for r in rows],[r["false_safe_rate"] for r in rows],label=str(seed),linewidth=1.2)
        ax.scatter([selected_metrics[str(seed)]["acceptance_rate"]],[selected_metrics[str(seed)]["false_safe_rate"]],s=45)
    ax.axhline(.05,color="black",linestyle="--",linewidth=1,label="empirical 5% target")
    ax.set(xlabel="Candidate acceptance rate",ylabel="Q95 false-safe rate",title="G6A calibration: exact-breakpoint tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots/"calibration_acceptance_vs_false_safe.png",dpi=160)
    plt.close(fig)
    print(f"G6A calibration shared cutoff={selected:.12g} Delta={limit-selected:.12g}; worst FSR={worst:.4f}; worst acceptance={worst_accept:.4f}",flush=True)


if __name__=="__main__":
    main()
