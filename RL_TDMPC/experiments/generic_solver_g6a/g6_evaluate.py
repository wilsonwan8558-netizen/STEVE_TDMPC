"""Untouched G6A confirmation evaluation at raw and frozen shared cutoffs."""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g6_prepare import OUT, SEEDS, verify_source, new_json, sha
from g6_metrics import load_rows, metrics, availability, contact_emergence
from g6_freeze import verify


def summary(per_seed,fields):
    result={}
    for field in fields:
        entries={str(seed):per_seed[str(seed)][field] for seed in SEEDS}
        values=np.asarray([v for v in entries.values() if v is not None],dtype=float)
        result[field]={"individual":entries,
                       "mean":float(np.mean(values)) if len(values) else None,
                       "sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                       "minimum":float(np.min(values)) if len(values) else None,
                       "maximum":float(np.max(values)) if len(values) else None}
    return result


def main():
    verify()
    q=verify_source()
    limit=q["q95"]
    selected=json.loads((OUT/"selected_cutoff.json").read_text())
    cutoff=float(selected["c_pred_N"])
    if selected["F_lim_N"]!=limit or cutoff>limit:
        raise RuntimeError("Frozen G6A cutoff/reference differs")
    audit=json.loads((OUT/"confirmation_scoring_audit.json").read_text())
    if audit["status"]!="PASS_THREE_FROZEN_REPLICA_PREDICTIONS" or audit["candidate_rows"]!=2520:
        raise RuntimeError("Untouched confirmation scoring incomplete")
    data=load_rows("confirmation")
    raw={str(seed):metrics(data["truth"],data["prediction"][seed],limit,limit) for seed in SEEDS}
    calibrated={str(seed):metrics(data["truth"],data["prediction"][seed],cutoff,limit) for seed in SEEDS}
    if len({raw[str(seed)]["true_unsafe_count"] for seed in SEEDS})!=1:
        raise RuntimeError("Replicas do not share confirmation labels")
    fields=("false_safe_rate","false_unsafe_rate","acceptance_rate","rejection_rate","unsafe_recall","specificity","NPV")
    calibration=json.loads((OUT/"calibration_tradeoff.json").read_text())
    confirmation_summary={"status":"COMPLETE_FROZEN_THREE_REPLICA_UNTOUCHED_CONFIRMATION",
        "seeds":list(range(96000,96030)),"candidate_states":120,"candidate_labels":2520,
        "true_unsafe_count":raw[str(SEEDS[0])]["true_unsafe_count"],"F_lim_N":limit,"raw_cutoff_N":limit,
        "shared_calibrated_cutoff_N":cutoff,"Delta_N":limit-cutoff,
        "raw_by_replica":raw,"calibrated_by_replica":calibrated,
        "raw_three_seed_summary":summary(raw,fields),"calibrated_three_seed_summary":summary(calibrated,fields),
        "raw_worst_replica_false_safe_rate":max(raw[str(seed)]["false_safe_rate"] for seed in SEEDS),
        "calibrated_worst_replica_false_safe_rate":max(calibrated[str(seed)]["false_safe_rate"] for seed in SEEDS),
        "calibrated_worst_replica_acceptance_rate":min(calibrated[str(seed)]["acceptance_rate"] for seed in SEEDS),
        "calibration_vs_confirmation":{str(seed):{
            "calibration_false_safe_rate":calibration["selected_by_replica"][str(seed)]["false_safe_rate"],
            "confirmation_false_safe_rate":calibrated[str(seed)]["false_safe_rate"],
            "calibration_acceptance_rate":calibration["selected_by_replica"][str(seed)]["acceptance_rate"],
            "confirmation_acceptance_rate":calibrated[str(seed)]["acceptance_rate"]} for seed in SEEDS},
        "no_checkpoint_or_cutoff_change_after_freeze":True,
        "confirmation_candidate_results_sha256":sha(OUT/"confirmation_candidate_results.csv")}
    new_json(OUT/"confirmation_summary.json",confirmation_summary)
    contact={"F_lim_N":limit,"raw_cutoff_N":limit,"shared_calibrated_cutoff_N":cutoff,
        "raw_by_replica":{str(seed):contact_emergence(data["truth"],data["prediction"][seed],data["current_contact"],limit,limit) for seed in SEEDS},
        "calibrated_by_replica":{str(seed):contact_emergence(data["truth"],data["prediction"][seed],data["current_contact"],cutoff,limit) for seed in SEEDS}}
    new_json(OUT/"confirmation_contact_emergence.json",contact)
    new_json(OUT/"confirmation_false_safe_severity.json",{"F_lim_N":limit,"raw_cutoff_N":limit,"selected_cutoff_N":cutoff,
        "raw_by_replica":{str(seed):raw[str(seed)]["false_safe_severity"] for seed in SEEDS},
        "calibrated_by_replica":{str(seed):calibrated[str(seed)]["false_safe_severity"] for seed in SEEDS}})
    calibration_availability=json.loads((OUT/"calibration_action_availability.json").read_text())["by_replica"]
    confirmation_availability={str(seed):availability(data["prediction"][seed],cutoff) for seed in SEEDS}
    new_json(OUT/"state_action_availability.json",{"shared_calibrated_cutoff_N":cutoff,
        "calibration_by_replica":calibration_availability,"confirmation_by_replica":confirmation_availability,
        "calibration_worst_replica_fraction_states_with_accepted_action":min(calibration_availability[str(seed)]["fraction_states_with_at_least_one_accepted"] for seed in SEEDS),
        "confirmation_worst_replica_fraction_states_with_accepted_action":min(confirmation_availability[str(seed)]["fraction_states_with_at_least_one_accepted"] for seed in SEEDS),
        "confirmation_maximum_zero_accepted_state_fraction":max(confirmation_availability[str(seed)]["fraction_states_with_zero_accepted"] for seed in SEEDS)})
    new_json(OUT/"confirmation_reference_diagnostics.json",{"purpose":"descriptive only; Q95 cutoff unchanged",
        "selected_cutoff_N":cutoff,"by_reference":{reference:{"true_reference_N":value,
            "by_replica":{str(seed):metrics(data["truth"],data["prediction"][seed],cutoff,value) for seed in SEEDS}}
            for reference,value in q.items() if reference!="q95"}})
    rows=[]
    for seed in SEEDS:
        for name,measurement in (("raw",raw[str(seed)]),("calibrated",calibrated[str(seed)])):
            sev=measurement["false_safe_severity"]
            rows.append({"training_seed":seed,"cutoff_type":name,"c_pred_N":limit if name=="raw" else cutoff,
                **{k:v for k,v in measurement.items() if k not in ("false_safe_CI95_exact","false_safe_severity")},
                "false_safe_CI95_low":measurement["false_safe_CI95_exact"][0],
                "false_safe_CI95_high":measurement["false_safe_CI95_exact"][1],
                **{f"severity_{k}":v for k,v in sev.items()}})
    path=OUT/"confirmation_metrics.csv"
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,5))
    for i,seed in enumerate(SEEDS):
        x=[raw[str(seed)]["acceptance_rate"],calibrated[str(seed)]["acceptance_rate"]]
        y=[raw[str(seed)]["false_safe_rate"],calibrated[str(seed)]["false_safe_rate"]]
        ax.plot(x,y,marker="o",label=str(seed))
    ax.axhline(.05,color="black",linestyle="--",label="calibration empirical 5% target")
    ax.set(xlabel="Candidate acceptance rate",ylabel="Q95 false-safe rate",title="Untouched G6A: raw → frozen calibrated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/"plots/confirmation_raw_vs_calibrated.png",dpi=160)
    plt.close(fig)
    print(f"G6A untouched three-replica confirmation complete; worst FSR={confirmation_summary['calibrated_worst_replica_false_safe_rate']:.4f}",flush=True)


if __name__=="__main__":
    main()
