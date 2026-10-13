"""Exact candidate-cutoff metrics for independent frozen G6A replicas."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import beta

from g6_prepare import OUT, SEEDS


def load_rows(group):
    path = OUT / f"{group}_candidate_results.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2520:
        raise RuntimeError(f"G6A {group} candidate row count changed")
    state = np.asarray([int(r["state_id"]) for r in rows],dtype=int)
    candidate = np.asarray([int(r["candidate_id"]) for r in rows],dtype=int)
    if not np.array_equal(state,np.repeat(np.arange(120),21)) or not np.array_equal(candidate,np.tile(np.arange(21),120)):
        raise RuntimeError(f"G6A {group} candidate order changed")
    truth=np.asarray([float(r["F_true_N"]) for r in rows],dtype=np.float64)
    contact=np.asarray([bool(int(r["current_contact"])) for r in rows],dtype=bool)
    pred={seed:np.asarray([float(r[f"prediction_seed_{seed}_N"]) for r in rows],dtype=np.float64) for seed in SEEDS}
    if not np.all(np.isfinite(truth)) or np.any(truth<0) or any(not np.all(np.isfinite(v)) for v in pred.values()):
        raise RuntimeError("Nonfinite/invalid G6A candidate data")
    return {"truth":truth,"current_contact":contact,"prediction":pred,"rows":rows}


def exact_ci(count, denominator):
    if denominator == 0:
        return None
    alpha=.05
    low=0. if count==0 else float(beta.ppf(alpha/2,count,denominator-count+1))
    high=1. if count==denominator else float(beta.ppf(1-alpha/2,count+1,denominator-count))
    return [low,high]


def severity(truth, prediction, cutoff, limit):
    missed=truth[(truth>limit)&(prediction<=cutoff)]
    excess=missed-limit
    if not len(excess):
        return {"count":0,"mean_N":None,"median_N":None,"q90_N":None,"max_N":None,"max_true_force_N":None}
    return {"count":int(len(excess)),"mean_N":float(np.mean(excess)),"median_N":float(np.median(excess)),
            "q90_N":float(np.quantile(excess,.9)),"max_N":float(np.max(excess)),"max_true_force_N":float(np.max(missed))}


def metrics(truth,prediction,cutoff,limit,include_severity=True):
    unsafe=truth>limit
    predicted_safe=prediction<=cutoff
    true_unsafe=int(np.sum(unsafe))
    true_safe=int(np.sum(~unsafe))
    fn=int(np.sum(unsafe&predicted_safe))
    tp=true_unsafe-fn
    tn=int(np.sum(~unsafe&predicted_safe))
    fp=true_safe-tn
    accepted=fn+tn
    rejected=tp+fp
    result={"candidate_count":len(truth),"true_unsafe_count":true_unsafe,"true_safe_count":true_safe,
            "false_safe_count":fn,"false_safe_rate":fn/true_unsafe if true_unsafe else None,
            "false_safe_CI95_exact":exact_ci(fn,true_unsafe),
            "false_unsafe_count":fp,"false_unsafe_rate":fp/true_safe if true_safe else None,
            "accepted_count":accepted,"acceptance_rate":accepted/len(truth),
            "rejected_count":rejected,"rejection_rate":rejected/len(truth),
            "unsafe_recall":tp/true_unsafe if true_unsafe else None,
            "precision":tp/rejected if rejected else None,
            "specificity":tn/true_safe if true_safe else None,
            "NPV":tn/accepted if accepted else None,
            "TP":tp,"FN":fn,"FP":fp,"TN":tn}
    if include_severity:
        result["false_safe_severity"] = severity(truth,prediction,cutoff,limit)
    return result


def availability(prediction,cutoff):
    counts=np.sum((prediction<=cutoff).reshape(120,21),axis=1)
    return {"state_count":120,"states_with_at_least_one_accepted":int(np.sum(counts>=1)),
            "fraction_states_with_at_least_one_accepted":float(np.mean(counts>=1)),
            "states_with_zero_accepted":int(np.sum(counts==0)),
            "fraction_states_with_zero_accepted":float(np.mean(counts==0)),
            "median_accepted_actions_per_state":float(np.median(counts)),
            "mean_accepted_actions_per_state":float(np.mean(counts)),
            "accepted_action_count_quantiles":{str(q):float(np.quantile(counts,q)) for q in (0,.1,.25,.5,.75,.9,1)}}


def contact_emergence(truth,prediction,current_contact,cutoff,limit):
    no_contact=~current_contact
    positive=no_contact&(truth>limit)
    n=int(np.sum(positive))
    missed=int(np.sum(positive&(prediction<=cutoff)))
    return {"no_current_contact_states":int(np.sum(no_contact[::21])),
            "no_current_contact_candidate_count":int(np.sum(no_contact)),
            "no_contact_to_Q95_high_force_count":n,
            "false_safe_count":missed,"false_safe_rate":missed/n if n else None,
            "false_safe_CI95_exact":exact_ci(missed,n),
            "accepted_high_force_count":missed,"rejected_high_force_count":n-missed,
            "all_no_contact_accepted_count":int(np.sum(no_contact&(prediction<=cutoff))),
            "all_no_contact_rejected_count":int(np.sum(no_contact&(prediction>cutoff)))}
