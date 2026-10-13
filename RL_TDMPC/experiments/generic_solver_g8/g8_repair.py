"""Frozen discrete nearest-to-nominal action repair; never executes an action."""

from __future__ import annotations

import numpy as np


def validate_candidates(candidates,behavior_action):
    if len(candidates)!=21 or [int(c["candidate_id"]) for c in candidates]!=list(range(21)):
        raise RuntimeError("Frozen 21-action candidate order changed")
    actions=np.asarray([c["action"] for c in candidates],dtype=np.float32)
    behavior=np.asarray(behavior_action,dtype=np.float32)
    if (actions.shape!=(21,2) or behavior.shape!=(2,) or not np.array_equal(actions[0],behavior) or
            candidates[0]["kind"]!="behavior" or np.any(~np.isfinite(actions)) or np.any(np.abs(actions)>1+1e-6)):
        raise RuntimeError("Candidate 0 is not the exact normalized nominal action")
    if len({tuple(map(float,a)) for a in actions})!=21:
        raise RuntimeError("Frozen candidate set contains duplicates")
    return actions.astype(np.float64)


def nearest(safe_ids,actions):
    """Euclidean distance in normalized translation/rotation; candidate-id tie break."""
    return min((int(i) for i in safe_ids),key=lambda i:(float(np.linalg.norm(actions[i]-actions[0])),i))


def select(predicted,cutoff,actions):
    predicted=np.asarray(predicted,dtype=np.float64)
    if predicted.shape!=(21,) or not np.all(np.isfinite(predicted)):
        raise RuntimeError("Invalid frozen predictor scores")
    safe=np.flatnonzero(predicted<=cutoff).astype(int).tolist()
    if 0 in safe:
        chosen=0;case="nominal_predicted_safe"
    elif safe:
        chosen=nearest(safe,actions);case="nearest_predicted_safe_repair"
    else:
        chosen=None;case="predicted_infeasible"
    return {"selected_candidate_id":chosen,"selection_case":case,"predicted_safe_ids":safe,
        "predicted_safe_count":len(safe),"intervention":chosen is not None and chosen!=0}


def oracle(truth,limit,actions):
    truth=np.asarray(truth,dtype=np.float64)
    if truth.shape!=(21,) or np.any(~np.isfinite(truth)) or np.any(truth<0):
        raise RuntimeError("Invalid one-step SOFA truth")
    safe=np.flatnonzero(truth<=limit).astype(int).tolist()
    if 0 in safe:chosen=0;case="nominal_true_safe"
    elif safe:chosen=nearest(safe,actions);case="nearest_true_safe_repair"
    else:chosen=None;case="oracle_infeasible"
    return {"selected_candidate_id":chosen,"selection_case":case,"true_safe_ids":safe,
        "true_safe_count":len(safe),"intervention":chosen is not None and chosen!=0}
