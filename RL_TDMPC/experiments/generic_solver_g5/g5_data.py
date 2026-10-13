"""G5 feature views of the exact G4 development rows and candidate labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
from g4_data import source as g4_source, physical_lookup as g4_physical_lookup, candidate_states as g4_candidate_states  # noqa: E402
from g4_data import behavior_matrix as g4_behavior_matrix, candidate_matrix as g4_candidate_matrix  # noqa: E402
from g5_prepare import OUT, GROUPS  # noqa: E402

PRIMARY = ("Z", "P", "ZP", "ZK", "ZD", "ZC", "ZP_noK", "ZP_noD", "ZP_noC", "PK", "PD", "PKD")
GROUP_FOR_SUBSET = {"K": ("ZK", "PK"), "D": ("ZD", "PD"), "KD": ("ZP_noC", "PKD")}


def selected_indices(groups):
    indices = sorted({i for group in groups for i in GROUPS[group]})
    if len(indices) != sum(len(GROUPS[group]) for group in groups):
        raise RuntimeError("Overlapping G5 feature groups")
    return indices


def spec(name, minimal_groups=None):
    fixed = {
        "Z": (True, ()), "P": (False, ("K", "D", "C")), "ZP": (True, ("K", "D", "C")),
        "ZK": (True, ("K",)), "ZD": (True, ("D",)), "ZC": (True, ("C",)),
        "ZP_noK": (True, ("D", "C")), "ZP_noD": (True, ("K", "C")),
        "ZP_noC": (True, ("K", "D")), "PK": (False, ("K",)),
        "PD": (False, ("D",)), "PKD": (False, ("K", "D")),
    }
    if name in ("Z1", "P1", "Pmin1", "Pmin", "ZPmin"):
        if name == "Z1":
            return spec("Z")
        if name == "P1":
            return spec("P")
        if minimal_groups is None:
            raise RuntimeError("P_min groups must have been selected on validation")
        return (name == "ZPmin", selected_indices(minimal_groups))
    if name not in fixed:
        raise ValueError(name)
    include_z, groups = fixed[name]
    return include_z, selected_indices(groups)


def state_vector(name, seed, d, arrays, index, physical, minimal_groups=None):
    key = (int(seed), int(d))
    if int(d) < 1 or key not in index or key not in physical:
        raise RuntimeError(f"Unmatched G4 decision state {key}")
    include_z, p_indices = spec(name, minimal_groups)
    parts = []
    if include_z:
        parts.append(np.asarray(arrays["latent"][index[key]], dtype=np.float64))
    if p_indices:
        p = np.asarray(physical[key], dtype=np.float64)
        if p.shape != (38,):
            raise RuntimeError(f"G4 P dimension changed {key}")
        parts.append(p[p_indices])
    vector = np.concatenate(parts)
    if vector.shape != ((256 if include_z else 0)+len(p_indices),) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Invalid G5 input state {name}/{key}")
    return vector


def behavior(name, group, arrays, split, index, physical, minimal_groups=None):
    allowed = set(split[group])
    keys = [(int(s), int(d)) for s,d in zip(arrays["seed"], arrays["decision_index"])
            if int(s) in allowed and int(d)>=1]
    x = np.asarray([np.r_[state_vector(name, s, d, arrays, index, physical, minimal_groups),
                           arrays["action"][index[(s,d)]]] for s,d in keys], dtype=np.float64)
    y = np.asarray([arrays["force_next_max_N"][index[k]] for k in keys], dtype=np.float64)
    if len(x) != len(keys) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise RuntimeError("Invalid G5 behavior matrix")
    return x,y,keys


def candidates(name, states, arrays, index, physical, minimal_groups=None):
    x,y,contact = [],[],[]
    for state in states:
        key = (state["seed"],state["decision_index"])
        vector = state_vector(name,*key,arrays,index,physical,minimal_groups)
        current_contact = float(physical[key][34]) > 0
        if len(state["candidates"]) != 21:
            raise RuntimeError("Candidate protocol changed")
        for candidate in state["candidates"]:
            x.append(np.r_[vector,candidate["action"]])
            y.append(candidate["force_next_max_N"])
            contact.append(current_contact)
    x,y,contact = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64),np.asarray(contact,dtype=bool)
    if len(x) != 21*len(states) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(y<0):
        raise RuntimeError("Invalid G5 candidate matrix")
    return x,y,contact


def source():
    arrays,split,index=g4_source()
    physical=g4_physical_lookup()
    return arrays,split,index,physical


def verify_z_p_reference_inputs(arrays,split,index,physical):
    for model in ("Z","P"):
        for group in ("train","validation"):
            actual_x,actual_y,_=behavior(model,group,arrays,split,index,physical)
            g4_x,g4_y,_=g4_behavior_matrix(model,group,arrays,split,index,physical)
            if not np.array_equal(actual_x,g4_x) or not np.array_equal(actual_y,g4_y):
                raise RuntimeError(f"G4 {model}/{group} behavior semantics changed")
        states=g4_candidate_states("train",split)
        actual_x,actual_y,_=candidates(model,states,arrays,index,physical)
        g4_x,g4_y=g4_candidate_matrix(model,states,arrays,index,physical)
        if not np.array_equal(actual_x,g4_x) or not np.array_equal(actual_y,g4_y):
            raise RuntimeError(f"G4 {model} train candidate semantics changed")


def minimal_groups():
    selection=json.loads((OUT/"minimal_subset_selection.json").read_text())
    if selection["status"] != "FROZEN_VALIDATION_ONLY_P_MIN_SELECTED":
        raise RuntimeError("P_min is not frozen")
    return tuple(selection["included_groups"])
