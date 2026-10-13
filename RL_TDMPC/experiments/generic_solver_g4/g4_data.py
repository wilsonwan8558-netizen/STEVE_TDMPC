"""Exact matched G4 development matrices; no confirmation data access."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
G3 = RL / "logs/generic_solver_phase_g3_force_predictor"
OUT = RL / "logs/generic_solver_phase_g4_state_observability"
REPRESENTATIONS = ("Z", "O", "H", "P")
GROUPS = ("train", "validation", "development_test")
STATE_DIMS = {"Z": 256, "O": 14, "H": 30, "P": 38}


def source():
    manifest = json.loads((G3 / "data_manifest.json").read_text())
    g4_audit = json.loads((OUT / "g3_data_audit.json").read_text())
    behavior_hash = hashlib.sha256((G3 / "behavior_features.npz").read_bytes()).hexdigest()
    split_hash = hashlib.sha256((G3 / "development_split.json").read_bytes()).hexdigest()
    if behavior_hash != manifest["behavior_features_sha256"] or behavior_hash != g4_audit["g3_behavior_features_sha256"]:
        raise RuntimeError("G3 behavior source changed")
    if split_hash != g4_audit["g3_split_sha256"]:
        raise RuntimeError("Frozen G3 episode split changed")
    digest = hashlib.sha256()
    for group, expected in (("train", 560), ("validation", 120), ("development_test", 120)):
        paths = sorted((G3 / "development_candidates" / group).glob("state_*.json"))
        if len(paths) != expected:
            raise RuntimeError(f"G3 {group} candidate state count changed")
        for path in paths:
            digest.update(str(path.relative_to(G3)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    if digest.hexdigest() != manifest["development_candidate_all_files_sha256"]:
        raise RuntimeError("Frozen G3 development candidate labels changed")
    with np.load(G3 / "behavior_features.npz") as source_file:
        source_arrays = {name: source_file[name].copy() for name in source_file.files}
    split = json.loads((G3 / "development_split.json").read_text())
    # Keyed by the actually executed same-episode predecessor, never by a
    # shuffled row offset or a post-action/next observation.
    index = {(int(seed), int(d)): i for i, (seed, d) in enumerate(zip(source_arrays["seed"], source_arrays["decision_index"]))}
    if len(index) != 15500:
        raise RuntimeError("G3 behavior keys incomplete or duplicated")
    return source_arrays, split, index


def physical_lookup():
    status = json.loads((OUT / "physical_state_alignment_audit.json").read_text())
    if status["status"] != "PASS_G3_G2_PARENT_CANDIDATE_AND_PREACTION_PHYSICAL_ALIGNMENT":
        raise RuntimeError("P alignment gate has not passed")
    lookup = {}
    for seed in range(90000, 90200):
        with np.load(OUT / "development_physical_by_episode" / f"seed_{seed}.npz") as data:
            for d, p in zip(data["decision_index"], data["physical"]):
                key = (seed, int(d))
                if key in lookup:
                    raise RuntimeError(f"Duplicate P key {key}")
                lookup[key] = p.copy()
    if len(lookup) != 15300:
        raise RuntimeError(f"Expected 15300 P rows, got {len(lookup)}")
    return lookup


def state_vector(repr_name, seed, d, source_arrays, index, physical):
    key = (int(seed), int(d))
    if d < 1 or key not in index:
        raise RuntimeError(f"Invalid common G4 row {key}")
    current = index[key]
    if repr_name == "Z":
        vector = source_arrays["latent"][current]
    elif repr_name == "O":
        vector = source_arrays["observation"][current]
    elif repr_name == "H":
        previous = index.get((int(seed), int(d)-1))
        if previous is None:
            raise RuntimeError(f"Missing same-episode predecessor {key}")
        vector = np.r_[source_arrays["observation"][previous], source_arrays["observation"][current], source_arrays["action"][previous]]
    elif repr_name == "P":
        vector = physical[key]
    else:
        raise ValueError(repr_name)
    if np.shape(vector) != (STATE_DIMS[repr_name],) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Invalid {repr_name} vector {key}")
    return np.asarray(vector, dtype=np.float64)


def behavior_matrix(repr_name, group, source_arrays, split, index, physical):
    allowed = set(split[group])
    keys = [(int(seed), int(d)) for seed, d in zip(source_arrays["seed"], source_arrays["decision_index"])
            if int(seed) in allowed and int(d) >= 1]
    x = np.asarray([np.r_[state_vector(repr_name, seed, d, source_arrays, index, physical),
                           source_arrays["action"][index[(seed, d)]].astype(float)] for seed, d in keys], dtype=np.float64)
    y = np.asarray([source_arrays["force_next_max_N"][index[key]] for key in keys], dtype=np.float64)
    if x.shape != (len(keys), STATE_DIMS[repr_name]+2) or not np.all(np.isfinite(y)):
        raise RuntimeError(f"Invalid {repr_name}/{group} behavior matrix")
    return x, y, keys


def candidate_states(group, split):
    paths = sorted((G3 / "development_candidates" / group).glob("state_*.json"))
    expected = 560 if group == "train" else 120
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} states for {group}, got {len(paths)}")
    states = [json.loads(path.read_text()) for path in paths]
    if any(s["group"] != group or s["seed"] not in split[group] or s["decision_index"] < 1 or len(s["candidates"]) != 21 for s in states):
        raise RuntimeError(f"Invalid {group} candidate states")
    return states


def candidate_matrix(repr_name, states, source_arrays, index, physical):
    x, y = [], []
    for state in states:
        vector = state_vector(repr_name, state["seed"], state["decision_index"], source_arrays, index, physical)
        if not np.array_equal(np.asarray(state["observation"], np.float32), source_arrays["observation"][index[(state["seed"], state["decision_index"])]]):
            raise RuntimeError("G3 candidate state not matched to behavior row")
        for candidate in state["candidates"]:
            x.append(np.r_[vector, candidate["action"]])
            y.append(candidate["force_next_max_N"])
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != (21*len(states), STATE_DIMS[repr_name]+2) or np.any(y < 0) or not np.all(np.isfinite(y)):
        raise RuntimeError("Invalid candidate matrix")
    return x, y
