"""Predeclare and audit G9 episode-disjoint direct-horizon behavior data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
from g4_data import source, physical_lookup  # noqa: E402

G2 = RL / "logs/generic_solver_phase_g2_force_baseline"
G3 = RL / "logs/generic_solver_phase_g3_force_predictor"
G4 = RL / "logs/generic_solver_phase_g4_state_observability"
G85 = RL / "logs/generic_solver_phase_g85_continuous_oracle_feasibility"
OUT = RL / "logs/generic_solver_phase_g9_temporal_anticipatory_safety"
HORIZONS = (1, 2, 3, 5)
F_LIM = 3.685331040756903
CONFIRM = list(range(102000,102030))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_new(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def audit_sources():
    summary=json.loads((G85/"recoverability_summary.json").read_text())
    if (summary["status"]!="COMPLETE_TRUE_SOFA_FINITE_CONTINUOUS_ACTION_SEARCH" or
        summary["old21_still_no_safe_found_count"]!=11 or
        json.loads((G85/"search_parent_audit.json").read_text())["status"]!="PASS"):
        raise RuntimeError("G8.5 difficult-state source changed")
    arrays,split,index=source()
    physical=physical_lookup()
    groups=("train","validation","development_test")
    development_seeds=sum((split[name] for name in groups),[])
    if len(development_seeds)!=200 or len(set(development_seeds))!=200:
        raise RuntimeError("G3 episode split changed")
    if any(seed in development_seeds for seed in CONFIRM):
        raise RuntimeError("Confirmation overlaps development")
    with (G2/"step_level_force.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=15500:
        raise RuntimeError("G2 behavior row count changed")
    by_seed={}
    for row in rows:
        seed=int(row["seed"]);d=int(row["step_index"])
        by_seed.setdefault(seed,[]).append(row)
        i=index.get((seed,d))
        if i is None or not np.array_equal(arrays["observation"][i],np.asarray(json.loads(row["decision_observation_json"]),dtype=np.float32)):
            raise RuntimeError(f"G2/G3 pre-action observation mismatch {seed}/{d}")
        if not np.array_equal(arrays["action"][i],np.asarray(json.loads(row["selected_action_json"]),dtype=np.float32)):
            raise RuntimeError(f"G2/G3 selected action mismatch {seed}/{d}")
        if abs(float(arrays["force_next_max_N"][i])-float(row["distal_contact_force_max_N"]))>1e-8:
            raise RuntimeError(f"G2/G3 next-interval force mismatch {seed}/{d}")
    if sorted(by_seed)!=list(range(90000,90200)):
        raise RuntimeError("G2 behavior episodes changed")
    for seed,episode in by_seed.items():
        if [int(row["step_index"]) for row in episode]!=list(range(len(episode))):
            raise RuntimeError(f"G2 decision indexing changed {seed}")
        for d in range(1,len(episode)):
            if not np.array_equal(np.asarray(json.loads(episode[d-1]["next_observation_json"]),dtype=np.float32),
                                  np.asarray(json.loads(episode[d]["decision_observation_json"]),dtype=np.float32)):
                raise RuntimeError(f"G2 same-episode observation continuity failed {seed}/{d}")
    return arrays,split,index,physical,by_seed,summary


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    arrays,split,index,physical,by_seed,summary=audit_sources()
    records={name:[] for name in ("seed","decision_index","history_observation","history_action",
        "latent","privileged_physical","behavior_action_sequence","future_force_sequence")}
    for seed in range(90000,90200):
        episode=by_seed[seed]
        # The common H=5 complete-case cohort is used for every horizon.
        for d in range(2,len(episode)-4):
            keys=[(seed,k) for k in range(d-2,d+5)]
            if any(key not in index for key in keys) or (seed,d) not in physical:
                raise RuntimeError(f"G9 temporal key unavailable {seed}/{d}")
            records["seed"].append(seed)
            records["decision_index"].append(d)
            records["history_observation"].append([arrays["observation"][index[(seed,k)]] for k in range(d-2,d+1)])
            records["history_action"].append([arrays["action"][index[(seed,k)]] for k in range(d-2,d)])
            records["latent"].append(arrays["latent"][index[(seed,d)]])
            records["privileged_physical"].append(physical[(seed,d)])
            records["behavior_action_sequence"].append([arrays["action"][index[(seed,k)]] for k in range(d,d+5)])
            records["future_force_sequence"].append([float(arrays["force_next_max_N"][index[(seed,k)]]) for k in range(d,d+5)])
    matrix={name:np.asarray(values,dtype=(np.int32 if name in ("seed","decision_index") else np.float32))
            for name,values in records.items()}
    n=len(matrix["seed"])
    if (n<14000 or matrix["history_observation"].shape!=(n,3,14) or
        matrix["history_action"].shape!=(n,2,2) or matrix["latent"].shape!=(n,256) or
        matrix["privileged_physical"].shape!=(n,38) or matrix["behavior_action_sequence"].shape!=(n,5,2) or
        matrix["future_force_sequence"].shape!=(n,5) or
        not all(np.all(np.isfinite(v)) for v in matrix.values())):
        raise RuntimeError("G9 behavior temporal matrix invalid")
    # Confirm the first risk component is the already-audited next-interval target.
    for seed,d,force in zip(matrix["seed"],matrix["decision_index"],matrix["future_force_sequence"][:,0]):
        if abs(float(force)-float(by_seed[int(seed)][int(d)]["distal_contact_force_max_N"]))>2e-6:
            raise RuntimeError(f"G9 first future force indexing mismatch {seed}/{d}")
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/"development_temporal_behavior.npz",**matrix)
    counts={name:int(np.sum(np.isin(matrix["seed"],split[name]))) for name in ("train","validation","development_test")}
    json_new(OUT/"dataset_split.json",{"train":split["train"],"validation":split["validation"],
        "development_test":split["development_test"],"untouched_confirmation":CONFIRM,
        "common_H5_complete_case_rows":counts,"episode_disjoint":True,
        "G8_and_G85_101000_series_reserved_for_post_selection_event_analysis":True,
        "confirmation_unopened":True})
    json_new(OUT/"horizon_target_definition.json",{"horizons":HORIZONS,
        "at_decision_d":"F_(t+k) is G2 distal_contact_force_max_N from behavior decision d+k-1",
        "R_H":"max future_force_sequence[0:H] within the same behavior episode",
        "future_force_sequence_retained":True,
        "common_complete_case_cohort":"d>=2 and d+5<=episode length; no cross-episode extension or terminal padding",
        "H1_exact_G2_next_interval_force":True,"force_units":"nominal N", "F_lim_N":F_LIM,
        "action_sequence_input":"realized same-episode behavior actions d:d+H used as an offline conditional training sequence; future actions are never included in the current-state estimator",
        "prospective_evaluation_requirement":"a TD-MPC2 candidate plan requires its own matched true-SOFA counterfactual rollout label; behavior target cannot be paired with an unexecuted plan"})
    json_new(OUT/"difficult_event_manifest.json",{"source":"G8.5 finite-search no-safe-found states",
        "state_ids":summary["old21_still_no_safe_found_state_ids"],"count":11,
        "source_summary_sha256":sha(G85/"recoverability_summary.json"),
        "not_model_selection_data":True})
    json_new(OUT/"metadata.json",{"phase":"G9","branch":"feature/generic-contact-force-safety",
        "study":"direct multi-horizon true-force risk; no recursive predicted-state rollout",
        "G2_step_level_force_sha256":sha(G2/"step_level_force.csv"),
        "G3_behavior_features_sha256":sha(G3/"behavior_features.npz"),
        "G3_split_sha256":sha(G3/"development_split.json"),
        "G4_physical_alignment_audit_sha256":sha(G4/"physical_state_alignment_audit.json"),
        "G85_summary_sha256":sha(G85/"recoverability_summary.json"),
        "temporal_dataset_sha256":sha(OUT/"development_temporal_behavior.npz"),
        "dataset_builder_sha256":sha(__file__),"development_rows":n,
        "confirmation_unopened":True})
    print(f"G9 temporal data prepared: {n} rows, split {counts}; confirmation unopened",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("prepare",),required=True)
    args=parser.parse_args()
    if args.stage=="prepare":prepare()


if __name__=="__main__":main()
