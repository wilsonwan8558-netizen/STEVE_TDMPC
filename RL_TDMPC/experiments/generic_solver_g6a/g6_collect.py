"""G6A ordered Generic-SOFA behavior/replay/fork and three-replica scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from g6_prepare import OUT, G5, MODEL, SEEDS, CALIBRATION, CONFIRMATION, verify_source, sha, new_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g5"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g2"))
import g5_confirm  # noqa: E402
from g5_metrics import load_predictor, predict  # noqa: E402
from collect import evaluate_one, STEP_FIELDS, EPISODE_FIELDS, check_scene  # noqa: E402
from protocol import choose_states, generate_candidates  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from physical_features import extract  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def protocol(group):
    if group not in ("calibration", "confirmation"):
        raise ValueError(group)
    if group == "confirmation":
        from g6_freeze import verify as verify_g6
        verify_g6()
    else:
        verify_source()
    seeds = CALIBRATION if group == "calibration" else CONFIRMATION
    return seeds, OUT / f"{group}_behavior_episodes.csv", OUT / f"{group}_behavior_steps.csv", OUT / f"{group}_candidates"


def setup(seeds):
    # The G5 setup function is the exact frozen Generic scene/controller constructor.
    g5_confirm.CONFIRM = seeds
    return g5_confirm.setup()


def behavior(group):
    seeds, episodes_path, steps_path, _ = protocol(group)
    if episodes_path.exists() or steps_path.exists() or (OUT / f"{group}_seed_manifest.json").exists():
        raise FileExistsError("G6A behavior output already exists")
    manifest = {"seeds":seeds,"seed_protocol_sha256":sha(OUT/f"{group}_seed_protocol.json"),
                "G5_pre_confirmation_freeze_sha256":sha(G5/"pre_confirmation_freeze.json"),
                "scene_lifecycle":"one reused Generic scene, ordered resets", "three_replica_predictions_not_fed_to_policy":True}
    if group == "confirmation":
        manifest["G6A_pre_confirmation_freeze_sha256"] = sha(OUT/"pre_confirmation_freeze.json")
    new_json(OUT/f"{group}_seed_manifest.json",manifest)
    env, agent, monitor = setup(seeds)
    try:
        with episodes_path.open("w",newline="") as ep_stream, steps_path.open("w",newline="") as step_stream:
            ep_writer, step_writer = csv.DictWriter(ep_stream,EPISODE_FIELDS),csv.DictWriter(step_stream,STEP_FIELDS)
            ep_writer.writeheader()
            step_writer.writeheader()
            for i,seed in enumerate(seeds):
                episode,steps = evaluate_one(env,agent,seed,i,g5_confirm.EXPECTED_CHECKPOINT_SHA,monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"G6A {group} simulation error {seed}")
                ep_writer.writerow(episode)
                step_writer.writerows(steps)
                ep_stream.flush()
                step_stream.flush()
                print(f"G6A {group} behavior seed={seed} steps={len(steps)}",flush=True)
    finally:
        env.close()


def candidates(group):
    seeds, _, steps_path, folder = protocol(group)
    manifest = json.loads((OUT/f"{group}_seed_manifest.json").read_text())
    if manifest["seed_protocol_sha256"] != sha(OUT/f"{group}_seed_protocol.json"):
        raise RuntimeError("Seed protocol changed after behavior collection")
    if group == "confirmation" and manifest["G6A_pre_confirmation_freeze_sha256"] != sha(OUT/"pre_confirmation_freeze.json"):
        raise RuntimeError("G6A freeze changed after confirmation opened")
    rows_by_seed = {}
    with steps_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]),[]).append(row)
    if sorted(rows_by_seed) != seeds:
        raise RuntimeError("G6A behavior set incomplete")
    chosen = choose_states(seeds,rows_by_seed,"confirmation")
    selected = {(state["seed"],state["decision_index"]):state for state in chosen}
    if len(selected) != 120:
        raise RuntimeError("Expected four selected states per episode")
    folder.mkdir(parents=True,exist_ok=False)
    env,agent,monitor = setup(seeds)
    differences, steps_compared = {},0
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation,_ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor,"_g6_added"):
                env._simulation.root.addObject(monitor)
                monitor._g6_added=True
            rows=rows_by_seed[seed]
            terminated=truncated=False
            d=0
            while not (terminated or truncated):
                if d>=len(rows):
                    raise RuntimeError(f"G6A {group} replay overran episode {seed}")
                row=rows[d]
                if not np.array_equal(observation,np.asarray(json.loads(row["decision_observation_json"]),dtype=np.float32)):
                    raise RuntimeError(f"G6A {group} observation replay mismatch {seed}/{d}")
                state=selected.get((seed,d))
                if state is not None:
                    if d<1:
                        raise RuntimeError("Physical history undefined at d=0")
                    physical,audit=extract(env)
                    delta=abs(audit["current_distal_force_final_N"]-float(rows[d-1]["distal_contact_force_final_N"]))
                    differences["preaction_previous_final_force_N"]=max(differences.get("preaction_previous_final_force_N",0.),delta)
                    if delta>1e-8:
                        raise RuntimeError(f"G6A {group} pre-action force alignment mismatch {seed}/{d}")
                    with torch.inference_mode():
                        latent=agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                action=agent.act(observation,first_step=d==0,eval_mode=True)
                if not np.array_equal(action,np.asarray(json.loads(row["selected_action_json"]),dtype=np.float32)):
                    raise RuntimeError(f"G6A {group} action replay mismatch {seed}/{d}")
                forked_behavior=None
                if state is not None:
                    outcomes=[]
                    for candidate in generate_candidates(action,state["state_id"],"confirmation"):
                        result=fork_step(env,monitor,np.asarray(candidate["action"],dtype=np.float32))
                        if candidate["candidate_id"]==0:
                            forked_behavior=result
                            compare(result,g2_signature(row),differences,f"G6A {group} behavior fork {seed}/{d}")
                        outcomes.append({**candidate,"force_next_max_N":result["force_next_max_N"],"outcome":result})
                    new_json(folder/f"state_{state['state_id']:04d}.json",{**state,"observation":observation.tolist(),
                        "latent":latent.tolist(),"physical":physical.tolist(),"behavior_action":action.tolist(),"candidates":outcomes})
                    print(f"G6A {group} candidate state {len(list(folder.glob('state_*.json')))}/120 seed={seed} d={d}",flush=True)
                parent=signature(env,monitor,action)
                compare(parent,g2_signature(row),differences,f"G6A {group} parent {seed}/{d}")
                if forked_behavior is not None:
                    compare(parent,forked_behavior,differences,f"G6A {group} parent-fork {seed}/{d}")
                observation=np.asarray(parent["observation"],dtype=np.float32)
                terminated,truncated=parent["terminated"],parent["truncated"]
                d+=1
                steps_compared+=1
            if d!=len(rows):
                raise RuntimeError(f"G6A {group} episode length mismatch {seed}")
    finally:
        env.close()
    if len(list(folder.glob("state_*.json")))!=120:
        raise RuntimeError("G6A candidate set incomplete")
    new_json(OUT/f"{group}_replay_audit.json",{"status":"PASS","parent_steps_compared":steps_compared,
        "candidate_states":120,"candidate_evaluations":2520,"max_abs_difference":differences})


def score(group):
    seeds,_,_,folder=protocol(group)
    audit=json.loads((OUT/f"{group}_replay_audit.json").read_text())
    if audit["status"]!="PASS" or audit["candidate_evaluations"]!=2520:
        raise RuntimeError("Replay integrity gate failed")
    paths=sorted(folder.glob("state_*.json"))
    if len(paths)!=120:
        raise RuntimeError("Candidate labels incomplete")
    states=[json.loads(path.read_text()) for path in paths]
    if [s["state_id"] for s in states]!=list(range(120)) or sorted(set(s["seed"] for s in states))!=seeds:
        raise RuntimeError("Candidate state ids/seeds changed")
    x=g5_confirm.confirmation_x(MODEL,states,("K","D"))
    y=np.asarray([c["force_next_max_N"] for s in states for c in s["candidates"]],dtype=np.float64)
    if x.shape!=(2520,291) or y.shape!=(2520,) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("G6A scoring inputs/labels invalid")
    q=verify_source()
    predictions={seed:predict(load_predictor(MODEL,seed,torch.device("cuda:0")),x,torch.device("cuda:0")) for seed in SEEDS}
    rows=[]
    for i,state in enumerate(states):
        physical=np.asarray(state["physical"],dtype=float)
        for j,candidate in enumerate(state["candidates"]):
            k=i*21+j
            rows.append({"episode_seed":state["seed"],"decision_index":state["decision_index"],"state_id":state["state_id"],
                "candidate_id":j,"candidate_kind":candidate["kind"],"current_contact":int(physical[34]>0),
                "current_distal_force_N":float(physical[33]),"action_translation":float(candidate["action"][0]),
                "action_rotation":float(candidate["action"][1]),"predictor_input_json":json.dumps(x[k].tolist(),separators=(",",":")),
                "F_true_N":float(y[k]),"true_Q95_high_force":int(y[k]>q["q95"]),
                **{f"prediction_seed_{seed}_N":float(predictions[seed][k]) for seed in SEEDS}})
    path=OUT/f"{group}_candidate_results.csv"
    if path.exists():
        raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    new_json(OUT/f"{group}_scoring_audit.json",{"status":"PASS_THREE_FROZEN_REPLICA_PREDICTIONS",
        "seeds":seeds,"states":120,"candidate_rows":2520,"model_training_seeds":list(SEEDS),
        "checkpoint_hashes":{str(seed):sha(G5/"models"/MODEL/f"seed_{seed}"/"model.pt") for seed in SEEDS},
        "input_dimension":291,"Q95_true_high_force_count":int(np.sum(y>q["q95"])),
        "candidate_results_sha256":sha(path)})
    print(f"G6A {group} scored 2520 shared SOFA labels with three frozen independent replicas",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--group",choices=("calibration","confirmation"),required=True)
    parser.add_argument("--stage",choices=("behavior","candidates","score"),required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Frozen CUDA_VISIBLE_DEVICES=0 required")
    {"behavior":behavior,"candidates":candidates,"score":score}[args.stage](args.group)


if __name__=="__main__":
    main()
