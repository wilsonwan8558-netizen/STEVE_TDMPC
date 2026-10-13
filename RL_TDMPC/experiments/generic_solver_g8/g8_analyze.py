"""Offline one-step repair/oracle analysis; no repaired action is executed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from g8_prepare import OUT,TEST_SEEDS,MODEL_SEEDS,verify,sha,new_json
from g8_repair import validate_candidates,select,oracle


def csv_new(path,rows):
    if path.exists():raise FileExistsError(path)
    if not rows:raise RuntimeError(f"Empty output {path}")
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(values):
    a=np.asarray([v for v in values if v is not None],dtype=np.float64)
    if not len(a):return {"count":0,"mean":None,"median":None,"q90":None,"max":None,"min":None}
    return {"count":len(a),"mean":float(np.mean(a)),"median":float(np.median(a)),
        "q90":float(np.quantile(a,.9)),"max":float(np.max(a)),"min":float(np.min(a))}


def result_row(method,seed,state_id,episode_seed,d,current_contact,actions,truth,scores,cutoff,limit,true_set,selected,case,pred_set):
    nominal=float(truth[0]);nominal_unsafe=nominal>limit
    repairable=nominal_unsafe and bool(true_set)
    chosen_force=float(truth[selected]) if selected is not None else None
    selected_unsafe=chosen_force>limit if chosen_force is not None else None
    intervention=selected is not None and selected!=0
    distance=float(np.linalg.norm(actions[selected]-actions[0])) if selected is not None else None
    delta_t=float(abs(actions[selected,0]-actions[0,0])) if selected is not None else None
    delta_r=float(abs(actions[selected,1]-actions[0,1])) if selected is not None else None
    intersection=len(set(pred_set)&set(true_set)) if pred_set is not None else None
    if not true_set:
        cause="CANDIDATE_SET_INFEASIBLE"
    elif selected is None:
        cause="PREDICTED_INFEASIBLE"
    elif selected_unsafe and nominal_unsafe and selected==0:
        cause="UNSAFE_PASS_THROUGH"
    elif selected_unsafe and nominal_unsafe and intervention:
        cause="WRONG_REPAIR"
    elif selected_unsafe:
        cause="OTHER_AMBIGUOUS"
    else:
        cause="NONE"
    return {"method":method,"training_seed":seed,"state_id":state_id,"episode_seed":episode_seed,
        "decision_index":d,"current_contact":int(current_contact),
        "nominal_candidate_id":0,"nominal_true_force_N":nominal,"nominal_true_unsafe":int(nominal_unsafe),
        "nominal_Q95_or_point_score_N":float(scores[0]) if scores is not None else None,
        "nominal_predicted_safe":int(scores[0]<=cutoff) if scores is not None else None,
        "repairable_unsafe_nominal":int(repairable),"candidate_set_true_infeasible":int(not true_set),
        "true_safe_set_size":len(true_set),"predicted_safe_set_size":len(pred_set) if pred_set is not None else None,
        "safe_set_intersection_size":intersection,
        "selected_candidate_id":selected,"selection_case":case,"selected_true_force_N":chosen_force,
        "selected_true_unsafe":int(selected_unsafe) if selected_unsafe is not None else None,
        "intervention":int(intervention),"missed_unsafe_nominal":int(nominal_unsafe and selected==0 and scores is not None and scores[0]<=cutoff),
        "false_intervention":int(not nominal_unsafe and intervention),
        "successful_repair":int(repairable and intervention and selected_unsafe is False),
        "failed_repair":int(nominal_unsafe and intervention and selected_unsafe is True),
        "predicted_infeasible":int(selected is None) if scores is not None else None,
        "conservative_infeasible":int(selected is None and bool(true_set)) if scores is not None else None,
        "action_distance_l2":distance,"abs_delta_translation":delta_t,"abs_delta_rotation":delta_r,
        "delta_true_force_N":chosen_force-nominal if chosen_force is not None else None,
        "nominal_Q95_excess_N":max(0.,nominal-limit),
        "selected_Q95_excess_N":max(0.,chosen_force-limit) if chosen_force is not None else None,
        "failure_cause":cause}


def metric_summary(rows,nominal_counts,limit,q99):
    selected=[r for r in rows if r["selected_candidate_id"] is not None]
    intervened=[r for r in rows if r["intervention"]]
    success=[r for r in rows if r["successful_repair"]]
    unsafe_nom=[r for r in rows if r["nominal_true_unsafe"]]
    safe_nom=[r for r in rows if not r["nominal_true_unsafe"]]
    selected_unsafe=[r for r in selected if r["selected_true_unsafe"]]
    pred_infeasible=[r for r in rows if r["predicted_infeasible"]]
    conservative=[r for r in rows if r["conservative_infeasible"]]
    repairable=nominal_counts["repairable_unsafe_nominal_states"]
    q99_nominal=[r for r in rows if r["nominal_true_force_N"]>q99]
    q99_selected=[r for r in q99_nominal if r["selected_candidate_id"] is not None]
    return {"state_count":len(rows),"selected_action_count":len(selected),
        "nominal_true_unsafe_count":len(unsafe_nom),"repairable_unsafe_nominal_count":repairable,
        "true_candidate_set_infeasible_count":sum(r["candidate_set_true_infeasible"] for r in rows),
        "missed_unsafe_nominal_count":sum(r["missed_unsafe_nominal"] for r in rows),
        "intervention_count":len(intervened),"intervention_rate":len(intervened)/len(rows),
        "false_intervention_count":sum(r["false_intervention"] for r in rows),
        "false_intervention_rate_given_nominal_true_safe":sum(r["false_intervention"] for r in rows)/len(safe_nom) if safe_nom else None,
        "successful_repair_count":len(success),"rescue_rate_given_repairable_unsafe":len(success)/repairable if repairable else None,
        "failed_repair_count":sum(r["failed_repair"] for r in rows),
        "final_selected_true_exceedance_count":len(selected_unsafe),
        "final_selected_true_exceedance_rate":len(selected_unsafe)/len(selected) if selected else None,
        "predicted_infeasible_count":len(pred_infeasible),
        "predicted_infeasible_rate":len(pred_infeasible)/len(rows),
        "conservative_infeasible_count":len(conservative),
        "fraction_predicted_infeasible_with_true_safe_alternative":len(conservative)/len(pred_infeasible) if pred_infeasible else None,
        "successful_repair_action_distance_l2":stats([r["action_distance_l2"] for r in success]),
        "successful_repair_abs_delta_translation":stats([r["abs_delta_translation"] for r in success]),
        "successful_repair_abs_delta_rotation":stats([r["abs_delta_rotation"] for r in success]),
        "all_intervention_action_distance_l2":stats([r["action_distance_l2"] for r in intervened]),
        "intervened_force_delta_N":stats([r["delta_true_force_N"] for r in intervened]),
        "fraction_interventions_reducing_true_force":sum(r["delta_true_force_N"]<0 for r in intervened)/len(intervened) if intervened else None,
        "fraction_interventions_increasing_true_force":sum(r["delta_true_force_N"]>0 for r in intervened)/len(intervened) if intervened else None,
        "largest_force_reduction_N":min((r["delta_true_force_N"] for r in intervened),default=None),
        "largest_force_increase_N":max((r["delta_true_force_N"] for r in intervened),default=None),
        "nominal_unsafe_selected_below_F_lim_count":sum(r["nominal_true_unsafe"] and r["selected_candidate_id"] is not None and not r["selected_true_unsafe"] for r in rows),
        "nominal_Q95_excess_N":stats([r["nominal_Q95_excess_N"] for r in unsafe_nom]),
        "selected_Q95_excess_among_nominal_unsafe_N":stats([r["selected_Q95_excess_N"] for r in unsafe_nom if r["selected_candidate_id"] is not None]),
        "nominal_above_G2_Q99_count":len(q99_nominal),
        "nominal_above_Q99_with_selected_count":len(q99_selected),
        "nominal_above_Q99_selected_still_above_Q99_count":sum(r["selected_true_force_N"]>q99 for r in q99_selected),
        "nominal_above_Q99_selected_below_F_lim_count":sum(r["selected_true_force_N"]<=limit for r in q99_selected),
        "nominal_above_Q99_force_change_N":stats([r["delta_true_force_N"] for r in q99_selected]),
        "failure_taxonomy_counts":{name:sum(r["failure_cause"]==name for r in rows) for name in
            ("CANDIDATE_SET_INFEASIBLE","PREDICTED_INFEASIBLE","UNSAFE_PASS_THROUGH","WRONG_REPAIR","OTHER_AMBIGUOUS","NONE")}}


def main():
    freeze=verify()
    audit=json.loads((OUT/"candidate_scoring_audit.json").read_text())
    if audit["status"]!="PASS_G8_SAME_2520_LABELS_G7_G6B_ORACLE" or audit["candidate_results_sha256"]!=sha(OUT/"candidate_results.csv"):
        raise RuntimeError("G8 candidate score/audit incomplete")
    with (OUT/"candidate_results.csv").open(newline="") as stream:raw=list(csv.DictReader(stream))
    if len(raw)!=2520:raise RuntimeError("G8 candidate row count changed")
    limit=float(freeze["F_lim_N"]);q99=float(freeze["G2_Q99_nominal_N"])
    by_state=[]
    for i in range(120):
        group=raw[i*21:(i+1)*21]
        if [int(r["state_id"]) for r in group]!=[i]*21 or [int(r["candidate_id"]) for r in group]!=list(range(21)):
            raise RuntimeError("G8 candidate row ordering changed")
        candidates=[{"candidate_id":int(r["candidate_id"]),"kind":r["candidate_kind"],
            "action":[float(r["action_translation"]),float(r["action_rotation"])]} for r in group]
        actions=validate_candidates(candidates,candidates[0]["action"])
        truth=np.asarray([float(r["F_true_N"]) for r in group],dtype=np.float64)
        if any(int(r["true_Q95_high_force"])!=int(f>limit) or int(r["true_Q99_extreme"])!=int(f>q99) for r,f in zip(group,truth)):
            raise RuntimeError("G8 force truth classification changed")
        current=np.asarray([int(r["current_contact"]) for r in group],dtype=int)
        if len(set(current))!=1:raise RuntimeError("Current contact varies within state")
        by_state.append((group,actions,truth,bool(current[0])))
    nominal=[];oracle_rows=[];g7rows=[];g6brows=[];infeasible=[];safe_state=[];taxonomy=[]
    methods={"G7":g7rows,"G6B":g6brows}
    for state_id,(group,actions,truth,current) in enumerate(by_state):
        seed=int(group[0]["episode_seed"]);d=int(group[0]["decision_index"])
        oracle_result=oracle(truth,limit,actions)
        true_set=oracle_result["true_safe_ids"]
        nominal_unsafe=truth[0]>limit
        nominal.append({"state_id":state_id,"episode_seed":seed,"decision_index":d,"current_contact":int(current),
            "nominal_action_translation":float(actions[0,0]),"nominal_action_rotation":float(actions[0,1]),
            "nominal_true_force_N":float(truth[0]),"nominal_true_unsafe":int(nominal_unsafe),
            "nominal_excess_over_F_lim_N":float(max(0.,truth[0]-limit)),
            "nominal_above_G2_Q99":int(truth[0]>q99),"true_safe_candidate_count":len(true_set),
            "true_candidate_set_infeasible":int(not true_set),"repairable_unsafe_nominal":int(nominal_unsafe and bool(true_set)),
            "oracle_selected_candidate_id":oracle_result["selected_candidate_id"],
            "oracle_selected_true_force_N":float(truth[oracle_result["selected_candidate_id"]]) if oracle_result["selected_candidate_id"] is not None else None})
        oracle_rows.append(result_row("ORACLE",None,state_id,seed,d,current,actions,truth,None,None,limit,true_set,
            oracle_result["selected_candidate_id"],oracle_result["selection_case"],None))
        for method,target in methods.items():
            cutoff=limit if method=="G7" else float(freeze["G6B_point_cutoff_N"])
            for training_seed in MODEL_SEEDS:
                field=f"G7_Q95_seed_{training_seed}_N" if method=="G7" else f"G6B_point_seed_{training_seed}_N"
                scores=np.asarray([float(r[field]) for r in group],dtype=np.float64)
                decision=select(scores,cutoff,actions)
                chosen=decision["selected_candidate_id"]
                r=result_row(method,training_seed,state_id,seed,d,current,actions,truth,scores,cutoff,limit,true_set,
                    chosen,decision["selection_case"],decision["predicted_safe_ids"])
                target.append(r)
                infeasible.append({"method":method,"training_seed":training_seed,"state_id":state_id,
                    "predicted_safe_set_size":r["predicted_safe_set_size"],"true_safe_set_size":len(true_set),
                    "predicted_infeasible":r["predicted_infeasible"],"candidate_set_true_infeasible":r["candidate_set_true_infeasible"],
                    "conservative_infeasible":r["conservative_infeasible"],
                    "nominal_true_unsafe":r["nominal_true_unsafe"]})
                intersection=r["safe_set_intersection_size"]
                union=r["predicted_safe_set_size"]+len(true_set)-intersection
                safe_state.append({"method":method,"training_seed":training_seed,"state_id":state_id,
                    "predicted_safe_set_size":r["predicted_safe_set_size"],"true_safe_set_size":len(true_set),
                    "intersection_size":intersection,"union_size":union,
                    "Jaccard":intersection/union if union else 1.})
                if r["failure_cause"]!="NONE":
                    taxonomy.append({"method":method,"training_seed":training_seed,"state_id":state_id,
                        "episode_seed":seed,"decision_index":d,"primary_failure_cause":r["failure_cause"],
                        "nominal_true_force_N":r["nominal_true_force_N"],
                        "selected_true_force_N":r["selected_true_force_N"],
                        "nominal_true_unsafe":r["nominal_true_unsafe"],
                        "predicted_infeasible":r["predicted_infeasible"],
                        "true_safe_candidate_count":len(true_set)})
    csv_new(OUT/"nominal_state_summary.csv",nominal)
    csv_new(OUT/"oracle_repair_results.csv",oracle_rows)
    csv_new(OUT/"g7_repair_results.csv",g7rows)
    csv_new(OUT/"g6b_repair_results.csv",g6brows)
    csv_new(OUT/"infeasibility_analysis.csv",infeasible)
    csv_new(OUT/"safe_set_state_details.csv",safe_state)
    csv_new(OUT/"failure_taxonomy.csv",taxonomy)
    counts={"states":120,"nominal_true_safe_states":sum(not r["nominal_true_unsafe"] for r in nominal),
        "nominal_true_unsafe_states":sum(r["nominal_true_unsafe"] for r in nominal),
        "repairable_unsafe_nominal_states":sum(r["repairable_unsafe_nominal"] for r in nominal),
        "nominal_unsafe_no_true_safe_alternative":sum(r["nominal_true_unsafe"] and r["true_candidate_set_infeasible"] for r in nominal),
        "true_candidate_set_infeasible_states":sum(r["true_candidate_set_infeasible"] for r in nominal),
        "nominal_above_G2_Q99_states":sum(r["nominal_above_G2_Q99"] for r in nominal)}
    new_json(OUT/"nominal_characterization.json",counts)
    all_rows={"G7":g7rows,"G6B":g6brows,"ORACLE":oracle_rows}
    summaries={method:{str(seed):metric_summary([r for r in rows if r["training_seed"]==seed],counts,limit,q99)
        for seed in MODEL_SEEDS} if method!="ORACLE" else metric_summary(rows,counts,limit,q99)
        for method,rows in all_rows.items()}
    new_json(OUT/"intervention_metrics.json",{m:{s:{k:v for k,v in r.items() if k in
        ("intervention_count","intervention_rate","false_intervention_count","false_intervention_rate_given_nominal_true_safe",
         "missed_unsafe_nominal_count","successful_repair_count","rescue_rate_given_repairable_unsafe","failed_repair_count",
         "final_selected_true_exceedance_count","final_selected_true_exceedance_rate","selected_action_count",
         "successful_repair_action_distance_l2","successful_repair_abs_delta_translation","successful_repair_abs_delta_rotation",
         "all_intervention_action_distance_l2")}
        for s,r in summaries[m].items()} if m!="ORACLE" else summaries[m] for m in summaries})
    new_json(OUT/"force_reduction_metrics.json",{m:{s:{k:v for k,v in r.items() if k in
        ("intervened_force_delta_N","fraction_interventions_reducing_true_force","fraction_interventions_increasing_true_force",
         "largest_force_reduction_N","largest_force_increase_N","nominal_unsafe_selected_below_F_lim_count",
         "nominal_Q95_excess_N","selected_Q95_excess_among_nominal_unsafe_N",
         "nominal_above_G2_Q99_count","nominal_above_Q99_with_selected_count",
         "nominal_above_Q99_selected_still_above_Q99_count","nominal_above_Q99_selected_below_F_lim_count",
         "nominal_above_Q99_force_change_N")}
        for s,r in summaries[m].items()} if m!="ORACLE" else summaries[m] for m in summaries})
    no_contact=[r for r in nominal if not r["current_contact"] and r["nominal_true_unsafe"]]
    no_contact_ids={r["state_id"] for r in no_contact}
    new_json(OUT/"no_contact_high_force_metrics.json",{"state_count":len(no_contact_ids),
        "state_ids":sorted(no_contact_ids),"by_method":{m:{str(seed):{
            "state_count":len(no_contact_ids),
            "missed_unsafe_nominal_count":sum(r["missed_unsafe_nominal"] for r in rows if r["state_id"] in no_contact_ids),
            "interventions":sum(r["intervention"] for r in rows if r["state_id"] in no_contact_ids),
            "successful_repairs":sum(r["successful_repair"] for r in rows if r["state_id"] in no_contact_ids),
            "selected_action_true_exceedances":sum(r["selected_true_unsafe"] for r in rows if r["state_id"] in no_contact_ids and r["selected_candidate_id"] is not None),
            "predicted_infeasible":sum(r["predicted_infeasible"] for r in rows if r["state_id"] in no_contact_ids),
            "candidate_set_true_infeasible":sum(r["candidate_set_true_infeasible"] for r in rows if r["state_id"] in no_contact_ids)}
            for seed in MODEL_SEEDS for rows in [[r for r in all_rows[m] if r["training_seed"]==seed]]}
            for m in ("G7","G6B")}})
    new_json(OUT/"safe_set_metrics.json",{m:{str(seed):{
        "candidate_true_safe":sum(r["true_safe_set_size"] for r in details),
        "candidate_predicted_safe":sum(r["predicted_safe_set_size"] for r in details),
        "candidate_intersection":sum(r["intersection_size"] for r in details),
        "safe_action_precision":sum(r["intersection_size"] for r in details)/sum(r["predicted_safe_set_size"] for r in details) if sum(r["predicted_safe_set_size"] for r in details) else None,
        "safe_action_recall":sum(r["intersection_size"] for r in details)/sum(r["true_safe_set_size"] for r in details) if sum(r["true_safe_set_size"] for r in details) else None,
        "pooled_Jaccard":sum(r["intersection_size"] for r in details)/sum(r["union_size"] for r in details) if sum(r["union_size"] for r in details) else None,
        "mean_state_Jaccard":float(np.mean([r["Jaccard"] for r in details])),
        "predicted_safe_set_size":stats([r["predicted_safe_set_size"] for r in details]),
        "true_safe_set_size":stats([r["true_safe_set_size"] for r in details])}
        for seed in MODEL_SEEDS for details in [[r for r in safe_state if r["method"]==m and r["training_seed"]==seed]]}
        for m in ("G7","G6B")})
    robustness={}
    robust_fields={"rescue_rate_given_repairable_unsafe":False,
        "final_selected_true_exceedance_rate":True,
        "false_intervention_rate_given_nominal_true_safe":True,
        "predicted_infeasible_rate":True}
    for field,higher_bad in robust_fields.items():
        values=np.asarray([summaries["G7"][str(seed)][field] for seed in MODEL_SEEDS],dtype=np.float64)
        robustness[field]={"by_seed":{str(seed):float(v) for seed,v in zip(MODEL_SEEDS,values)},
            "mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),
            "worst":float(np.max(values) if higher_bad else np.min(values))}
    deviations=np.asarray([summaries["G7"][str(seed)]["all_intervention_action_distance_l2"]["mean"] for seed in MODEL_SEEDS],dtype=np.float64)
    robustness["mean_action_deviation_l2_all_interventions"]={"by_seed":{str(seed):float(v) for seed,v in zip(MODEL_SEEDS,deviations)},
        "mean":float(np.mean(deviations)),"sample_sd":float(np.std(deviations,ddof=1)),"worst":float(np.max(deviations))}
    chosen_by_seed={seed:{r["state_id"]:r["selected_candidate_id"] for r in g7rows if r["training_seed"]==seed} for seed in MODEL_SEEDS}
    triple=[tuple(chosen_by_seed[seed][i] for seed in MODEL_SEEDS) for i in range(120)]
    all_same=sum(v[0]==v[1]==v[2] for v in triple)
    same_action=sum(v[0] is not None and v[0]==v[1]==v[2] for v in triple)
    all_different=sum(None not in v and len(set(v))==3 for v in triple)
    any_infeasible=sum(None in v for v in triple)
    new_json(OUT/"training_seed_robustness.json",{"G7_three_seed_metrics":robustness,
        "same_selection_including_all_infeasible_count":all_same,
        "same_selection_including_all_infeasible_fraction":all_same/120,
        "all_three_choose_same_action_count":same_action,"all_three_choose_same_action_fraction":same_action/120,
        "not_all_three_choose_same_action_count":120-same_action,
        "all_three_choose_different_actions_count":all_different,"all_three_choose_different_actions_fraction":all_different/120,
        "any_replica_predicted_infeasible_count":any_infeasible,"any_replica_predicted_infeasible_fraction":any_infeasible/120})
    examples={method:{str(seed):{cause:[r["state_id"] for r in rows if r["failure_cause"]==cause][:12]
        for cause in ("CANDIDATE_SET_INFEASIBLE","PREDICTED_INFEASIBLE","UNSAFE_PASS_THROUGH","WRONG_REPAIR","OTHER_AMBIGUOUS")}
        for seed in MODEL_SEEDS for rows in [[r for r in all_rows[method] if r["training_seed"]==seed]]}
        for method in ("G7","G6B")}
    new_json(OUT/"G7_vs_G6B_vs_ORACLE.json",{"status":"COMPLETE_OFFLINE_ONE_STEP_ONLY",
        "F_lim_N":limit,"G2_Q99_nominal_N":q99,"state_count":120,"candidate_count":2520,
        "nominal":counts,"by_method":summaries,"G7_failure_examples":examples["G7"],
        "G6B_failure_examples":examples["G6B"],
        "no_repaired_action_executed_in_closed_loop":True})
    print("G8 nominal",counts,flush=True)
    print("G8 G7",[(seed,summaries["G7"][str(seed)]["rescue_rate_given_repairable_unsafe"],
        summaries["G7"][str(seed)]["final_selected_true_exceedance_count"],
        summaries["G7"][str(seed)]["predicted_infeasible_count"]) for seed in MODEL_SEEDS],flush=True)


if __name__=="__main__":main()
