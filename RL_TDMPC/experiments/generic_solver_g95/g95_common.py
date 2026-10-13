"""Frozen G9.5 planner capture, E1 representation, and true two-step SOFA fork."""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g9", "generic_solver_g6a", "generic_solver_g5",
               "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from g9_estimate import Mapper  # noqa: E402
from g9_risk import QuantileNet, F_LIM  # noqa: E402
from restore_audit import signature, compare  # noqa: E402
from prepare import EXPECTED_CHECKPOINT_SHA, check_force_source  # noqa: E402

G2 = RL / "logs/generic_solver_phase_g2_force_baseline"
G3 = RL / "logs/generic_solver_phase_g3_force_predictor"
G9 = RL / "logs/generic_solver_phase_g9_temporal_anticipatory_safety"
OUT = RL / "logs/generic_solver_phase_g95_planner_conditioned_risk"
CONFIRM = tuple(range(103000, 103030))
SEEDS = (260312, 260313, 260314)
H = 2
TAUS = (.50, .90, .95, .99)
SAMPLED_PER_STATE = 8
PLANS_PER_STATE = 9  # Eight final CEM candidates plus its optimized mean.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cuda_gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or
            not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G9.5 requires CUDA_VISIBLE_DEVICES=0 on the frozen RTX 5090; no CPU fallback")


def source_gate():
    cuda_gate()
    check_force_source()
    g9 = json.loads((G9 / "model_manifest.json").read_text())
    key = "E1_H2"
    model = G9 / "models/risk/E1/H2/model.pt"
    norm = G9 / "models/risk/E1/H2/normalization.json"
    if (sha(model) != g9["checkpoint_sha256"][key] or
            sha(norm) != g9["normalization_sha256"][key] or
            sha(RL / "tdmpc2/agent.py") != json.loads((G9 / "action_sequence_protocol.json").read_text())["controller_source_sha256"]):
        raise RuntimeError("Frozen G9 E1/H2 model or TD-MPC2 planner changed")
    freeze = json.loads((G9 / "pre_confirmation_freeze.json").read_text())
    if (freeze["controller_checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA or
            abs(float(freeze["F_lim_N"])-F_LIM) > 0):
        raise RuntimeError("Frozen G9 checkpoint or force reference changed")
    mapper = G9 / "models/E1/model.pt"
    mapper_norm = G9 / "models/E1/normalization.json"
    if not mapper.is_file() or not mapper_norm.is_file():
        raise RuntimeError("G9 E1 mapper unavailable")
    return {"g9_model_sha256": sha(model), "g9_risk_normalization_sha256": sha(norm),
            "g9_E1_mapper_sha256": sha(mapper), "g9_E1_mapper_normalization_sha256": sha(mapper_norm),
            "planner_source_sha256": sha(RL / "tdmpc2/agent.py"),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "g2_behavior_sha256": sha(G2 / "step_level_force.csv"),
            "g3_state_selection_sha256": sha(G3 / "development_state_selection.json"),
            "g3_split_sha256": sha(G3 / "development_split.json"),
            "g9_behavior_temporal_sha256": sha(G9 / "development_temporal_behavior.npz")}


class E1Representation:
    """Exactly the G9 E1 mapper, with the frozen G9 train-only normalization."""

    def __init__(self):
        norm = json.loads((G9 / "models/E1/normalization.json").read_text())
        self.mean = np.asarray(norm["input_mean"], dtype=np.float64)
        self.scale = np.asarray(norm["input_std"], dtype=np.float64)
        self.target_mean = np.asarray(norm["target_mean"], dtype=np.float64)
        self.target_std = np.asarray(norm["target_std"], dtype=np.float64)
        if (self.mean.shape != (46,) or self.scale.shape != (46,) or
                self.target_mean.shape != (38,) or self.target_std.shape != (38,)):
            raise RuntimeError("G9 E1 normalization dimension changed")
        self.net = Mapper(46).to("cuda:0")
        self.net.load_state_dict(torch.load(G9 / "models/E1/model.pt", map_location="cuda:0", weights_only=True))
        self.net.eval()

    def estimate(self, observations, previous_actions):
        obs = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(previous_actions, dtype=np.float32)
        if obs.shape != (3, 14) or actions.shape != (2, 2):
            raise RuntimeError("G9.5 E1 requires obs[d-2:d] and actions[d-2:d-1]")
        raw = np.r_[obs.reshape(-1), actions.reshape(-1)]
        normalized = (raw.astype(np.float64)-self.mean)/self.scale
        with torch.inference_mode():
            p = self.net(torch.as_tensor(normalized[None, :], dtype=torch.float32, device="cuda:0"))[0].cpu().numpy()
        result = (p*self.target_std+self.target_mean).astype(np.float32)
        if result.shape != (38,) or not np.all(np.isfinite(result)):
            raise RuntimeError("G9.5 E1 estimate invalid")
        return result


def captured_plan(agent, observation, first_step):
    """Passive `_estimate_value` tap: return exact values and do not alter RNG."""
    original = agent._estimate_value
    calls = []

    def tapped(latent, actions):
        value = original(latent, actions)
        calls.append((actions.detach().cpu().numpy().copy(), value.detach().cpu().numpy().copy()))
        return value

    agent._estimate_value = tapped
    try:
        selected = agent.act(observation, first_step=first_step, eval_mode=True)
    finally:
        agent._estimate_value = original
    if len(calls) != 6:
        raise RuntimeError(f"TD-MPC2 final CEM capture expected 6 value calls, got {len(calls)}")
    actions, values = calls[-1]
    mean = agent.previous_mean.detach().cpu().numpy().copy()
    if (actions.shape != (3, 512, 2) or values.shape != (512, 1) or
            mean.shape != (3, 2) or not np.array_equal(np.clip(mean[0], -1, 1), selected) or
            np.any(~np.isfinite(actions)) or np.any(np.abs(actions) > 1+1e-6)):
        raise RuntimeError("Frozen TD-MPC2 candidate shape/range/action invariant failed")
    scores = np.nan_to_num(values[:, 0]).astype(np.float64)
    return selected, actions.astype(np.float32), scores, mean.astype(np.float32)


def choose_plans(final_actions, scores, optimized_mean):
    """Label-free deterministic score-stratified and first-two-action diverse set."""
    first2 = final_actions[:2].transpose(1, 0, 2)
    order = np.lexsort((np.arange(len(scores)), -scores))  # descending score; index tie-break
    selected = []
    seen = set()

    def add(index, kind):
        pair = np.asarray(first2[index], dtype=np.float32)
        key = pair.tobytes()
        if key in seen:
            return False
        seen.add(key)
        selected.append({"source": kind, "population_index": int(index),
                         "planner_score": float(scores[index]), "actions": pair.tolist(),
                         "population_score_rank": int(np.flatnonzero(order == index)[0])})
        return True

    def take_nearest_rank(target, kind, count):
        ranks = sorted(range(len(order)), key=lambda r: (abs(r-target), r))
        added = 0
        for rank in ranks:
            if add(int(order[rank]), kind):
                added += 1
                if added == count:
                    break
        if added != count:
            raise RuntimeError(f"TD-MPC2 final population lacks {kind} unique plans")

    take_nearest_rank(0, "sampled_high_score", 2)
    take_nearest_rank(len(order)//2, "sampled_medium_score", 2)
    take_nearest_rank(len(order)-1, "sampled_low_score", 2)
    for _ in range(2):
        existing = np.asarray([p["actions"] for p in selected], dtype=np.float32).reshape(len(selected), 4)
        candidates = []
        for index in range(len(scores)):
            pair = np.asarray(first2[index], dtype=np.float32)
            if pair.tobytes() in seen:
                continue
            distance = float(np.min(np.linalg.norm(existing-pair.reshape(1, 4), axis=1)))
            candidates.append((-distance, index))
        if not candidates:
            raise RuntimeError("TD-MPC2 final population lacks diverse plans")
        add(min(candidates)[1], "sampled_action_diversity")
    mean_pair = np.asarray(optimized_mean[:2], dtype=np.float32)
    selected.append({"source": "optimized_mean_executed_first_action", "population_index": None,
                     "planner_score": None, "actions": mean_pair.tolist(), "population_score_rank": None})
    if len(selected) != PLANS_PER_STATE or not np.array_equal(np.asarray(selected[-1]["actions"])[0],
                                                               np.clip(optimized_mean[0], -1, 1)):
        raise RuntimeError("G9.5 planner subsample count or optimized mean mismatch")
    for candidate_id, item in enumerate(selected):
        item["candidate_id"] = candidate_id
    return selected


def fork_two(env, monitor, action_pair):
    pair = np.asarray(action_pair, dtype=np.float32)
    if pair.shape != (2, 2) or not np.all(np.isfinite(pair)) or np.any(np.abs(pair) > 1+1e-6):
        raise RuntimeError("G9.5 fork requires original normalized 2-action plan")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 1)
                os.dup2(null.fileno(), 2)
            outcomes = []
            for action in pair:
                outcome = signature(env, monitor, action)
                outcomes.append(outcome)
                if outcome["terminated"] or outcome["truncated"]:
                    break
            payload = {"status": "PASS", "outcomes": outcomes}
        except BaseException:
            payload = {"status": "ERROR", "traceback": traceback.format_exc()}
        try:
            os.write(write_fd, json.dumps(payload).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 120)
    if not ready:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(read_fd)
        raise RuntimeError("G9.5 true-SOFA two-step child timed out")
    raw = os.read(read_fd, 1_000_000)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if not raw:
        raise RuntimeError(f"G9.5 true-SOFA child empty result status={status}")
    result = json.loads(raw)
    if result["status"] != "PASS":
        raise RuntimeError(result["traceback"])
    outcomes = result["outcomes"]
    if len(outcomes) not in (1, 2):
        raise RuntimeError("G9.5 invalid two-step fork length")
    f1 = float(outcomes[0]["force_next_max_N"])
    f2 = float(outcomes[1]["force_next_max_N"]) if len(outcomes) == 2 else None
    return {"outcomes": outcomes, "F1_true_N": f1, "F2_true_N": f2,
            "R2_true_N": max(f1, f2) if f2 is not None else None,
            "true_high_risk": (max(f1, f2) > F_LIM) if f2 is not None else None,
            "complete_two_intervals": f2 is not None,
            "censor_reason": None if f2 is not None else
                ("terminal_success_after_one" if outcomes[0]["success"] else "terminal_or_truncated_after_one")}


def compare_two(a, b, differences, location):
    if len(a["outcomes"]) != len(b["outcomes"]):
        raise RuntimeError(f"{location}: two-step fork length mismatch")
    for j, (left, right) in enumerate(zip(a["outcomes"], b["outcomes"])):
        compare(left, right, differences, f"{location}/step{j}")

