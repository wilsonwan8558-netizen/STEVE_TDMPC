"""One-step soft force cost inside the unchanged TD-MPC2 MPPI update."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402
from g12_force_head import ForceHead  # noqa: E402
from g12_train import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, OUT as G12, sha  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g121"


class OneStepForceAwareAgent(TDMPC2Agent):
    """Preserve the original planner, changing only its candidate scalar score."""

    def __init__(self, *args, force_head: ForceHead, delta_F: float,
                 lambda_F: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if delta_F < 0 or lambda_F < 0:
            raise ValueError("G12.1 calibration and force weight must be nonnegative")
        self.force_head = force_head.eval().requires_grad_(False)
        self.delta_F = float(delta_F)
        self.lambda_F = float(lambda_F)
        self.trace_enabled = False
        self.last_scored = None

    @torch.no_grad()
    def _estimate_value(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # The original reward, termination, terminal-Q, and random two-critic
        # draws occur first and in exactly the original order. lambda=0 returns
        # their value without calling the force head or altering RNG state.
        task = super()._estimate_value(latent, actions)
        if self.lambda_F == 0 and not self.trace_enabled:
            return task
        predicted = self.force_head(latent, actions[0]) + self.delta_F
        cost = torch.relu(predicted - F_LIM).square()
        total = task - self.lambda_F * cost
        if not torch.isfinite(total).all():
            raise FloatingPointError("G12.1 nonfinite force-aware planner score")
        if self.trace_enabled:
            self.last_scored = {
                "actions": actions.detach().clone(),
                "J_task": task.detach().clone(),
                "F_hat_cal_N": predicted.detach().clone(),
                "c_force_N2": cost.detach().clone(),
                "J_total": total.detach().clone(),
            }
        return total


def load_agent(weight: float) -> OneStepForceAwareAgent:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G12.1 planner requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    inherited = json.loads((OUT / "inherited_g12_audit.json").read_text())
    calibration = json.loads((OUT / "force_calibration.json").read_text())
    scale = json.loads((OUT / "lambda_scale_definition.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G12 / "force_head.pt") != inherited["force_head_checkpoint_sha256"] or
            calibration["force_head_checkpoint_sha256"] != inherited["force_head_checkpoint_sha256"] or
            float(weight) not in [float(x) for x in scale["fixed_lambda_set"]] or
            scale["H_force"] != 1 or scale["F_lim_N_nominal"] != F_LIM):
        raise RuntimeError("G12.1 planner frozen identity or fixed lambda set changed")
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    state = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)["state_dict"]
    head = ForceHead(state["input_mean"], state["input_std"]).to("cuda:0")
    head.load_state_dict(state)
    agent = OneStepForceAwareAgent(14, 2, build_agent_config(config),
                                  episode_length=int(config["environment"]["max_episode_steps"]),
                                  device=torch.device("cuda:0"), force_head=head,
                                  delta_F=float(calibration["delta_F_N_nominal"]),
                                  lambda_F=float(weight))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    agent.model.eval().requires_grad_(False)
    return agent
