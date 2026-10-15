"""G12.3 final-iteration binary force mask on frozen MPPI elite weights."""

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

G121 = RL / "logs/world_model_force_aware_planning_g121"
G122 = RL / "logs/world_model_force_aware_planning_g122"
OUT = RL / "logs/world_model_force_aware_planning_g123"


class MaskedMPPIAgent(TDMPC2Agent):
    """Call the original _plan unchanged; mask only its final elite weights.

    The original _plan performs all six CEM/MPPI updates and writes its original
    final mean to previous_mean. For an active mask, we replace previous_mean
    with the masked final mean, preserving the normal warm-start semantics.
    """

    def __init__(self, *args, force_head: ForceHead, delta_F: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.force_head = force_head.eval().requires_grad_(False)
        self.delta_F = float(delta_F)
        self.mask_enabled = True
        self.trace_enabled = False
        self.last_gate = None
        self.last_scored = None
        self._last_values = None
        self._last_actions = None
        self._last_latent = None

    @torch.no_grad()
    def _estimate_value(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        value = super()._estimate_value(latent, actions)
        if self.mask_enabled or self.trace_enabled:
            # Called after the original scorer's RNG draws. Saving the final
            # call's tensors has no randomness or effect on candidate scoring.
            self._last_values = value.detach().clone()
            self._last_actions = actions.detach().clone()
            self._last_latent = latent.detach()
        return value

    @torch.no_grad()
    def _plan(self, observation: torch.Tensor, *, first_step: bool,
              eval_mode: bool) -> torch.Tensor:
        if not self.mask_enabled and not self.trace_enabled:
            self.last_gate = None
            return super()._plan(observation, first_step=first_step, eval_mode=eval_mode)
        if not eval_mode:
            raise RuntimeError("G12.3 frozen masked MPPI is evaluation-only")
        self._last_values = self._last_actions = self._last_latent = None
        baseline_action = super()._plan(observation, first_step=first_step,
                                        eval_mode=eval_mode)
        if any(x is None for x in (self._last_values, self._last_actions,
                                   self._last_latent)):
            raise RuntimeError("Final original MPPI candidates were not captured")
        actions = self._last_actions
        values = torch.nan_to_num(self._last_values, nan=0.0)
        indices = torch.topk(values.squeeze(-1), int(self.config["num_elites"])).indices
        elite_values = values[indices]
        elite_actions = actions[:, indices]
        max_value = elite_values.max(dim=0).values
        u = torch.exp(float(self.config["temperature"]) * (elite_values - max_value))
        w = u / (u.sum(dim=0, keepdim=True) + 1e-9)
        original_mean = (w.unsqueeze(0) * elite_actions).sum(dim=1) / (
            w.sum(dim=0) + 1e-9)
        if not torch.allclose(original_mean[0].clamp(-1, 1), baseline_action,
                              atol=1e-6, rtol=0):
            raise RuntimeError("G12.3 baseline MPPI weighting reconstruction changed")
        if not torch.allclose(self.previous_mean, original_mean, atol=1e-6, rtol=0):
            raise RuntimeError("G12.3 baseline MPPI warm-start mean changed")
        raw = self.force_head(self._last_latent, actions[0]).squeeze(-1)
        calibrated = raw + self.delta_F
        safe = calibrated <= F_LIM
        elite_safe = safe[indices]
        retained = (w.squeeze(-1) * elite_safe.to(w.dtype)).sum()
        baseline_weights = w.squeeze(-1) / w.sum()
        baseline_ess = 1 / baseline_weights.square().sum()
        if bool(elite_safe.all()):
            # This branch also handles the important all-512-safe invariant.
            # It avoids epsilon-induced drift when the mask removes no mass.
            action = baseline_action
            gated_ess = baseline_ess
        elif float(retained) > 0:
            gated_weights = w * elite_safe.unsqueeze(-1).to(w.dtype)
            gated_weights = gated_weights / retained
            masked_mean = (gated_weights.unsqueeze(0) * elite_actions).sum(dim=1) / (
                gated_weights.sum(dim=0) + 1e-9)
            if not torch.isfinite(masked_mean).all():
                raise FloatingPointError("G12.3 masked MPPI mean is nonfinite")
            self.previous_mean.copy_(masked_mean)
            action = masked_mean[0].clamp(-1, 1)
            gated_ess = 1 / gated_weights.squeeze(-1).square().sum()
        else:
            # Frozen G12.3 F0: no weighted candidate survives; retain exactly
            # the original MPPI action and original previous_mean.
            action = baseline_action
            gated_ess = baseline_ess
        if not torch.isfinite(action).all():
            raise FloatingPointError("G12.3 masked MPPI action is nonfinite")
        if bool(safe.all()):
            gate_class = "FORCE-INACTIVE"
        elif float(retained) == 0:
            gate_class = "GATE-EMPTY"
        elif float(retained) >= 0.8:
            gate_class = "LIGHT-GATE"
        elif float(retained) >= 0.2:
            gate_class = "MODERATE-GATE"
        else:
            gate_class = "SEVERE-GATE"
        baseline_pred = self.force_head(self._last_latent[:1],
                                        baseline_action.unsqueeze(0)).item() + self.delta_F
        gated_pred = self.force_head(self._last_latent[:1],
                                     action.unsqueeze(0)).item() + self.delta_F
        self.last_gate = {
            "gate_class": gate_class,
            "safe_candidate_count": int(safe.sum().item()),
            "total_candidate_count": int(safe.numel()),
            "safe_candidate_fraction": float(safe.float().mean().item()),
            "retained_weight_mass": float(retained.item()),
            "baseline_ESS": float(baseline_ess.item()),
            "gated_ESS": float(gated_ess.item()),
            "baseline_action": baseline_action.detach().cpu().numpy().tolist(),
            "gated_action": action.detach().cpu().numpy().tolist(),
            "baseline_predicted_calibrated_force_N": baseline_pred,
            "gated_predicted_calibrated_force_N": gated_pred,
            "masked_elite_count": int((~elite_safe).sum().item()),
            "masked_weighting_applied": bool(float(retained) > 0 and not bool(elite_safe.all())),
            "baseline_fallback": bool(float(retained) == 0),
        }
        if self.trace_enabled:
            self.last_scored = {
                "actions": actions.detach().clone(),
                "J_task": values.detach().clone(),
                "F_hat_cal_N": calibrated.detach().clone(),
                "elite_indices": indices.detach().clone(),
                "elite_weights": w.detach().clone(),
                "safe": safe.detach().clone(),
            }
        return action


def load_agent() -> MaskedMPPIAgent:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12.3 requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    frozen = json.loads((OUT / "inherited_g122_audit.json").read_text())
    selection = json.loads((G122 / "selected_gate_calibration.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G12 / "force_head.pt") != frozen["force_head_checkpoint_sha256"] or
            selection["selected_variant"] != "G1" or
            selection["delta_N"] != frozen["G1_delta_N"] or
            selection["F_gate_N"] != F_LIM or selection["H_force"] != 1):
        raise RuntimeError("G12.3 frozen model, head, or gate identity changed")
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    head_checkpoint = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    state = head_checkpoint["state_dict"]
    head = ForceHead(state["input_mean"], state["input_std"]).to("cuda:0")
    head.load_state_dict(state)
    agent = MaskedMPPIAgent(14, 2, build_agent_config(config),
                            episode_length=int(config["environment"]["max_episode_steps"]),
                            device=torch.device("cuda:0"), force_head=head,
                            delta_F=float(selection["delta_N"]))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    agent.model.eval().requires_grad_(False)
    return agent
