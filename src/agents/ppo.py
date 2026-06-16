"""Minimal single-environment PPO with GAE."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..models.actor_critic import ActorCritic, ACTION_SCALE


@dataclass
class Rollout:
    """Buffer of one patient course (T = ``cfg.n_fractions`` transitions).

    Each transition is its own 1-step PPO episode (``done = True``).
    Tensors are CPU and stacked along the time axis so the PPO update can
    move the whole rollout to ``cfg.device`` once.
    """
    states:               torch.Tensor   # (T, C, G, G, G)
    fraction_progresses:  torch.Tensor   # (T,)   fraction_index / n_fractions
    raw_action_samples:   torch.Tensor   # (T, B) pre-softplus actor samples
    log_probabilities:    torch.Tensor   # (T,)
    rewards:              torch.Tensor   # (T,)
    values:               torch.Tensor   # (T,)
    dones:                torch.Tensor   # (T,)


class PPO:
    def __init__(self, cfg: Config, in_channels: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(in_channels, cfg.n_beamlets).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------ act
    @torch.no_grad()
    def act(self,
            state: np.ndarray,
            fraction_progress: float,
            deterministic: bool = False):
        """Sample one action from the current policy.

        Returns ``(action_3d, raw_action_sample, log_probability, state_value)``
        where ``action_3d`` is the softplus-mapped non-negative beamlet
        intensity vector ready to feed into ``DoseEnv.step``.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        fraction_progress_tensor = torch.tensor(
            [fraction_progress], dtype=torch.float32, device=self.device,
        )
        policy_distribution, state_value = self.net(
            state_tensor, fraction_progress_tensor,
        )
        raw_action_sample = (policy_distribution.mean
                             if deterministic
                             else policy_distribution.sample())
        log_probability = policy_distribution.log_prob(
            raw_action_sample
        ).sum(-1)
        action = (
            ActorCritic.to_action(raw_action_sample).squeeze(0).cpu().numpy()
        )
        return (action,
                raw_action_sample.squeeze(0).cpu(),
                float(log_probability.item()),
                float(state_value.item()))

    # ------------------------------------------------------------ GAE
    def _generalised_advantage_estimation(self,
                                          rewards: torch.Tensor,
                                          values: torch.Tensor,
                                          dones: torch.Tensor,
                                          last_value: float):
        """Compute GAE advantages and discounted returns.

        With every transition flagged ``done = True`` (1-step episodes) the
        ``next_non_terminal`` mask is zero everywhere, so the geometric sum
        collapses to ``advantage = reward - value`` and ``return = reward``.
        """
        n_steps = rewards.shape[0]
        advantages = torch.zeros(n_steps, dtype=torch.float32)
        last_gae_advantage = 0.0
        for step_index in reversed(range(n_steps)):
            next_value = (last_value
                          if step_index == n_steps - 1
                          else values[step_index + 1].item())
            next_non_terminal = 1.0 - dones[step_index].item()
            td_residual = (rewards[step_index].item()
                           + self.cfg.gamma * next_value * next_non_terminal
                           - values[step_index].item())
            last_gae_advantage = (
                td_residual
                + self.cfg.gamma * self.cfg.gae_lambda
                * next_non_terminal * last_gae_advantage
            )
            advantages[step_index] = last_gae_advantage
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------ update
    def update(self, rollout: Rollout, last_value: float = 0.0) -> dict:
        # --- defensive: clip raw rewards so a single huge spike can't poison
        #     advantages / value targets.  PPO is well-known to need this on
        #     environments with heavy-tailed reward.
        rewards = torch.clamp(rollout.rewards, min=-50.0, max=50.0)
        values  = torch.clamp(rollout.values,  min=-50.0, max=50.0)

        advantages, returns = self._generalised_advantage_estimation(
            rewards, values, rollout.dones, last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # If anything is non-finite, skip the update rather than poisoning
        # the network with NaN gradients.
        if not (torch.isfinite(advantages).all()
                and torch.isfinite(returns).all()):
            return {"policy_loss": float("nan"),
                    "value_loss":  float("nan"),
                    "entropy":     float("nan"),
                    "skipped":     1.0}

        states               = rollout.states.to(self.device)
        fraction_progresses  = rollout.fraction_progresses.to(self.device)
        raw_action_samples   = rollout.raw_action_samples.to(self.device)
        old_log_probabilities = rollout.log_probabilities.to(self.device)
        old_values           = values.to(self.device)
        advantages           = advantages.to(self.device)
        returns              = returns.to(self.device)

        n_transitions = states.shape[0]
        shuffled_indices = np.arange(n_transitions)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _epoch in range(self.cfg.ppo_epochs):
            np.random.shuffle(shuffled_indices)
            for batch_start in range(0, n_transitions, self.cfg.minibatch):
                minibatch_indices = shuffled_indices[
                    batch_start:batch_start + self.cfg.minibatch
                ]
                minibatch_indices_tensor = torch.as_tensor(
                    minibatch_indices, dtype=torch.long, device=self.device,
                )
                policy_distribution, state_values = self.net(
                    states[minibatch_indices_tensor],
                    fraction_progresses[minibatch_indices_tensor],
                )
                new_log_probabilities = policy_distribution.log_prob(
                    raw_action_samples[minibatch_indices_tensor]
                ).sum(-1)
                entropy = policy_distribution.entropy().sum(-1).mean()

                probability_ratio = (
                    new_log_probabilities
                    - old_log_probabilities[minibatch_indices_tensor]
                ).exp()
                minibatch_advantages = advantages[minibatch_indices_tensor]
                policy_loss = -torch.min(
                    probability_ratio * minibatch_advantages,
                    torch.clamp(
                        probability_ratio,
                        1 - self.cfg.clip_eps,
                        1 + self.cfg.clip_eps,
                    ) * minibatch_advantages,
                ).mean()
                # Clipped value loss (PPO2 style): limits how far the critic
                # can move per update, which prevents target-chasing blow-ups.
                values_clipped = old_values[minibatch_indices_tensor] + torch.clamp(
                    state_values - old_values[minibatch_indices_tensor],
                    -self.cfg.clip_eps, self.cfg.clip_eps,
                )
                value_loss = 0.5 * torch.max(
                    (state_values - returns[minibatch_indices_tensor]).pow(2),
                    (values_clipped - returns[minibatch_indices_tensor]).pow(2),
                ).mean()
                loss = (policy_loss
                        + self.cfg.vf_coef * value_loss
                        - self.cfg.ent_coef * entropy)

                self.opt.zero_grad()
                loss.backward()
                # Skip the step if any grad is non-finite (rather than letting
                # NaNs propagate through Adam moments forever).
                has_non_finite_gradient = False
                for parameter in self.net.parameters():
                    if (parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()):
                        has_non_finite_gradient = True
                        break
                if has_non_finite_gradient:
                    self.opt.zero_grad()
                    stats["skipped"] = stats.get("skipped", 0.0) + 1.0
                    continue
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += entropy.item()
        return stats

    # ------------------------------------------------------------ warm-start
    def pretrain_actor(self,
                       states_cpu: torch.Tensor,
                       fraction_progresses_cpu: torch.Tensor,
                       target_actions_cpu: torch.Tensor,
                       *,
                       epochs: int = 5,
                       minibatch: int = 4,
                       lr: float | None = None) -> dict:
        """Supervised MSE pretraining of the actor head against ``a*``.

        For each ``(state, fraction_progress)`` we minimise

            MSE( actor_mu(encoder(state) || fraction_progress),
                 inv_softplus(a* / ACTION_SCALE) )

        which pushes the *deterministic* policy output
        ``softplus(actor_mu) * ACTION_SCALE`` towards the cached NNLS
        beamlet plan. Critic and ``log_std`` are intentionally not
        touched; PPO takes over from there.

        Tensors are kept on the host and only minibatches are shipped to
        ``self.device`` so a 200-patient state stack (~2-3 GB) fits in
        normal RAM but does not need to fit in GPU memory.
        """
        if states_cpu.shape[0] == 0:
            return {"pretrain_mse_first": float("nan"),
                    "pretrain_mse_last":  float("nan"),
                    "pretrain_n":         0}

        # inverse softplus, with a floor so beamlets that the warm-start
        # zeros out map to a large-negative raw value (softplus(-8) ~ 3e-4)
        # rather than -inf.
        target_actions_scaled = target_actions_cpu / float(ACTION_SCALE)
        target_raw_cpu = torch.where(
            target_actions_scaled > 1e-4,
            torch.log(torch.expm1(target_actions_scaled.clamp(min=1e-4))),
            torch.full_like(target_actions_scaled, -8.0),
        ).clamp(min=-8.0, max=8.0)

        # Only update the actor pathway (encoder + trunk + mu head).
        trainable_params = (
            list(self.net.encoder.parameters())
            + list(self.net.actor_trunk.parameters())
            + list(self.net.actor_mu.parameters())
        )
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr if lr is not None else self.cfg.lr,
        )

        n_samples = states_cpu.shape[0]
        was_training = self.net.training
        self.net.train()
        first_epoch_mse = float("nan")
        last_epoch_mse  = float("nan")

        try:
            for epoch_index in range(epochs):
                shuffled = np.random.permutation(n_samples)
                running_sum  = 0.0
                running_seen = 0
                for batch_start in range(0, n_samples, minibatch):
                    batch_idx = shuffled[batch_start:batch_start + minibatch]
                    batch_idx_tensor = torch.as_tensor(
                        batch_idx, dtype=torch.long
                    )
                    batch_states = states_cpu.index_select(
                        0, batch_idx_tensor
                    ).to(self.device, non_blocking=True)
                    batch_fp = fraction_progresses_cpu.index_select(
                        0, batch_idx_tensor
                    ).to(self.device, non_blocking=True)
                    batch_target_raw = target_raw_cpu.index_select(
                        0, batch_idx_tensor
                    ).to(self.device, non_blocking=True)

                    features = self.net.features(batch_states, batch_fp)
                    predicted_mu = self.net.actor_mu(
                        self.net.actor_trunk(features)
                    )
                    loss = ((predicted_mu - batch_target_raw) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                    running_sum  += float(loss.item()) * len(batch_idx)
                    running_seen += len(batch_idx)

                epoch_mse = running_sum / max(running_seen, 1)
                if epoch_index == 0:
                    first_epoch_mse = epoch_mse
                last_epoch_mse = epoch_mse
                print(
                    f"  [warmstart] epoch {epoch_index + 1}/{epochs}  "
                    f"mse={epoch_mse:.4f}"
                )
        finally:
            if not was_training:
                self.net.eval()

        return {"pretrain_mse_first": first_epoch_mse,
                "pretrain_mse_last":  last_epoch_mse,
                "pretrain_n":         n_samples}

    # ------------------------------------------------------------ I/O
    def save(self, path: str):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"])