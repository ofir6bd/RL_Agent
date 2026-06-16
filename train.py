"""Train the PPO agent on the OpenKBP dose-planning environment.

Each fraction is a 1-step PPO episode (``done=True`` after every step).
We buffer all ``n_fractions`` (35) transitions from one patient and run a
single PPO update per patient.  ``cfg.total_episodes`` therefore counts
*patients trained on*, not 35-step rollouts as in the previous design.

Usage
-----
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --resume runs/last.pt
python train.py --config configs/default.yaml --skip-warmstart \
    --resume runs/warmstart.pt
"""
from __future__ import annotations
import argparse
import os
from collections import deque
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm, trange

from src.config import load_config
from src.env.dose_env import DoseEnv
from src.agents.ppo import PPO, Rollout
from src.utils.metrics import dvh_score


def _print_fraction(episode_index: int,
                    info: dict,
                    action: np.ndarray,
                    cumulative_dose: np.ndarray,
                    reward: float,
                    n_fractions: int) -> None:
    """Print a one-line diagnostic between fractions."""
    print(
        f"  [ep {episode_index:>4d} | {info['patient']} | "
        f"fx {info['fraction_index']:>2d}/{n_fractions}] "
        f"R={reward:+8.3f}  oar={info['oar_penalty']:7.3f}  "
        f"ptv={info['ptv_reward']:+7.3f}  "
        f"a[mean/max]={action.mean():.3f}/{action.max():.3f}  "
        f"cum[max/mean]={cumulative_dose.max():.2f}"
        f"/{cumulative_dose.mean():.3f} Gy"
    )


def collect_patient(env: DoseEnv,
                    agent: PPO,
                    print_fractions: bool = False,
                    episode_index: int = 0):
    """Play one patient (35 single-fraction transitions) and return the
    buffered rollout + cumulative reward.

    Each transition has ``done=True`` so PPO treats it as a 1-step
    episode; the cumulative dose, however, persists across fractions
    because :meth:`DoseEnv.reset` is only called at the *start* of a
    patient course."""
    states: list[torch.Tensor]               = []
    fraction_progresses: list[torch.Tensor]  = []
    raw_action_samples: list[torch.Tensor]   = []
    log_probabilities: list[torch.Tensor]    = []
    rewards: list[torch.Tensor]              = []
    values: list[torch.Tensor]               = []
    dones: list[torch.Tensor]                = []

    state, fraction_progress = env.reset()
    patient_done = False
    patient_total_reward = 0.0
    while not patient_done:
        (action,
         raw_action_sample,
         log_probability,
         state_value) = agent.act(state, fraction_progress)
        # reshape (B,) -> (n_beams, H, W)
        action_3d = action.reshape(
            env.cfg.n_beams, env.cfg.beamlet_h, env.cfg.beamlet_w,
        )
        (next_state,
         next_fraction_progress,
         reward,
         done,
         info) = env.step(action_3d)
        patient_done = bool(info["patient_done"])

        if print_fractions:
            _print_fraction(episode_index, info, action, env.cumulative_dose,
                            reward, env.cfg.n_fractions)

        states.append(torch.from_numpy(state))
        fraction_progresses.append(
            torch.tensor(fraction_progress, dtype=torch.float32)
        )
        raw_action_samples.append(raw_action_sample)
        log_probabilities.append(
            torch.tensor(log_probability, dtype=torch.float32)
        )
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        values.append(torch.tensor(state_value, dtype=torch.float32))
        # ``done`` is True for every fraction (1-step PPO episodes), so
        # GAE collapses to advantage = reward - value, return = reward.
        dones.append(torch.tensor(float(done), dtype=torch.float32))

        state, fraction_progress = next_state, next_fraction_progress
        patient_total_reward += reward

    rollout = Rollout(
        states               = torch.stack(states),
        fraction_progresses  = torch.stack(fraction_progresses),
        raw_action_samples   = torch.stack(raw_action_samples),
        log_probabilities    = torch.stack(log_probabilities),
        rewards              = torch.stack(rewards),
        values               = torch.stack(values),
        dones                = torch.stack(dones),
    )
    return rollout, patient_total_reward


def _warmstart_actor(env: DoseEnv,
                     agent: PPO,
                     processed_split_root: Path,
                     *,
                     epochs: int,
                     minibatch: int,
                     lr: float) -> None:
    """Supervised pretraining of the actor against the cached NNLS plans.

    For every training patient that has a ``warmstart_action.npy`` we
    take ``state`` at fraction 0 and ask the actor head to reproduce
    ``inv_softplus(a*)``. This breaks the symmetry of the uniform-init
    policy (every beamlet emits ``softplus(-1) ~ 0.31``) which PPO
    cannot escape from in any reasonable number of episodes — see the
    "policy frozen at init" diagnosis in the project notes.
    """
    states_cpu:        list[torch.Tensor] = []
    fraction_progress_cpu: list[float]    = []
    targets_cpu:       list[torch.Tensor] = []
    n_missing = 0
    for patient_id in tqdm(env.patient_ids, desc="warmstart[collect]"):
        warmstart_path = (
            processed_split_root / patient_id / "warmstart_action.npy"
        )
        if not warmstart_path.exists():
            n_missing += 1
            continue
        state_at_fraction_zero, fraction_progress = env.reset(patient_id)
        warmstart_action = np.load(warmstart_path).astype(np.float32)
        if warmstart_action.shape[0] != env.cfg.n_beamlets:
            raise RuntimeError(
                f"{warmstart_path}: expected {env.cfg.n_beamlets} beamlets, "
                f"got {warmstart_action.shape[0]}."
            )
        states_cpu.append(torch.from_numpy(state_at_fraction_zero))
        fraction_progress_cpu.append(float(fraction_progress))
        targets_cpu.append(torch.from_numpy(warmstart_action))

    if not states_cpu:
        print(
            "[warmstart] no warmstart_action.npy found in "
            f"{processed_split_root}. Run "
            f"`python scripts/compute_warmstart_actions.py --splits "
            f"{processed_split_root.name}` first, or set "
            f"warmstart_enabled: false in the config to skip."
        )
        return

    print(
        f"[warmstart] pretraining actor on {len(states_cpu)} patient(s) "
        f"({n_missing} missing warmstart files) for {epochs} epochs..."
    )
    states_tensor = torch.stack(states_cpu)
    fraction_progress_tensor = torch.tensor(
        fraction_progress_cpu, dtype=torch.float32
    )
    targets_tensor = torch.stack(targets_cpu)
    stats = agent.pretrain_actor(
        states_tensor,
        fraction_progress_tensor,
        targets_tensor,
        epochs=epochs,
        minibatch=minibatch,
        lr=lr,
    )
    print(
        f"[warmstart] done. mse: first={stats['pretrain_mse_first']:.4f}  "
        f"last={stats['pretrain_mse_last']:.4f}  n={stats['pretrain_n']}"
    )
    # Show the symmetry-break: deterministic action stats on one patient.
    with torch.no_grad():
        state_after, fp_after = env.reset(env.patient_ids[0])
        action, _raw, _logp, _v = agent.act(
            state_after, fp_after, deterministic=True,
        )
        print(
            f"[warmstart] sample deterministic action on "
            f"{env.patient_ids[0]}: mean={action.mean():.3f}  "
            f"max={action.max():.3f}  active(>1e-3)={(action > 1e-3).mean():.2f}"
        )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument(
        "--episodes", type=int, default=None,
        help="override total_episodes (== number of patients to "
             "train on; one PPO update per patient)",
    )
    arg_parser.add_argument(
        "--print-fractions", action="store_true",
        help="print per-fraction diagnostics (reward components, "
             "action stats, cumulative dose stats)",
    )
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.total_episodes = args.episodes
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = DoseEnv(cfg, split=cfg.train_split)
    n_patients = len(env.patient_ids)
    if n_patients == 0:
        raise SystemExit(
            f"No processed patients found under "
            f"{cfg.processed_dir}/{cfg.train_split}"
        )

    # default: one episode (= one patient = 35 fractions = 1 PPO update)
    # per processed patient.  Override with --episodes for longer runs.
    if args.episodes is None:
        cfg.total_episodes = n_patients
    print(f"[train] {n_patients} patient(s) found, "
          f"running {cfg.total_episodes} patient-episode(s) "
          f"(1 PPO update per patient, {cfg.n_fractions} fractions buffered)")

    # peek a state to get the channel count
    initial_state, _initial_fraction_progress = env.reset()
    in_channels = initial_state.shape[0]
    agent = PPO(cfg, in_channels=in_channels)

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- warm-start
    if getattr(cfg, "warmstart_enabled", False) and cfg.warmstart_epochs > 0:
        processed_split_root = Path(cfg.processed_dir) / cfg.train_split
        _warmstart_actor(
            env, agent, processed_split_root,
            epochs=cfg.warmstart_epochs,
            minibatch=cfg.warmstart_minibatch,
            lr=cfg.warmstart_lr,
        )
        agent.save(os.path.join(cfg.ckpt_dir, "warmstart.pt"))

    # ---------------------------------------------------------------- PPO loop
    # ``best.pt`` is saved by *rolling-mean* patient reward over the last
    # ``cfg.best_rolling_window`` patients, not a single patient's return
    # (which used to pin best.pt to the easiest training patient).
    recent_patient_rewards: deque[float] = deque(
        maxlen=max(int(cfg.best_rolling_window), 1)
    )
    best_rolling_mean_reward = -float("inf")

    progress_bar = trange(cfg.total_episodes, desc="train")
    for episode_index in progress_bar:
        rollout, patient_total_reward = collect_patient(
            env, agent,
            print_fractions=args.print_fractions,
            episode_index=episode_index,
        )
        # ``last_value=0`` is correct here: every transition already has
        # ``done=True`` so the bootstrap value would be masked out anyway.
        stats = agent.update(rollout, last_value=0.0)
        recent_patient_rewards.append(float(patient_total_reward))

        # ---- policy-drift diagnostics (so a frozen policy is visible) ----
        with torch.no_grad():
            actor_mu_bias_mean = float(
                agent.net.actor_mu.bias.mean().item()
            )
            actor_log_std_mean = float(
                agent.net.log_std.mean().item()
            )
            # last-fraction deterministic action stats from this rollout
            last_raw_sample = rollout.raw_action_samples[-1]
            last_action = (
                torch.nn.functional.softplus(last_raw_sample).numpy()
            )
        progress_bar.set_postfix(
            R=f"{patient_total_reward:.2f}",
            Ravg=f"{np.mean(recent_patient_rewards):.2f}",
            pi=f"{stats['policy_loss']:.3f}",
            v=f"{stats['value_loss']:.3f}",
            mu=f"{actor_mu_bias_mean:+.3f}",
            lstd=f"{actor_log_std_mean:+.3f}",
            a_mean=f"{last_action.mean():.3f}",
            a_max=f"{last_action.max():.2f}",
        )

        if len(recent_patient_rewards) >= recent_patient_rewards.maxlen:
            rolling_mean = float(np.mean(recent_patient_rewards))
            if rolling_mean > best_rolling_mean_reward:
                best_rolling_mean_reward = rolling_mean
                agent.save(os.path.join(cfg.ckpt_dir, "best.pt"))
        if (episode_index + 1) % cfg.eval_every == 0:
            agent.save(
                os.path.join(cfg.ckpt_dir, f"ep{episode_index + 1}.pt")
            )

    # Always keep a final checkpoint, even if total_episodes is not a
    # multiple of eval_every.
    agent.save(os.path.join(cfg.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    main()
