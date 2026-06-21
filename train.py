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


def _effective_lambda_oar(cfg, episode_index: int) -> float:
    """Linearly ramp lambda_oar over the first ``lambda_oar_ramp_episodes``
    patients of PPO training.

    Starts at ``cfg.lambda_oar * cfg.lambda_oar_ramp_start_factor`` and
    reaches the full ``cfg.lambda_oar`` at ``episode_index ==
    cfg.lambda_oar_ramp_episodes``. After that it stays at the full
    value. Setting ``cfg.lambda_oar_ramp_episodes`` to 0 disables the
    ramp (returns ``cfg.lambda_oar`` immediately).
    """
    ramp_episodes = int(getattr(cfg, "lambda_oar_ramp_episodes", 0) or 0)
    target = float(cfg.lambda_oar)
    if ramp_episodes <= 0:
        return target
    start_factor = float(
        getattr(cfg, "lambda_oar_ramp_start_factor", 0.25)
    )
    progress = min(1.0, max(0.0, episode_index / float(ramp_episodes)))
    factor = start_factor + (1.0 - start_factor) * progress
    return target * factor


@torch.no_grad()
def _validation_dvh_score(agent: PPO,
                          val_env: DoseEnv,
                          n_patients: int) -> float:
    """Run ``n_patients`` deterministic rollouts and return the mean
    DVH-score (lower is better) of the predicted vs ground-truth dose.

    This is the criterion used to save ``best.pt``.  Uses the
    ``val_env`` (a separate ``DoseEnv`` over ``cfg.eval_split``); does
    NOT mutate the training env's lambda_oar/lambda_ptv.
    """
    from src.config import OAR_NAMES, PTV_NAMES  # local to avoid cycle in tests

    if n_patients <= 0 or len(val_env.patient_ids) == 0:
        return float("nan")

    was_training = agent.net.training
    agent.net.eval()
    try:
        scores = []
        for patient_id in val_env.patient_ids[:n_patients]:
            state, fraction_progress = val_env.reset(patient_id)
            patient_done = False
            while not patient_done:
                action, _raw, _logp, _v = agent.act(
                    state, fraction_progress, deterministic=True,
                )
                action_3d = action.reshape(
                    val_env.cfg.n_beams,
                    val_env.cfg.beamlet_h,
                    val_env.cfg.beamlet_w,
                )
                state, fraction_progress, _r, _d, info = val_env.step(action_3d)
                patient_done = bool(info["patient_done"])

            predicted_dose = val_env.cumulative_dose
            reference_dose = val_env._data["dose_gt"]
            structure_masks = val_env._structure_masks()
            scores.append(
                dvh_score(predicted_dose, reference_dose, structure_masks)
            )
        return float(np.mean(scores)) if scores else float("nan")
    finally:
        if was_training:
            agent.net.train()


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
    arg_parser.add_argument(
        "--resume", default=None, metavar="CKPT",
        help="resume PPO from an existing checkpoint (skips warm-start "
             "unless --force-warmstart is given)",
    )
    arg_parser.add_argument(
        "--force-warmstart", action="store_true",
        help="run the supervised warm-start even when --resume is set",
    )
    arg_parser.add_argument(
        "--skip-warmstart", action="store_true",
        help="skip the supervised actor warm-start regardless of "
             "warmstart_enabled in the config (useful when reusing "
             "runs/warmstart.pt)",
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

    # Episode count resolution:
    #   1) --episodes on the CLI always wins (already applied above).
    #   2) Otherwise respect cfg.total_episodes from the YAML.
    #   3) Fall back to n_patients only when cfg.total_episodes is unset
    #      or non-positive. (Older runs silently overwrote the YAML value
    #      with n_patients, which made YAML's total_episodes meaningless
    #      and capped training at one pass through the train split.)
    if args.episodes is None and (
        not getattr(cfg, "total_episodes", 0)
        or int(cfg.total_episodes) <= 0
    ):
        cfg.total_episodes = n_patients
    cfg.total_episodes = int(cfg.total_episodes)
    print(f"[train] {n_patients} patient(s) found, "
          f"running {cfg.total_episodes} patient-episode(s) "
          f"(1 PPO update per patient, {cfg.n_fractions} fractions buffered)")

    # peek a state to get the channel count
    initial_state, _initial_fraction_progress = env.reset()
    in_channels = initial_state.shape[0]
    agent = PPO(cfg, in_channels=in_channels)

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- resume
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")
        agent.load(str(resume_path))
        print(f"[resume] loaded {resume_path}")

    # ---------------------------------------------------------------- warm-start
    do_warmstart = (
        getattr(cfg, "warmstart_enabled", False)
        and cfg.warmstart_epochs > 0
        and not args.skip_warmstart
        and (args.resume is None or args.force_warmstart)
    )
    if do_warmstart:
        processed_split_root = Path(cfg.processed_dir) / cfg.train_split
        _warmstart_actor(
            env, agent, processed_split_root,
            epochs=cfg.warmstart_epochs,
            minibatch=cfg.warmstart_minibatch,
            lr=cfg.warmstart_lr,
        )
        agent.save(os.path.join(cfg.ckpt_dir, "warmstart.pt"))
    elif args.skip_warmstart:
        print("[train] --skip-warmstart given; warm-start phase skipped")
    elif args.resume is not None:
        print("[train] --resume given; warm-start phase skipped "
              "(pass --force-warmstart to run it anyway)")

    # ---------------------------------------------------------------- validation env
    # A small, separate env over cfg.eval_split is used to score
    # candidate policies for best.pt by DVH score (lower is better),
    # which is the metric the project actually cares about.  We capped
    # it at cfg.best_n_val_patients to keep the cost per check small.
    n_val_patients = max(1, int(getattr(cfg, "best_n_val_patients", 3)))
    try:
        val_env = DoseEnv(cfg, split=cfg.eval_split)
        n_val_available = min(n_val_patients, len(val_env.patient_ids))
        if n_val_available == 0:
            print(f"[val] no patients in split '{cfg.eval_split}'; "
                  f"best.pt will fall back to training-reward criterion.")
            val_env = None  # type: ignore[assignment]
        else:
            print(f"[val] using {n_val_available} patient(s) from "
                  f"split '{cfg.eval_split}' for best.pt selection")
    except FileNotFoundError as e:
        print(f"[val] {e}; best.pt will fall back to training-reward criterion.")
        val_env = None  # type: ignore[assignment]

    # ---------------------------------------------------------------- PPO loop
    # best.pt selection: validation DVH score (primary). The legacy
    # rolling-mean training reward is kept only as a fallback when no
    # validation patients are available (and as a printed diagnostic).
    recent_patient_rewards: deque[float] = deque(
        maxlen=max(int(cfg.best_rolling_window), 1)
    )
    best_val_dvh = float("inf")
    best_rolling_mean_reward = -float("inf")
    eval_every_best = max(1, int(getattr(cfg, "best_eval_every", 25)))

    progress_bar = trange(cfg.total_episodes, desc="train")
    for episode_index in progress_bar:
        # OAR-weight curriculum: ramp lambda_oar over the first
        # cfg.lambda_oar_ramp_episodes patients so PPO can learn to
        # actually deliver dose before being heavily penalised for
        # incidental OAR irradiation.
        env.lambda_oar = _effective_lambda_oar(cfg, episode_index)
        env.lambda_ptv = float(cfg.lambda_ptv)

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
            loar=f"{env.lambda_oar:.2f}",
            pi=f"{stats['policy_loss']:.3f}",
            v=f"{stats['value_loss']:.3f}",
            mu=f"{actor_mu_bias_mean:+.3f}",
            lstd=f"{actor_log_std_mean:+.3f}",
            a_mean=f"{last_action.mean():.3f}",
            a_max=f"{last_action.max():.2f}",
        )

        # Periodic validation-DVH check for best.pt (primary criterion).
        if val_env is not None and (episode_index + 1) % eval_every_best == 0:
            val_dvh = _validation_dvh_score(agent, val_env, n_val_patients)
            tqdm.write(
                f"  [val ep {episode_index + 1}] "
                f"mean DVH score on {n_val_patients} patient(s) = "
                f"{val_dvh:.3f}  "
                f"(best so far: "
                f"{best_val_dvh if np.isfinite(best_val_dvh) else float('nan'):.3f})"
            )
            if np.isfinite(val_dvh) and val_dvh < best_val_dvh:
                best_val_dvh = val_dvh
                agent.save(os.path.join(cfg.ckpt_dir, "best.pt"))
                tqdm.write(
                    f"  [val ep {episode_index + 1}] saved best.pt "
                    f"(DVH score {val_dvh:.3f})"
                )

        # Fallback rolling-mean criterion only when no validation env.
        if val_env is None and len(recent_patient_rewards) >= recent_patient_rewards.maxlen:
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

    # Final validation pass so the user sees the end-of-training DVH
    # score and so best.pt has a chance to be updated by the very last
    # policy if it improved between the last periodic check and the
    # end of training.
    if val_env is not None:
        final_val_dvh = _validation_dvh_score(agent, val_env, n_val_patients)
        print(f"[final val] mean DVH score on "
              f"{n_val_patients} patient(s) = {final_val_dvh:.3f}")
        if np.isfinite(final_val_dvh) and final_val_dvh < best_val_dvh:
            best_val_dvh = final_val_dvh
            agent.save(os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"[final val] saved best.pt (DVH score {final_val_dvh:.3f})")
        print(f"[final val] best DVH score across run: {best_val_dvh:.3f}")


if __name__ == "__main__":
    main()
