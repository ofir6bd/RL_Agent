"""Deterministic evaluation on a list of (held-out) patient ids.

Usage
-----
python evaluate.py --config configs/default.yaml --ckpt runs/best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import torch

from src.config import load_config, OAR_NAMES, PTV_NAMES
from src.env.dose_env import DoseEnv
from src.agents.ppo import PPO
from src.utils.metrics import (d_at_volume, mae, oar_mean_doses,
                               dvh_score)
from src.utils.visualize import generate_eval_charts


def _print_fraction(info: dict, action: np.ndarray, cumulative_dose: np.ndarray,
                    reward: float, n_fractions: int) -> None:
    """Print a one-line diagnostic between fractions."""
    print(
        f"    [{info['patient']} | fx {info['fraction_index']:>2d}/{n_fractions}] "
        f"R={reward:+8.3f}  oar={info['oar_penalty']:7.3f}  "
        f"ptv={info['ptv_reward']:+7.3f}  "
        f"a[mean/max]={action.mean():.3f}/{action.max():.3f}  "
        f"cum[max/mean]={cumulative_dose.max():.2f}"
        f"/{cumulative_dose.mean():.3f} Gy"
    )


def run_one(env: DoseEnv,
            agent: PPO,
            print_fractions: bool = False,
            collect_charts_data: bool = False) -> Dict:
    state, fraction_progress = env.reset()
    patient_done = False
    accumulated_action = np.zeros(
        env.cfg.n_beams * env.cfg.beamlet_h * env.cfg.beamlet_w,
        dtype=np.float32,
    )
    fraction_records: List[Dict] = []
    while not patient_done:
        action, _raw, _logp, _value = agent.act(
            state, fraction_progress, deterministic=True,
        )
        action_3d = action.reshape(
            env.cfg.n_beams, env.cfg.beamlet_h, env.cfg.beamlet_w,
        )
        (state,
         fraction_progress,
         reward,
         _done,
         info) = env.step(action_3d)
        patient_done = bool(info["patient_done"])
        if collect_charts_data:
            accumulated_action += action.ravel()
            fraction_records.append({
                "fraction_index": info["fraction_index"],
                "oar_penalty":    info["oar_penalty"],
                "ptv_reward":     info["ptv_reward"],
            })
        if print_fractions:
            _print_fraction(info, action, env.cumulative_dose, reward,
                            env.cfg.n_fractions)

    predicted_dose = env.cumulative_dose
    reference_dose = env._data["dose_gt"]
    structure_masks = env._structure_masks()
    ptv_masks = {
        ptv_name: structure_masks[ptv_name]
        for ptv_name in PTV_NAMES
        if ptv_name in structure_masks and structure_masks[ptv_name].sum() > 0
    }
    oar_masks = {
        oar_name: structure_masks[oar_name]
        for oar_name in OAR_NAMES
        if oar_name in structure_masks and structure_masks[oar_name].sum() > 0
    }

    # NOTE: D95r_<PTV> / %reach_<PTV> were dropped from the report once the
    # DIM downsampler fix put PTV reachability at 99.7-100% across all
    # patients, so D95r matches D95 within rounding. The helpers
    # ``d_at_volume_reachable`` / ``reachable_fraction`` in
    # ``src/utils/metrics.py`` are kept as a regression guardrail; recompute
    # ``reachable_volume = (DIM.sum(axis=1) > 0)`` here if you need them.

    patient_metrics = {
        "patient":             env._patient,
        "mae":                 mae(predicted_dose, reference_dose,
                                   env._data["possible_dose_mask"]),
        "dvh_score":           dvh_score(predicted_dose, reference_dose,
                                         structure_masks),
        "oar_mean":            oar_mean_doses(predicted_dose, oar_masks),
        # chart extras (only populated when collect_charts_data=True)
        "_accumulated_action": accumulated_action,
        "_fraction_records":   fraction_records,
        "_predicted_dose":     predicted_dose,
        "_reference_dose":     reference_dose,
    }
    for ptv_name, ptv_mask in ptv_masks.items():
        patient_metrics[f"D95_{ptv_name}"] = d_at_volume(
            predicted_dose, ptv_mask, 0.95,
        )
    return patient_metrics


def _fmt_cell(value) -> str:
    """Render a numeric cell with 2 decimals, dash for missing values."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "  -  "
    return f"{value:.2f}"


def print_results_table(results: Sequence[Dict],
                        oar_tolerance: Dict[str, float] | None = None) -> None:
    """Print all per-patient metrics as a single aligned table."""
    if not results:
        print("(no results)")
        return

    # Build column list: fixed metrics first, then any PTV D95 / OAR mean
    # columns observed across patients (preserving canonical ordering).
    base_columns = [("patient", "Patient"),
                    ("mae", "MAE"),
                    ("dvh_score", "DVH")]
    ptv_columns: list[tuple[str, str]] = []
    for ptv_name in PTV_NAMES:
        if any(f"D95_{ptv_name}" in patient_metrics
               for patient_metrics in results):
            ptv_columns.append((f"D95_{ptv_name}",  f"D95_{ptv_name}"))
    def _oar_header(oar_name: str) -> str:
        if oar_tolerance and oar_name in oar_tolerance:
            tol = oar_tolerance[oar_name]
            tol_str = str(int(tol)) if tol == int(tol) else f"{tol:.1f}"
            return f"{oar_name}({tol_str}Gy)"
        return oar_name

    oar_columns = [
        (f"OAR_{oar_name}", _oar_header(oar_name)) for oar_name in OAR_NAMES
        if any(oar_name in patient_metrics.get("oar_mean", {})
               for patient_metrics in results)
    ]
    columns = base_columns + ptv_columns + oar_columns

    # Flatten oar_mean into top-level keys so rows are uniform.
    flat_rows: List[Dict[str, object]] = []
    for patient_metrics in results:
        flat_row = {
            key: value for key, value in patient_metrics.items()
            if key != "oar_mean"
        }
        for oar_name, oar_mean in patient_metrics.get("oar_mean", {}).items():
            flat_row[f"OAR_{oar_name}"] = oar_mean
        flat_rows.append(flat_row)

    # Format every cell as a string and compute column widths.
    rendered_rows: List[List[str]] = []
    for row in flat_rows:
        rendered_cells: List[str] = []
        for column_key, _ in columns:
            value = row.get(column_key)
            rendered_cells.append(
                value if column_key == "patient" and isinstance(value, str)
                else _fmt_cell(value)
            )
        rendered_rows.append(rendered_cells)
    headers = [header for _, header in columns]
    column_widths = [
        max(len(headers[col_idx]),
            *(len(rendered_row[col_idx]) for rendered_row in rendered_rows))
        for col_idx in range(len(columns))
    ]

    def format_row(cells: Sequence[str]) -> str:
        return " | ".join(
            cell.rjust(width) if col_idx > 0 else cell.ljust(width)
            for col_idx, (cell, width) in enumerate(zip(cells, column_widths))
        )

    separator = "-+-".join("-" * width for width in column_widths)
    print()
    print(format_row(headers))
    print(separator)
    for rendered_row in rendered_rows:
        print(format_row(rendered_row))

    # Mean row across patients (only over numeric columns).
    mean_cells: List[str] = []
    for col_idx, (column_key, _) in enumerate(columns):
        if col_idx == 0:
            mean_cells.append("MEAN")
            continue
        numeric_values = [
            row.get(column_key) for row in flat_rows
            if isinstance(row.get(column_key), (int, float))
            and np.isfinite(row.get(column_key))
        ]
        mean_cells.append(
            _fmt_cell(float(np.mean(numeric_values))) if numeric_values
            else "  -  "
        )
    print(separator)
    print(format_row(mean_cells))


def print_config_summary(cfg, config_path: str) -> None:
    """Print every user-controllable hyperparameter that can influence results.

    Anything in this block is something *you* set in the YAML (or via the
    Config dataclass defaults) and that can change evaluation numbers.
    """
    sep = "=" * 60
    sub = "-" * 60
    print(f"\n{sep}")
    print(f"  CONFIG SNAPSHOT  ({config_path})")
    print(sep)

    # --- data / geometry ---
    print(f"  [data / geometry]")
    print(f"    {'raw_dir':<28} {cfg.raw_dir}")
    print(f"    {'processed_dir':<28} {cfg.processed_dir}")
    print(f"    {'train_split':<28} {cfg.train_split}")
    print(f"    {'eval_split':<28} {cfg.eval_split}")
    print(f"    {'grid':<28} {cfg.grid}")
    print(f"    {'n_fractions':<28} {cfg.n_fractions}")

    # --- beam geometry ---
    print(f"\n  [beam geometry]")
    print(f"    {'n_beams':<28} {cfg.n_beams}")
    print(f"    {'beamlet_h':<28} {cfg.beamlet_h}")
    print(f"    {'beamlet_w':<28} {cfg.beamlet_w}")
    print(f"    {'n_beamlets (derived)':<28} {cfg.n_beamlets}")

    # --- prescriptions ---
    print(f"\n  [prescriptions (Gy, full course)]")
    for ptv, rx in cfg.prescription.items():
        print(f"    {ptv:<28} {rx}")

    # --- OAR tolerances + weights + serial flags ---
    print(f"\n  [OAR tolerances (Gy) / weights / serial]")
    for oar in cfg.oar_tolerance:
        tol = cfg.oar_tolerance[oar]
        w = cfg.oar_weights.get(oar, 1.0)
        serial = cfg.oar_serial.get(oar, False)
        serial_tag = "serial" if serial else "parallel"
        print(f"    {oar:<22} tol={tol:<6}  w={w:<5}  [{serial_tag}]")

    # --- reward weights ---
    print(f"\n  [reward weights]")
    print(f"    {'lambda_ptv':<28} {cfg.lambda_ptv}")
    print(f"    {'lambda_oar':<28} {cfg.lambda_oar}")
    print(f"    {'oar_voxel_subweight':<28} {cfg.oar_voxel_subweight}")
    print(f"    {'oar_mean_subweight':<28} {cfg.oar_mean_subweight}")
    print(f"    {'oar_dmax_subweight':<28} {cfg.oar_dmax_subweight}")

    # --- PPO ---
    print(f"\n  [PPO]")
    print(f"    {'gamma':<28} {cfg.gamma}")
    print(f"    {'gae_lambda':<28} {cfg.gae_lambda}")
    print(f"    {'clip_eps':<28} {cfg.clip_eps}")
    print(f"    {'ent_coef':<28} {cfg.ent_coef}")
    print(f"    {'vf_coef':<28} {cfg.vf_coef}")
    print(f"    {'lr':<28} {cfg.lr}")
    print(f"    {'ppo_epochs':<28} {cfg.ppo_epochs}")
    print(f"    {'minibatch':<28} {cfg.minibatch}")

    # --- training loop ---
    print(f"\n  [training loop]")
    print(f"    {'total_episodes':<28} {cfg.total_episodes}")
    print(f"    {'eval_every':<28} {cfg.eval_every}")
    print(f"    {'seed':<28} {cfg.seed}")
    print(f"    {'device':<28} {cfg.device}")
    print(f"    {'ckpt_dir':<28} {cfg.ckpt_dir}")

    # --- warm-start (supervised actor pretraining) ---
    print(f"\n  [warm-start (supervised actor pretrain)]")
    print(f"    {'warmstart_enabled':<28} {cfg.warmstart_enabled}")
    print(f"    {'warmstart_epochs':<28} {cfg.warmstart_epochs}")
    print(f"    {'warmstart_minibatch':<28} {cfg.warmstart_minibatch}")
    print(f"    {'warmstart_lr':<28} {cfg.warmstart_lr}")

    # --- OAR-weight curriculum ---
    print(f"\n  [OAR-weight curriculum]")
    print(f"    {'lambda_oar_ramp_episodes':<28} {cfg.lambda_oar_ramp_episodes}")
    print(f"    {'lambda_oar_ramp_start_factor':<28} {cfg.lambda_oar_ramp_start_factor}")

    # --- best.pt selection ---
    print(f"\n  [best.pt selection]")
    print(f"    {'best_n_val_patients':<28} {cfg.best_n_val_patients}")
    print(f"    {'best_eval_every':<28} {cfg.best_eval_every}")
    print(f"    {'best_rolling_window':<28} {cfg.best_rolling_window}")

    print(sub)
    print("  (Every value above is set by you in the YAML / Config and can "
          "change\n   evaluation numbers. PPO/training/warm-start/curriculum "
          "knobs only\n   affect results indirectly via the checkpoint that "
          "produced runs/best.pt.)")
    print(sep)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument("--ckpt",   default="runs/best.pt")
    arg_parser.add_argument(
        "--split", default=None,
        help="processed split to evaluate on (defaults to "
             "cfg.eval_split, e.g. validation or test)",
    )
    arg_parser.add_argument(
        "--patients", nargs="*", default=None,
        help="patient ids to evaluate (defaults to every "
             "processed pt in the chosen split)",
    )
    arg_parser.add_argument(
        "--print-fractions", action="store_true",
        help="print per-fraction diagnostics (reward components, "
             "action stats, cumulative dose stats)",
    )
    arg_parser.add_argument(
        "--charts-patient", default=None, metavar="PATIENT_ID",
        help="patient id for which to generate evaluation charts "
             "(fluence maps, DIM sensitivity, DVH comparison, dose "
             "slices, per-fraction rewards). Saved to "
             "<ckpt_dir>/eval_charts/<patient>/. "
             "Use 'first' to pick the first evaluated patient automatically.",
    )
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg.eval_split
    env = DoseEnv(cfg, patient_ids=args.patients, split=split)
    print(f"[evaluate] split = {split}  (n={len(env.patient_ids)})  "
          f"ckpt = {args.ckpt}")
    initial_state, _initial_fraction_progress = env.reset()
    agent = PPO(cfg, in_channels=initial_state.shape[0])
    agent.load(args.ckpt)
    agent.net.eval()

    patient_ids = list(env.patient_ids)
    charts_target = args.charts_patient
    if charts_target == "first" and patient_ids:
        charts_target = patient_ids[0]

    results: List[Dict] = []
    for patient_position, patient_id in enumerate(patient_ids, start=1):
        env.patient_ids = [patient_id]
        print(f"  [{patient_position}/{len(patient_ids)}] {patient_id} ...",
              flush=True)
        want_charts = (patient_id == charts_target)
        patient_result = run_one(
            env, agent,
            print_fractions=args.print_fractions,
            collect_charts_data=want_charts,
        )
        results.append(patient_result)

        if want_charts:
            chart_dir = Path(cfg.ckpt_dir) / "eval_charts" / patient_id
            chart_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Generating evaluation charts → {chart_dir}")
            saved = generate_eval_charts(
                env,
                patient_result["_accumulated_action"],
                patient_result["_predicted_dose"],
                patient_result["_reference_dose"],
                patient_result["_fraction_records"],
                chart_dir,
            )
            for p in saved:
                print(f"    saved: {p}")

    print_results_table(results, oar_tolerance=cfg.oar_tolerance)
    print("\n(Doses in Gy. MAE = mean abs error in body; "
          "DVH = mean |D95 + Dmean| diff vs ground-truth, lower is better.)")
    print("D95_<PTV> = full-mask D95 (post-DIM-fix, every PTV voxel is "
          "reachable so this is the only coverage column we need).")
    print_config_summary(cfg, args.config)


if __name__ == "__main__":
    main()


