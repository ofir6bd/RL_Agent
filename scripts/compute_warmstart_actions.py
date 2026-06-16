"""Pre-compute and cache per-patient warm-start beamlet actions.

For each processed patient under ``data/processed/<split>/<pt>/``, this
script solves a *structure-weighted* non-negative least-squares fit to
the planner's per-fraction ground-truth dose:

    a* = argmin_{a >= 0}  || sqrt(W) * (DIM @ a - dose_gt / n_fractions) ||_2^2

with FISTA (accelerated projected gradient).  ``W`` upweights PTV voxels
(default 50x) and OARs (default 5x) so the fit doesn't get drowned by
the ~95% of body voxels with near-zero target dose.  Result is written to

    data/processed/<split>/<pt>/warmstart_action.npy        (shape (B,) float32)

Train.py loads these and supervised-pretrains the actor head against
inv_softplus(a*), which breaks the uniform-init policy ("every beamlet
~= softplus(-1) = 0.31") that PPO otherwise cannot escape in a
reasonable number of episodes.

Note on DIM coverage
--------------------
The current dose-influence matrix only covers ~70-85% of the body
voxels per patient (the 16x16 BEV fluence binning + integer-floor
downsampling miss the rest). Per-PTV-voxel D95 is therefore bounded
above by the *reachable* PTV mass, not by what the agent does. The
warm-start gives a near-best plan within that DIM capacity.

Usage
-----
python scripts/compute_warmstart_actions.py --config configs/default.yaml
    # by default: train + validation + test splits, skip patients already done

python scripts/compute_warmstart_actions.py --splits train --force
    # recompute only the train split
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# Allow running from project root: ``python scripts/compute_warmstart_actions.py``.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_config, PTV_NAMES, OAR_NAMES                 # noqa: E402
from src.data.preprocess import load_processed                           # noqa: E402


# Per-voxel weights for the NNLS loss.  PTVs dominate the fit so the
# coarse DIM concentrates dose on tumour voxels rather than spreading it
# evenly through the body. OARs get a moderate weight to avoid blowing
# them up when fitting dose_gt.
_DEFAULT_PTV_WEIGHT  = 50.0
_DEFAULT_OAR_WEIGHT  =  5.0
_DEFAULT_BODY_WEIGHT =  1.0


# ---------------------------------------------------------------------------
# NNLS solver (weighted FISTA, projected non-negative)
# ---------------------------------------------------------------------------

def _estimate_weighted_lipschitz(dose_influence_matrix: sp.csr_matrix,
                                 weights_per_voxel: np.ndarray,
                                 n_power_iters: int = 20,
                                 rng_seed: int = 0) -> float:
    """Largest eigenvalue of ``A.T @ diag(W) @ A`` via power iteration."""
    n_beamlets = dose_influence_matrix.shape[1]
    rng = np.random.default_rng(rng_seed)
    v = rng.standard_normal(n_beamlets).astype(np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return 1.0
    v /= v_norm
    for _ in range(n_power_iters):
        Av = dose_influence_matrix @ v
        v = dose_influence_matrix.T @ (weights_per_voxel * Av)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return 1.0
        v /= v_norm
    Av = dose_influence_matrix @ v
    largest_eigenvalue_estimate = float(np.dot(Av, weights_per_voxel * Av))
    return largest_eigenvalue_estimate + 1e-6


def fista_nnls_weighted(
    dose_influence_matrix: sp.csr_matrix,
    target_dose_flat: np.ndarray,
    weights_per_voxel: np.ndarray,
    n_iter: int = 400,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, float]:
    """Solve ``argmin_{a>=0} ||sqrt(W) (A a - b)||_2^2`` with FISTA.

    Returns ``(a_star, relative_residual_unweighted, relative_residual_weighted)``.
    """
    n_beamlets = dose_influence_matrix.shape[1]
    a_current = np.zeros(n_beamlets, dtype=np.float64)
    momentum_iterate = a_current.copy()
    momentum_weight = 1.0
    target_dose_flat = target_dose_flat.astype(np.float64)
    weights_per_voxel = weights_per_voxel.astype(np.float64)

    lipschitz = _estimate_weighted_lipschitz(
        dose_influence_matrix, weights_per_voxel,
    )
    step_size = 1.0 / lipschitz
    target_norm_unweighted = float(np.linalg.norm(target_dose_flat))
    target_norm_weighted = float(
        np.linalg.norm(np.sqrt(weights_per_voxel) * target_dose_flat)
    )
    last_objective = float("inf")

    for iteration_index in range(n_iter):
        residual = (
            dose_influence_matrix @ momentum_iterate - target_dose_flat
        )
        gradient = dose_influence_matrix.T @ (
            weights_per_voxel * residual
        )
        a_next = np.maximum(0.0, momentum_iterate - step_size * gradient)
        momentum_weight_next = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum_weight * momentum_weight)
        )
        momentum_iterate = a_next + (
            (momentum_weight - 1.0) / momentum_weight_next
        ) * (a_next - a_current)
        a_current = a_next
        momentum_weight = momentum_weight_next

        if iteration_index % 20 == 19:
            objective = 0.5 * float(
                np.dot(residual, weights_per_voxel * residual)
            )
            if last_objective - objective < tol * max(last_objective, 1.0):
                break
            last_objective = objective

    final_residual = dose_influence_matrix @ a_current - target_dose_flat
    relative_residual_unweighted = (
        float(np.linalg.norm(final_residual)) / max(target_norm_unweighted, 1e-9)
    )
    relative_residual_weighted = (
        float(np.linalg.norm(np.sqrt(weights_per_voxel) * final_residual))
        / max(target_norm_weighted, 1e-9)
    )
    return (
        a_current.astype(np.float32),
        relative_residual_unweighted,
        relative_residual_weighted,
    )


# ---------------------------------------------------------------------------
# Per-patient driver
# ---------------------------------------------------------------------------

def _load_dim(patient_dir: Path) -> sp.csr_matrix | None:
    """Return the patient's sparse dose-influence matrix or ``None``."""
    new_path    = patient_dir / "dose_influence_matrix.npz"
    legacy_path = patient_dir / "dim.npz"
    if new_path.exists():
        return sp.load_npz(new_path).astype(np.float32).tocsr()
    if legacy_path.exists():
        return sp.load_npz(legacy_path).astype(np.float32).tocsr()
    return None


def _build_voxel_weights(patient_dir: Path,
                         grid: int,
                         possible_dose_mask: np.ndarray,
                         ptv_weight: float,
                         oar_weight: float,
                         body_weight: float) -> np.ndarray:
    """Build the per-voxel weight volume W (flattened) for the NNLS loss."""
    cached_arrays = load_processed(patient_dir)
    structure_masks = cached_arrays.get("masks")
    structure_names = cached_arrays.get("structure_names")
    weights_volume = np.full(
        (grid, grid, grid), body_weight, dtype=np.float32,
    )
    if structure_masks is not None and structure_names is not None:
        names_list = list(structure_names.tolist())
        name_to_index = {n: i for i, n in enumerate(names_list)}
        for oar_name in OAR_NAMES:
            i = name_to_index.get(oar_name)
            if i is not None:
                weights_volume = np.maximum(
                    weights_volume, structure_masks[i] * oar_weight,
                )
        for ptv_name in PTV_NAMES:
            i = name_to_index.get(ptv_name)
            if i is not None:
                weights_volume = np.maximum(
                    weights_volume, structure_masks[i] * ptv_weight,
                )
    weights_volume = weights_volume * possible_dose_mask
    return weights_volume.ravel()


def _per_fraction_target(patient_dir: Path,
                         n_fractions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (per-fraction target dose flat, possible_dose_mask volume)."""
    dose_gt = np.load(patient_dir / "dose_gt.npy").astype(np.float32)
    possible_dose_mask = np.load(
        patient_dir / "possible_dose_mask.npy"
    ).astype(np.float32)
    target = (dose_gt / float(n_fractions)) * possible_dose_mask
    return target.ravel(), possible_dose_mask


def compute_for_split(processed_root: Path,
                      split: str,
                      n_fractions: int,
                      grid: int,
                      n_iter: int,
                      ptv_weight: float,
                      oar_weight: float,
                      body_weight: float,
                      force: bool) -> None:
    split_root = processed_root / split
    if not split_root.is_dir():
        print(f"  [skip] {split_root} not found")
        return
    patient_dirs = sorted(
        [p for p in split_root.iterdir() if p.is_dir()]
    )
    if not patient_dirs:
        print(f"  [skip] {split_root} is empty")
        return

    progress_bar = tqdm(patient_dirs, desc=f"warmstart[{split}]")
    n_done = 0
    n_skipped = 0
    n_missing_dim = 0
    for patient_dir in progress_bar:
        output_path = patient_dir / "warmstart_action.npy"
        if output_path.exists() and not force:
            n_skipped += 1
            continue
        dose_influence_matrix = _load_dim(patient_dir)
        if dose_influence_matrix is None:
            progress_bar.write(
                f"  [skip] {patient_dir.name}: no dose_influence_matrix.npz "
                f"(run compute_dose_influence_matrix.py first)"
            )
            n_missing_dim += 1
            continue
        target_dose_flat, possible_dose_mask = _per_fraction_target(
            patient_dir, n_fractions,
        )
        weights_flat = _build_voxel_weights(
            patient_dir, grid, possible_dose_mask,
            ptv_weight=ptv_weight,
            oar_weight=oar_weight,
            body_weight=body_weight,
        )

        t_start = time.time()
        a_star, relative_residual_unweighted, relative_residual_weighted = (
            fista_nnls_weighted(
                dose_influence_matrix,
                target_dose_flat,
                weights_flat,
                n_iter=n_iter,
            )
        )
        elapsed = time.time() - t_start
        np.save(output_path, a_star)
        n_done += 1

        active_beamlets = int((a_star > 1e-3).sum())
        progress_bar.write(
            f"  [{patient_dir.name}] a*: nnz={active_beamlets}/{a_star.size}  "
            f"mean={a_star.mean():.3f}  max={a_star.max():.3f}  "
            f"residual_w={relative_residual_weighted:.3f}  "
            f"residual_u={relative_residual_unweighted:.3f}  ({elapsed:.1f}s)"
        )

    print(
        f"[{split}] done={n_done}  skipped(existing)={n_skipped}  "
        f"missing_dim={n_missing_dim}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument(
        "--splits", nargs="+",
        default=["train", "validation", "test"],
        help="processed sub-folders to scan",
    )
    arg_parser.add_argument(
        "--n-iter", type=int, default=400,
        help="FISTA iterations per patient (400 is usually plenty)",
    )
    arg_parser.add_argument(
        "--ptv-weight", type=float, default=_DEFAULT_PTV_WEIGHT,
        help="per-voxel weight for PTV voxels in the NNLS loss",
    )
    arg_parser.add_argument(
        "--oar-weight", type=float, default=_DEFAULT_OAR_WEIGHT,
        help="per-voxel weight for OAR voxels",
    )
    arg_parser.add_argument(
        "--body-weight", type=float, default=_DEFAULT_BODY_WEIGHT,
        help="per-voxel weight for non-structure body voxels",
    )
    arg_parser.add_argument(
        "--force", action="store_true",
        help="recompute even if warmstart_action.npy already exists",
    )
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg.processed_dir)
    if not processed_root.is_dir():
        raise SystemExit(
            f"Processed root not found: {processed_root}. "
            f"Run scripts/preprocess.py first."
        )

    for split in args.splits:
        compute_for_split(
            processed_root, split,
            n_fractions=cfg.n_fractions,
            grid=cfg.grid,
            n_iter=args.n_iter,
            ptv_weight=args.ptv_weight,
            oar_weight=args.oar_weight,
            body_weight=args.body_weight,
            force=args.force,
        )


if __name__ == "__main__":
    main()
