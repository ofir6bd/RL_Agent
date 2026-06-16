"""DVH / D95 / MAE / OAR-mean utilities."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def dvh_curve(dose: np.ndarray,
              mask: np.ndarray,
              n_bins: int = 200,
              max_dose: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (dose_axis, fraction_volume_at_or_above)."""
    if mask.sum() == 0:
        return np.linspace(0, max_dose, n_bins), np.zeros(n_bins)
    dose_in_mask = dose[mask > 0]
    dose_axis = np.linspace(0, max_dose, n_bins)
    fraction_volume_at_or_above = np.array(
        [(dose_in_mask >= dose_threshold).mean() for dose_threshold in dose_axis],
        dtype=np.float64,
    )
    return dose_axis, fraction_volume_at_or_above


def d_at_volume(dose: np.ndarray,
                mask: np.ndarray,
                volume_fraction: float = 0.95) -> float:
    """Dose received by at least ``volume_fraction`` of the masked volume (e.g. D95)."""
    if mask.sum() == 0:
        return 0.0
    sorted_dose_in_mask = np.sort(dose[mask > 0])
    threshold_voxel_index = int(
        np.floor((1.0 - volume_fraction) * sorted_dose_in_mask.size)
    )
    return float(sorted_dose_in_mask[threshold_voxel_index])


def d_at_volume_reachable(dose: np.ndarray,
                          mask: np.ndarray,
                          reachable_mask: np.ndarray,
                          volume_fraction: float = 0.95) -> float:
    """``d_at_volume`` restricted to voxels that any beamlet can reach.

    The dose-influence matrix in this project only covers ~70-85% of body
    voxels (16x16 BEV fluence binning + integer-floor downsampling). The
    unreachable voxels are guaranteed to remain at 0 dose regardless of
    the agent's action. Reporting D95 over the *reachable* subset of the
    mask is the honest upper bound the agent can actually approach. Pass
    a boolean ``reachable_mask`` (the same shape as ``mask``) where
    ``True`` indicates the voxel has at least one non-zero beamlet
    contribution.
    """
    if mask.sum() == 0:
        return 0.0
    combined_mask = (mask > 0) & (reachable_mask > 0)
    n_reachable = int(combined_mask.sum())
    if n_reachable == 0:
        return 0.0
    sorted_reachable_dose = np.sort(dose[combined_mask])
    threshold_voxel_index = int(
        np.floor((1.0 - volume_fraction) * n_reachable)
    )
    return float(sorted_reachable_dose[threshold_voxel_index])


def reachable_fraction(mask: np.ndarray,
                       reachable_mask: np.ndarray) -> float:
    """Fraction of ``mask`` voxels that any beamlet can reach (0 if empty)."""
    n_total = int(mask.sum())
    if n_total == 0:
        return 0.0
    n_reachable = int(((mask > 0) & (reachable_mask > 0)).sum())
    return n_reachable / n_total


def mae(predicted_dose: np.ndarray,
        reference_dose: np.ndarray,
        mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.mean(np.abs(predicted_dose - reference_dose)))
    if mask.sum() == 0:
        return 0.0
    return float(
        np.mean(np.abs(predicted_dose[mask > 0] - reference_dose[mask > 0]))
    )


def oar_mean_doses(dose: np.ndarray,
                   oar_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    organ_mean_doses: Dict[str, float] = {}
    for organ_name, organ_mask in oar_masks.items():
        organ_mean_doses[organ_name] = (
            float((dose * organ_mask).sum() / max(organ_mask.sum(), 1.0))
            if organ_mask.sum() else 0.0
        )
    return organ_mean_doses


def dvh_score(predicted_dose: np.ndarray,
              reference_dose: np.ndarray,
              structures: Dict[str, np.ndarray]) -> float:
    """Cheap proxy of the OpenKBP DVH score: mean |D95 + Dmean| diff over
    structures.  Lower is better."""
    per_structure_diffs = []
    for _structure_name, structure_mask in structures.items():
        if structure_mask.sum() == 0:
            continue
        d95_diff = abs(
            d_at_volume(predicted_dose, structure_mask, 0.95)
            - d_at_volume(reference_dose, structure_mask, 0.95)
        )
        mean_dose_diff = abs(
            (predicted_dose * structure_mask).sum() / structure_mask.sum()
            - (reference_dose * structure_mask).sum() / structure_mask.sum()
        )
        per_structure_diffs.append(d95_diff + mean_dose_diff)
    return float(np.mean(per_structure_diffs)) if per_structure_diffs else 0.0
