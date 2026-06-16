"""Reward terms used by the per-fraction PPO environment.

  total = - lambda_oar * oar_penalty(fraction_dose, oar_tolerance / n, ...)
        + lambda_ptv * fractional_ptv_reward(fraction_dose,
                                             prescription_volume,
                                             cumulative_dose_before,
                                             fractions_remaining, ...)

Each fraction is its own PPO episode (``done=True`` after every step).
Both terms are evaluated on **this fraction's** dose against a
per-fraction target/limit so the signal is symmetric and dense from
fraction 1 onwards:

* PTV target per voxel  = max(prescription - cumulative_dose_before, 0)
                          / fractions_remaining
* OAR limit per organ    = oar_tolerance_total / n_fractions

If every fraction stays under its OAR limit, the cumulative dose is
guaranteed to stay under the full-course tolerance.

Legacy helpers (``progress_shaping``, ``terminal_bonus``, ``dvh_bonus``,
``coverage``) are kept for evaluation / metrics utilities but are no longer
used by ``DoseEnv``.

OAR penalty internal mix
------------------------
For every OAR with a non-empty mask we combine three terms, all expressed as
an "excess fraction of tolerance"::

    excess = max(0, dose - tolerance) / tolerance

and passed through a piecewise function::

    piecewise(excess) = excess           if excess <= 0.1
                       = excess ** 2     otherwise

so small overshoots are linear (mild gradient) and big ones grow
super-linearly.

  per_voxel_penalty    = mean over OAR voxels of piecewise(voxel_excess)
  mean_dose_penalty    = piecewise(excess of organ-mean dose)
  max_dose_penalty     = piecewise(excess of organ-max  dose)  if serial else 0

The three terms are mixed with sub-weights (``oar_voxel_subweight``,
``oar_mean_subweight``, ``oar_dmax_subweight``) and then scaled by an
optional per-OAR weight from ``cfg.oar_weights``.  The sum is returned, and
``DoseEnv`` multiplies by ``-cfg.lambda_oar``.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np


# Hard cap on per-voxel "excess fraction of tolerance".  Without it a single
# wildly over-dosed voxel (e.g. 100x tolerance during early random rollouts)
# contributes ~1e4 through the squared branch and the reward explodes to
# values the critic cannot fit, which manifests as NaNs in the actor head.
_EXCESS_FRACTION_CLIP = 5.0


def _piecewise_excess_penalty(
    excess_fraction: np.ndarray | float,
) -> np.ndarray | float:
    """Linear up to 0.1, squared above, saturated at ``_EXCESS_FRACTION_CLIP``.

    Works on scalars or arrays. Saturating keeps the penalty bounded so the
    value target stays in a numerically tractable range even for very bad
    early-policy rollouts.
    """
    if isinstance(excess_fraction, np.ndarray):
        excess_clipped = np.minimum(excess_fraction, _EXCESS_FRACTION_CLIP)
        return np.where(excess_clipped <= 0.1,
                        excess_clipped,
                        excess_clipped * excess_clipped)
    excess_clipped = (excess_fraction
                      if excess_fraction <= _EXCESS_FRACTION_CLIP
                      else _EXCESS_FRACTION_CLIP)
    return (excess_clipped
            if excess_clipped <= 0.1
            else excess_clipped * excess_clipped)


def oar_penalty(dose_volume: np.ndarray,
                organ_masks: Dict[str, np.ndarray],
                organ_tolerances: Dict[str, float],
                *,
                organ_weights:    Optional[Dict[str, float]] = None,
                organ_is_serial:  Optional[Dict[str, bool]]  = None,
                voxel_subweight:  float = 1.0,
                mean_subweight:   float = 0.2,
                dmax_subweight:   float = 1.0) -> float:
    """Sum of per-OAR penalties — see module docstring for the formula.

    Parameters
    ----------
    dose_volume
        (G, G, G) dose in Gy that should be checked against the supplied
        tolerances.  In the active per-fraction environment this is the
        *fractional* dose and ``organ_tolerances`` are the *per-fraction*
        limits (``tolerance_total / n_fractions``); in legacy / metrics
        contexts it may be the cumulative dose with full-course tolerances.
    organ_masks
        ``{organ_name: binary mask}`` (only OARs with a tolerance are used).
    organ_tolerances
        ``{organ_name: tolerance_Gy}``.
    organ_weights
        Optional per-OAR multiplier (default 1.0 each).
    organ_is_serial
        Optional ``{organ_name: bool}`` flag. ``True`` adds a Dmax term for
        that organ (recommended for serial OARs such as spinal cord and
        brainstem where a single hot voxel is catastrophic).
    voxel_subweight, mean_subweight, dmax_subweight
        Sub-weights inside one OAR's penalty (per-voxel / mean / Dmax).
    """
    organ_weights   = organ_weights   or {}
    organ_is_serial = organ_is_serial or {}
    total_penalty   = 0.0

    for organ_name, tolerance in organ_tolerances.items():
        organ_mask = organ_masks.get(organ_name)
        if organ_mask is None or organ_mask.sum() == 0 or tolerance <= 0:
            continue
        dose_in_organ = dose_volume[organ_mask > 0]

        # --- per-voxel term (main signal) ---
        voxel_excess = np.maximum(0.0, (dose_in_organ - tolerance) / tolerance)
        if voxel_excess.size == 0 or not voxel_excess.any():
            per_voxel_penalty = 0.0
        else:
            per_voxel_penalty = float(_piecewise_excess_penalty(voxel_excess).mean())

        # --- organ-mean term (smooth stabiliser) ---
        mean_dose_excess = max(
            0.0, (float(dose_in_organ.mean()) - tolerance) / tolerance
        )
        mean_dose_penalty = float(_piecewise_excess_penalty(mean_dose_excess))

        # --- Dmax term, only for serial OARs ---
        max_dose_penalty = 0.0
        if organ_is_serial.get(organ_name, False) and dose_in_organ.size:
            max_dose_excess = max(
                0.0, (float(dose_in_organ.max()) - tolerance) / tolerance
            )
            max_dose_penalty = float(_piecewise_excess_penalty(max_dose_excess))

        organ_weight = float(organ_weights.get(organ_name, 1.0))
        total_penalty += organ_weight * (
            voxel_subweight * per_voxel_penalty
            + mean_subweight  * mean_dose_penalty
            + dmax_subweight  * max_dose_penalty
        )

    return float(total_penalty)


def progress_shaping(remaining_before: np.ndarray,
                     remaining_after: np.ndarray,
                     prescription_mean: float) -> float:
    """Mean Gy of gap closed across PTV voxels, normalised by mean prescription.

    .. note:: legacy — kept for reference / metrics.  The active environment
       now uses :func:`fractional_ptv_reward` instead.
    """
    if remaining_before.size == 0:
        return 0.0
    closed = float((remaining_before - remaining_after).mean())
    return closed / max(prescription_mean, 1e-6)


def fractional_ptv_reward(fraction_dose: np.ndarray,
                          prescription_volume: np.ndarray,
                          cumulative_dose_before: np.ndarray,
                          fractions_remaining: int,
                          *,
                          mean_error_clip: float = 2.0) -> float:
    """Per-fraction PTV reward in roughly ``[1 - mean_error_clip, 1]``.

    The "fair share" each PTV voxel should receive this fraction is::

        per_fraction_target_dose =
            max(prescription - cumulative_dose_before, 0) / fractions_remaining

    We score the closeness of this fraction's actual PTV dose to that
    target::

        relative_error  = |fraction_dose - per_fraction_target_dose|
                          / per_fraction_target_dose             (per voxel)
        reward          = 1 - clip(mean(relative_error), 0, mean_error_clip)

    Voxels already at prescription (``per_fraction_target_dose == 0``) are
    excluded — they contribute nothing here.  Adaptive targeting means an
    agent that under-dosed earlier fractions can compensate later without
    being permanently penalised.

    Parameters
    ----------
    fraction_dose
        Dose delivered by *this* fraction, shape ``(G, G, G)`` in Gy.
    prescription_volume
        Per-voxel prescription (union of PTVs, 0 elsewhere),
        shape ``(G, G, G)``.
    cumulative_dose_before
        Cumulative dose *before* this fraction was added,
        shape ``(G, G, G)``.
    fractions_remaining
        Number of fractions remaining *including the current one* (so this
        is ``n_fractions - fraction_index`` where ``fraction_index`` counts
        already-completed fractions).
    mean_error_clip
        Upper clip on the mean relative error before flipping sign.  With
        the default of ``2.0`` the reward lives in ``[-1, 1]``.
    """
    fractions_remaining = max(int(fractions_remaining), 1)
    remaining_prescription = np.clip(
        prescription_volume - cumulative_dose_before, 0.0, None
    )
    per_fraction_target_dose = remaining_prescription / float(fractions_remaining)
    active_voxels = per_fraction_target_dose > 0
    if not active_voxels.any():
        return 0.0
    relative_error = (
        np.abs(fraction_dose[active_voxels]
               - per_fraction_target_dose[active_voxels])
        / per_fraction_target_dose[active_voxels]
    )
    mean_relative_error = float(
        np.clip(relative_error.mean(), 0.0, mean_error_clip)
    )
    return 1.0 - mean_relative_error


def coverage(ptv_dose: np.ndarray, prescription: float) -> float:
    if ptv_dose.size == 0:
        return 0.0
    return float((ptv_dose >= prescription).mean())


def dvh_bonus(cum_dose: np.ndarray,
              ptv_masks: Dict[str, np.ndarray],
              prescriptions: Dict[str, float]) -> float:
    """Cheap proxy DVH bonus: average over PTVs of
        1 - mean(|dose - rx| / rx) clipped to [0, 1].
    Higher when dose tightly matches prescription."""
    bonuses = []
    for name, rx in prescriptions.items():
        m = ptv_masks.get(name)
        if m is None or m.sum() == 0:
            continue
        d = cum_dose[m > 0]
        err = np.mean(np.abs(d - rx) / rx)
        bonuses.append(max(0.0, 1.0 - float(err)))
    return float(np.mean(bonuses)) if bonuses else 0.0


def terminal_bonus(cum_dose: np.ndarray,
                   ptv_masks: Dict[str, np.ndarray],
                   prescriptions: Dict[str, float],
                   dvh_scale: float) -> float:
    """coverage  +  dvh_scale * dvh_bonus.

    .. note:: legacy — kept for evaluation / metrics only.  The active
       per-fraction environment no longer emits a terminal bonus.
    """
    covs = []
    for name, rx in prescriptions.items():
        m = ptv_masks.get(name)
        if m is None or m.sum() == 0:
            continue
        covs.append(coverage(cum_dose[m > 0], rx))
    cov = float(np.mean(covs)) if covs else 0.0
    return cov + dvh_scale * dvh_bonus(cum_dose, ptv_masks, prescriptions)
