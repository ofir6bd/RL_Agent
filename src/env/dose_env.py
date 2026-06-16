"""Treatment-planning environment.

One PPO **episode == one fraction** (``done=True`` after every ``step``).
A patient is played for ``n_fractions`` (35) consecutive fractions
without calling :meth:`reset` in between, so the cumulative dose, the
fraction counter and the Markovian state (``cumulative_dose``, remaining
gap, ``fraction_index/35``) all carry over.  The training loop watches
``info["patient_done"]`` to know when to reset to the next patient and
trigger a PPO update.

state  : (C, G, G, G) float32  +  scalar fraction_index/35
action : (n_beams, beamlet_h, beamlet_w) >= 0  (intensity weights)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

from ..config import Config, ALL_STRUCTURES, PTV_NAMES, OAR_NAMES
from ..data.preprocess import load_processed
from . import reward as reward_module


class DoseEnv:
    def __init__(self, cfg: Config,
                 patient_ids: Optional[list[str]] = None,
                 split: Optional[str] = None):
        self.cfg = cfg
        self.split = split
        root = Path(cfg.processed_dir)
        if split is not None:
            split_root = root / split
            if not split_root.is_dir():
                raise FileNotFoundError(
                    f"Processed split folder not found: {split_root}. "
                    f"Run scripts/preprocess.py with --splits {split} first."
                )
            root = split_root
        if patient_ids is None:
            patient_ids = sorted([p.name for p in root.iterdir() if p.is_dir()])
        self.patient_ids = patient_ids
        self.root = root

        # state filled during reset()
        self._patient: Optional[str] = None
        self._data: Dict[str, np.ndarray] = {}
        # (V, n_beamlets) sparse CSR (or legacy dense ndarray) loaded per
        # patient by ``reset()`` from ``dose_influence_matrix.npz``.
        self._dose_influence_matrix: Optional[
            Union[sp.csr_matrix, np.ndarray]
        ] = None
        # (G, G, G) running cumulative dose in Gy across all fractions of the
        # current patient course; persists across step() calls and is reset to
        # zeros at the start of each patient.
        self.cumulative_dose: Optional[np.ndarray] = None
        # Index of the *next* fraction to deliver (0-based).  Incremented at
        # the end of each step(); ``self.fraction_index == cfg.n_fractions``
        # marks the end of the patient course.
        self.fraction_index: int = 0

        # Effective reward weights for the *next* step().  Initialised from
        # the config and exposed as instance attributes so the training
        # loop can implement an OAR-penalty curriculum (e.g. ramp
        # ``lambda_oar`` from a fraction of its target value up to the
        # full target over the first N episodes) without having to mutate
        # the shared ``cfg`` object.
        self.lambda_oar: float = float(cfg.lambda_oar)
        self.lambda_ptv: float = float(cfg.lambda_ptv)

    # ------------------------------------------------------------------ utils
    def _structure_masks(self) -> Dict[str, np.ndarray]:
        """``{structure_name: (G, G, G) binary mask}`` for all loaded structures."""
        return {n: self._data["masks"][i]
                for i, n in enumerate(self._data["structure_names"].tolist())}

    def _ptv_masks(self) -> Dict[str, np.ndarray]:
        all_masks = self._structure_masks()
        return {k: all_masks[k] for k in PTV_NAMES if k in all_masks}

    def _oar_masks(self) -> Dict[str, np.ndarray]:
        all_masks = self._structure_masks()
        return {k: all_masks[k] for k in OAR_NAMES if k in all_masks}

    def _ptv_prescription_volume(self) -> np.ndarray:
        """Per-voxel prescribed dose over the union of PTVs (0 elsewhere)."""
        prescription_volume = np.zeros_like(self._data["ct"], dtype=np.float32)
        for ptv_name, prescription_gy in self.cfg.prescription.items():
            ptv_mask = self._structure_masks().get(ptv_name)
            if ptv_mask is not None:
                prescription_volume = np.maximum(
                    prescription_volume, ptv_mask * prescription_gy
                )
        return prescription_volume

    # ------------------------------------------------------------------ API
    def reset(self, patient_id: Optional[str] = None) -> Tuple[np.ndarray, float]:
        if patient_id is None:
            patient_id = np.random.choice(self.patient_ids)
        self._patient = patient_id
        self._data = load_processed(self.root / patient_id)

        pt_dir = self.root / patient_id
        # Prefer the new descriptive filename; fall back to the legacy
        # ``dim.npz`` / ``dim.npy`` so already-precomputed patient folders
        # still load without re-running compute_dose_influence_matrix.
        dose_influence_matrix_npz = pt_dir / "dose_influence_matrix.npz"
        legacy_dim_npz = pt_dir / "dim.npz"
        legacy_dim_npy = pt_dir / "dim.npy"
        if dose_influence_matrix_npz.exists():
            self._dose_influence_matrix = (
                sp.load_npz(dose_influence_matrix_npz)
                  .astype(np.float32).tocsr()
            )                                                            # (V, B)
        elif legacy_dim_npz.exists():
            self._dose_influence_matrix = (
                sp.load_npz(legacy_dim_npz).astype(np.float32).tocsr()
            )                                                            # (V, B)
        elif legacy_dim_npy.exists():
            self._dose_influence_matrix = (
                np.load(legacy_dim_npy).astype(np.float32)
            )                                                            # legacy dense
        else:
            raise FileNotFoundError(
                f"Missing dose-influence matrix for {patient_id} "
                f"(expected {dose_influence_matrix_npz}). "
                f"Run scripts/compute_dose_influence_matrix.py first."
            )

        self.cumulative_dose = np.zeros((self.cfg.grid,) * 3, dtype=np.float32)
        self.fraction_index = 0
        return (self._build_state(),
                float(self.fraction_index) / self.cfg.n_fractions)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, dict]:
        """Deliver one fraction.

        Each call is its own PPO episode: ``done`` is always ``True``.
        ``info["patient_done"]`` indicates whether the *patient course*
        (all ``n_fractions`` fractions) is finished — the training loop
        uses that flag to decide when to ``reset`` to the next patient
        and run a PPO update on the buffered 35 transitions.

        action: (n_beams, H, W) non-negative intensities.
        """
        assert self._dose_influence_matrix is not None and self.cumulative_dose is not None

        action = np.asarray(action, dtype=np.float32).reshape(-1)        # (B,)
        # beamlet intensities -> Gy delivered this fraction (works for both
        # scipy.sparse CSR and dense numpy ``self._dose_influence_matrix``)
        dose_per_voxel_flat = np.asarray(
            self._dose_influence_matrix @ action, dtype=np.float32
        ).ravel()                                                        # (V,)
        fraction_dose = dose_per_voxel_flat.reshape((self.cfg.grid,) * 3)
        fraction_dose *= self._data["possible_dose_mask"]

        # Snapshot the cumulative dose *before* adding this fraction;
        # needed for the adaptive per-fraction PTV target.
        cumulative_dose_before = self.cumulative_dose.copy()
        self.cumulative_dose = self.cumulative_dose + fraction_dose

        # Per-voxel PTV prescription map (union of PTVs, 0 elsewhere).
        prescription_volume = self._ptv_prescription_volume()

        # ---- reward ----
        # OAR penalty is now evaluated per-fraction (symmetric with the
        # per-fraction PTV reward): each OAR has a fraction-share limit
        # ``tolerance_total / n_fractions`` and we penalise this fraction's
        # dose exceeding it.  If every fraction stays under the per-fraction
        # limit, the cumulative dose is guaranteed to stay under the
        # full-course tolerance.
        per_fraction_oar_tolerance = {
            organ_name: full_course_tol / self.cfg.n_fractions
            for organ_name, full_course_tol in self.cfg.oar_tolerance.items()
        }
        oar_penalty_value = reward_module.oar_penalty(
            fraction_dose,
            self._oar_masks(),
            per_fraction_oar_tolerance,
            organ_weights=self.cfg.oar_weights,
            organ_is_serial=self.cfg.oar_serial,
            voxel_subweight=self.cfg.oar_voxel_subweight,
            mean_subweight=self.cfg.oar_mean_subweight,
            dmax_subweight=self.cfg.oar_dmax_subweight,
        )

        # Per-fraction PTV reward: hit your fair share of the remaining
        # gap (adaptive target =
        #     max(prescription - cumulative_dose_before, 0) / fractions_remaining
        # ).
        fractions_remaining = self.cfg.n_fractions - self.fraction_index
        ptv_reward_value = reward_module.fractional_ptv_reward(
            fraction_dose,
            prescription_volume,
            cumulative_dose_before,
            fractions_remaining,
        )

        self.fraction_index += 1
        patient_done = self.fraction_index >= self.cfg.n_fractions
        # Each fraction is its own PPO episode (1-step horizon).
        done = True

        # Reward weights are read from the *instance* attributes so the
        # training loop can drive a curriculum (see
        # ``train._effective_lambda_oar``).  They default to ``cfg.lambda_*``
        # when nothing is mutated externally, preserving previous behaviour.
        reward = (-self.lambda_oar * oar_penalty_value
                  + self.lambda_ptv * ptv_reward_value)

        info = {
            "oar_penalty":    oar_penalty_value,
            "ptv_reward":     ptv_reward_value,
            "patient":        self._patient,
            "fraction_index": self.fraction_index,
            "patient_done":   patient_done,
            "lambda_oar":     float(self.lambda_oar),
            "lambda_ptv":     float(self.lambda_ptv),
        }
        next_fraction_progress = float(self.fraction_index) / self.cfg.n_fractions
        return (self._build_state(),
                next_fraction_progress,
                float(reward),
                done,
                info)

    # --------------------------------------------------------------- state
    def _build_state(self) -> np.ndarray:
        """Return (C, G, G, G) float32.

        ch 0       : CT (already normalised)
        ch 1..S    : structure masks
        ch S+1     : cumulative_dose / 70
        ch S+2     : per-voxel dose gap (prescription - cumulative_dose),
                     clipped >= 0, /70
        ch S+3     : beam paths
        """
        ct = self._data["ct"][None]                                     # (1,G,G,G)
        structure_masks = self._data["masks"]                            # (S,G,G,G)
        cumulative_dose_channel = (self.cumulative_dose / 70.0)[None]
        ptv_dose_gap_channel = (
            np.clip(self._ptv_prescription_volume() - self.cumulative_dose,
                    0.0, None)[None]
            / 70.0
        )
        beam_paths = self._data["beam_paths"][None]
        state = np.concatenate(
            [ct, structure_masks, cumulative_dose_channel,
             ptv_dose_gap_channel, beam_paths],
            axis=0,
        )
        return state.astype(np.float32)
