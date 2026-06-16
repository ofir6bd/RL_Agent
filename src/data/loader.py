"""OpenKBP raw-CSV reader.

Each CSV stores only non-zero voxels as (flat_index, value) pairs.
Flat-index ordering is z -> y -> x with NX = NY = 128.
NZ is inferred from the maximum flat index.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..config import NX, NY, ALL_STRUCTURES


def _read_sparse_csv(path: Path) -> Optional[np.ndarray]:
    """Returns Nx2 array of (flat_index, value) or None if file is missing.

    Some OpenKBP CSVs (the binary masks and possible_dose_mask) only store the
    flat indices and leave the value column empty — that just means "this voxel
    is in the structure". We treat any missing value as 1.0.
    """
    if not path.exists():
        return None
    csv_dataframe = pd.read_csv(path)
    flat_indices = csv_dataframe.iloc[:, 0].to_numpy(dtype=np.int64)
    if csv_dataframe.shape[1] < 2:
        voxel_values = np.ones(flat_indices.shape[0], dtype=np.float32)
    else:
        voxel_values = csv_dataframe.iloc[:, 1].to_numpy(dtype=np.float64)
        # empty cells -> NaN -> treat as presence indicator (1.0)
        voxel_values = np.where(
            np.isnan(voxel_values), 1.0, voxel_values,
        ).astype(np.float32)
    return np.stack(
        [flat_indices.astype(np.float64),
         voxel_values.astype(np.float64)],
        axis=1,
    )


def infer_nz(max_flat_index: int) -> int:
    return int(max_flat_index // (NX * NY)) + 1


def _scatter(index_value_pairs: Optional[np.ndarray],
             shape: tuple) -> np.ndarray:
    """Place sparse pairs into a dense (NZ, NY, NX) volume; missing -> zeros."""
    dense_volume = np.zeros(shape, dtype=np.float32)
    if index_value_pairs is None or index_value_pairs.shape[0] == 0:
        return dense_volume
    flat_view = dense_volume.reshape(-1)
    flat_indices = index_value_pairs[:, 0].astype(np.int64)
    flat_view[flat_indices] = index_value_pairs[:, 1].astype(np.float32)
    return dense_volume


def load_patient(patient_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load all raw CSVs for one patient into dense 3-D arrays.

    Returns
    -------
    dict with keys:
      ct, dose_gt, possible_dose_mask          -> (NZ, NY, NX) float32
      masks                                     -> (S, NZ, NY, NX) float32 (binary)
      voxel_dims                                -> (3,) float32 (mm; x, y, z)
    """
    patient_dir = Path(patient_dir)

    # 1) voxel dimensions ------------------------------------------------------
    voxel_dimensions = np.loadtxt(
        patient_dir / "voxel_dimensions.csv"
    ).astype(np.float32)
    assert voxel_dimensions.shape == (3,), (
        f"voxel_dimensions must have 3 entries, got {voxel_dimensions.shape}"
    )

    # 2) read every sparse file once to find global max flat index -------------
    sparse_arrays: Dict[str, Optional[np.ndarray]] = {
        "ct":                 _read_sparse_csv(patient_dir / "ct.csv"),
        "dose":               _read_sparse_csv(patient_dir / "dose.csv"),
        "possible_dose_mask": _read_sparse_csv(
            patient_dir / "possible_dose_mask.csv",
        ),
    }
    for structure_name in ALL_STRUCTURES:
        sparse_arrays[structure_name] = _read_sparse_csv(
            patient_dir / f"{structure_name}.csv",
        )

    max_flat_index = 0
    for sparse_array in sparse_arrays.values():
        if sparse_array is not None and sparse_array.shape[0] > 0:
            max_flat_index = max(
                max_flat_index, int(sparse_array[:, 0].max()),
            )
    n_z_slices = infer_nz(max_flat_index)
    volume_shape = (n_z_slices, NY, NX)

    # 3) scatter to dense ------------------------------------------------------
    ct = _scatter(sparse_arrays["ct"], volume_shape)
    dose_gt = _scatter(sparse_arrays["dose"], volume_shape)
    possible_dose_mask = _scatter(
        sparse_arrays["possible_dose_mask"], volume_shape,
    )
    possible_dose_mask = (possible_dose_mask > 0).astype(np.float32)

    structure_masks = np.stack(
        [(_scatter(sparse_arrays[structure_name], volume_shape) > 0)
         .astype(np.float32)
         for structure_name in ALL_STRUCTURES],
        axis=0,
    )

    return {
        "ct": ct,
        "dose_gt": dose_gt,
        "possible_dose_mask": possible_dose_mask,
        "masks": structure_masks,
        "voxel_dims": voxel_dimensions,
        "structure_names": np.array(ALL_STRUCTURES),
    }
