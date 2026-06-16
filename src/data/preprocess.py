"""Resample raw patient volumes to a fixed cubic grid + cache as .npy.

Uses SimpleITK so resampling respects real-world (mm) spacing.
Also normalises CT to [-1, 1] and casts a simple ray-cast beam-path map.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import SimpleITK as sitk

from .loader import load_patient


def _to_sitk(volume_zyx: np.ndarray, spacing_xyz_mm) -> sitk.Image:
    """np (z,y,x) -> sitk image with spacing (x,y,z)."""
    sitk_image = sitk.GetImageFromArray(volume_zyx.astype(np.float32))
    sitk_image.SetSpacing([float(spacing_xyz_mm[0]),
                           float(spacing_xyz_mm[1]),
                           float(spacing_xyz_mm[2])])
    return sitk_image


def _resample(sitk_image: sitk.Image,
              target_size: int,
              interpolator=sitk.sitkLinear) -> np.ndarray:
    """Resample to target_size^3 voxels covering the same physical extent."""
    source_spacing_xyz = np.array(
        sitk_image.GetSpacing(), dtype=np.float64,
    )                                                                  # x,y,z
    source_size_xyz = np.array(
        sitk_image.GetSize(), dtype=np.float64,
    )                                                                  # x,y,z
    physical_extent_mm = source_spacing_xyz * source_size_xyz          # x,y,z
    new_spacing = (physical_extent_mm / target_size).tolist()
    new_size = [target_size, target_size, target_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampled_image = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(resampled_image).astype(np.float32)  # (z,y,x)


def normalise_ct(ct: np.ndarray) -> np.ndarray:
    """OpenKBP-stored CT (HU + 1000) -> [-1, 1].

    OpenKBP CSVs store CT as HU + 1000 (so air ≈ 0, water ≈ 1000,
    bone ≈ 2000) and the loader fills missing voxels with 0 (air). We
    subtract 1000 to recover true HU, then clip to the standard
    ``[-1000, 1000]`` window and divide by 1000.
    """
    ct = ct.astype(np.float32) - 1000.0
    ct = np.clip(ct, -1000.0, 1000.0)
    return ct / 1000.0


def ray_cast_beam_paths(grid: int, n_beams: int = 9) -> np.ndarray:
    """Very simple coplanar beam-path map.

    For each of n_beams gantry angles spaced 360/n_beams apart, draw a fan of
    rays through the iso-centre and mark traversed voxels. Returns a single
    (grid, grid, grid) float32 volume in [0, 1].
    """
    beam_path_volume = np.zeros((grid, grid, grid), dtype=np.float32)
    iso_centre = (grid - 1) / 2.0
    for beam_index in range(n_beams):
        gantry_angle_rad = 2.0 * np.pi * beam_index / n_beams
        unit_dx, unit_dy = np.cos(gantry_angle_rad), np.sin(gantry_angle_rad)
        for path_offset in np.linspace(-grid, grid, grid * 2):
            voxel_x = iso_centre + path_offset * unit_dx
            voxel_y = iso_centre + path_offset * unit_dy
            voxel_xi = int(round(voxel_x))
            voxel_yi = int(round(voxel_y))
            if 0 <= voxel_xi < grid and 0 <= voxel_yi < grid:
                # beam runs along z (axial slices)
                beam_path_volume[:, voxel_yi, voxel_xi] = 1.0
    # normalise so multiple overlapping beams just stay at 1
    return np.clip(beam_path_volume, 0.0, 1.0)


def preprocess_patient(raw_dir: str | Path,
                       out_dir: str | Path,
                       grid: int = 64,
                       n_beams: int = 9) -> Dict[str, np.ndarray]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_patient = load_patient(raw_dir)
    spacing_x, spacing_y, spacing_z = raw_patient["voxel_dims"]

    ct_image  = _to_sitk(raw_patient["ct"],
                         (spacing_x, spacing_y, spacing_z))
    dose_image = _to_sitk(raw_patient["dose_gt"],
                          (spacing_x, spacing_y, spacing_z))
    possible_dose_mask_image = _to_sitk(
        raw_patient["possible_dose_mask"],
        (spacing_x, spacing_y, spacing_z),
    )

    ct = _resample(ct_image, grid, sitk.sitkLinear)
    ct = normalise_ct(ct)
    dose_gt = _resample(dose_image, grid, sitk.sitkLinear)
    possible_dose_mask = (
        _resample(possible_dose_mask_image, grid, sitk.sitkNearestNeighbor)
        > 0.5
    ).astype(np.float32)

    structure_masks_resampled = []
    for structure_index in range(raw_patient["masks"].shape[0]):
        mask_image = _to_sitk(
            raw_patient["masks"][structure_index],
            (spacing_x, spacing_y, spacing_z),
        )
        resampled_mask = _resample(
            mask_image, grid, sitk.sitkNearestNeighbor,
        )
        structure_masks_resampled.append(
            (resampled_mask > 0.5).astype(np.float32)
        )
    structure_masks_resampled = np.stack(structure_masks_resampled, axis=0)

    beam_paths = ray_cast_beam_paths(grid, n_beams)

    cached_arrays = {
        "ct":                 ct,
        "masks":              structure_masks_resampled,
        "dose_gt":            dose_gt,
        "possible_dose_mask": possible_dose_mask,
        "beam_paths":         beam_paths,
        "voxel_dims":         raw_patient["voxel_dims"],
        "structure_names":    raw_patient["structure_names"],
    }
    for array_name, array_value in cached_arrays.items():
        np.save(out_dir / f"{array_name}.npy", array_value)
    return cached_arrays


def load_processed(patient_dir: str | Path) -> Dict[str, np.ndarray]:
    patient_dir = Path(patient_dir)
    return {
        npy_file.stem: np.load(npy_file, allow_pickle=True)
        for npy_file in patient_dir.glob("*.npy")
    }
