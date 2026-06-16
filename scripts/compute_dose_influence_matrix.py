"""Compute the Dose Influence Matrix per patient using pyRadPlan and
save to data/processed/<pt>/dose_influence_matrix.npz as a
(V, n_beamlets) float32 CSR sparse matrix.

For each patient we:
  1. Build a pyRadPlan CT + StructureSet from the raw OpenKBP CSVs (native
     resolution, typically 94 x 128 x 128).
  2. Run a photon IMRT plan (n_beams gantry angles, SVDPB pencil beam).
  3. Aggregate native bixels into a (beamlet_h x beamlet_w) fluence grid per
     beam, by binning ray BEV (x, z) positions. Yields
        (V_native, n_beams * beamlet_h * beamlet_w) sparse matrix.
  4. Down-sample rows from the native CT grid to cfg.grid^3 via mean-pooling
     (single sparse matmul). Yields the final
        (cfg.grid^3, n_beams * beamlet_h * beamlet_w) float32 array.

The action vector in DoseEnv has shape (n_beams, beamlet_h, beamlet_w) which
is reshaped to (-1,) before ``dose_influence_matrix @ action``, so the column
ordering here is

    col(beam=b, h=h, w=w) = b * (beamlet_h*beamlet_w) + h * beamlet_w + w

Diagnostics
-----------
For every patient, this script also writes a log file and six diagnostic PNGs
into ``data/processed/<pt>/charts/``:

  * ``dose_influence_matrix.log``    summary stats (per beam, per structure)
  * ``1_patient_overview.png``        CT slices + structure contours
  * ``2_beam_geometry.png``           polar plot of gantry angles
  * ``3_per_beam_dose.png``           per-beam MIP dose footprints
  * ``4_beam_statistics.png``         max/mean dose & non-zero voxel counts
  * ``5_dose_distribution.png``       uniform-action dose overlaid on CT
  * ``6_dvh.png``                     DVH for all structures

Pass ``--no-charts`` to skip the visualisation step.

Usage
-----
python scripts/compute_dose_influence_matrix.py --processed data/processed
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import SimpleITK as sitk
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_config, Config, ALL_STRUCTURES, PTV_NAMES


# OpenKBP CSV name -> (pyRadPlan VOI name, VOI type)
_VOI_MAP: Dict[str, Tuple[str, str]] = {
    "PTV70":        ("PTV_70",        "TARGET"),
    "PTV63":        ("PTV_63",        "TARGET"),
    "PTV56":        ("PTV_56",        "TARGET"),
    "Brainstem":    ("brainstem",     "OAR"),
    "SpinalCord":   ("spinal_cord",   "OAR"),
    "Mandible":     ("mandible",      "OAR"),
    "LeftParotid":  ("left_parotid",  "OAR"),
    "RightParotid": ("right_parotid", "OAR"),
}


def _read_sparse_csv(path: Path, shape_flat: int) -> np.ndarray | None:
    """Read an OpenKBP sparse-CSV and return the (flat) indices it covers."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    idx = df.iloc[:, 0].to_numpy(dtype=np.int64)
    return idx[idx < shape_flat]


def _read_ct_csv(path: Path, shape_flat: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    idx = df.iloc[:, 0].to_numpy(dtype=np.int64)
    val = df.iloc[:, 1].to_numpy(dtype=np.float32)
    keep = idx < shape_flat
    return idx[keep], val[keep]


def _build_ct_cst(raw_pt: Path, n_x: int, n_y: int):
    """Build pyRadPlan CT + StructureSet from raw OpenKBP CSVs.

    Returns
    -------
    ct_obj, cst_obj, ct_sitk, shape_zyx, ct_arr, masks
        ``ct_arr``  : (NZ, NY, NX) float32 HU array used for plots
        ``masks``   : Dict[str, np.ndarray] keyed by raw CSV name
                      (e.g. ``"PTV70"``, ``"Brainstem"``), each (NZ, NY, NX)
                      uint8 at native resolution.  ``"BODY"`` is included.
    """
    from pyRadPlan.ct import CT
    from pyRadPlan.cst import StructureSet, validate_voi

    spacing = np.loadtxt(raw_pt / "voxel_dimensions.csv").astype(np.float64)
    resolution_xyz_mm = {"x": float(spacing[0]),
                         "y": float(spacing[1]),
                         "z": float(spacing[2])}

    # Determine NZ from largest flat index across CT and possible_dose_mask
    ct_dataframe = pd.read_csv(raw_pt / "ct.csv")
    possible_dose_mask_dataframe = pd.read_csv(
        raw_pt / "possible_dose_mask.csv"
    )
    max_flat_index = max(
        int(ct_dataframe.iloc[:, 0].max()),
        int(possible_dose_mask_dataframe.iloc[:, 0].max()),
    )
    n_z_slices = max_flat_index // (n_x * n_y) + 1
    shape_zyx = (n_z_slices, n_y, n_x)
    shape_flat = n_z_slices * n_y * n_x

    # OpenKBP CSVs store CT as HU + 1000; missing voxels (air) are 0 in that
    # stored scale. We rebuild the dense volume in stored units, then convert
    # to true HU (-1000 = air, 0 = water, +1000 ≈ bone) before pyRadPlan.
    ct_flat_indices, ct_voxel_values = _read_ct_csv(
        raw_pt / "ct.csv", shape_flat,
    )
    ct_arr = np.zeros(shape_zyx, dtype=np.float32)
    ct_arr.flat[ct_flat_indices] = ct_voxel_values
    ct_arr -= 1000.0

    ct_sitk = sitk.GetImageFromArray(ct_arr)
    ct_sitk.SetSpacing([resolution_xyz_mm["x"],
                        resolution_xyz_mm["y"],
                        resolution_xyz_mm["z"]])
    ct_sitk.SetOrigin([0.0, 0.0, 0.0])
    ct_obj = CT.model_validate(
        {"cube_hu": ct_sitk, "resolution": resolution_xyz_mm}
    )

    # ---- VOIs ----
    volume_of_interest_list = []
    masks: Dict[str, np.ndarray] = {}

    # External / body from possible_dose_mask
    possible_dose_mask_indices = (
        possible_dose_mask_dataframe.iloc[:, 0].to_numpy(dtype=np.int64)
    )
    possible_dose_mask_indices = (
        possible_dose_mask_indices[possible_dose_mask_indices < shape_flat]
    )
    body_mask = np.zeros(shape_zyx, dtype=np.uint8)
    body_mask.flat[possible_dose_mask_indices] = 1
    masks["BODY"] = body_mask
    body_sitk = sitk.GetImageFromArray(body_mask)
    body_sitk.CopyInformation(ct_sitk)
    volume_of_interest_list.append(
        validate_voi(name="BODY", voi_type="EXTERNAL",
                     mask=body_sitk, ct_image=ct_obj)
    )

    # Targets and OARs
    n_targets = 0
    for csv_name, (voi_name, voi_type) in _VOI_MAP.items():
        voxel_flat_indices = _read_sparse_csv(
            raw_pt / f"{csv_name}.csv", shape_flat,
        )
        if voxel_flat_indices is None or voxel_flat_indices.size == 0:
            continue
        mask_volume = np.zeros(shape_zyx, dtype=np.uint8)
        mask_volume.flat[voxel_flat_indices] = 1
        masks[csv_name] = mask_volume
        mask_sitk_image = sitk.GetImageFromArray(mask_volume)
        mask_sitk_image.CopyInformation(ct_sitk)
        volume_of_interest_list.append(
            validate_voi(name=voi_name, voi_type=voi_type,
                         mask=mask_sitk_image, ct_image=ct_obj)
        )
        if voi_type == "TARGET":
            n_targets += 1

    if n_targets == 0:
        raise SystemExit(
            f"[error] {raw_pt} has no target (PTV) structures, cannot plan.")

    cst_obj = StructureSet(
        vois=volume_of_interest_list, ct_image=ct_obj,
    )
    return ct_obj, cst_obj, ct_sitk, shape_zyx, ct_arr, masks


def _bin_bixels_per_beam(stf, dij,
                         n_beams: int,
                         beamlet_h: int,
                         beamlet_w: int) -> np.ndarray:
    """Map each native bixel column to a (beam, h, w) bin index in
    [0, n_beams * beamlet_h * beamlet_w).

    Binning is per-beam: ray BEV (x, z) positions are min/max-normalised within
    that beam, then quantised to a (beamlet_h, beamlet_w) grid.
    """
    total_bixels = int(dij.total_num_of_bixels)
    bin_indices = np.full(total_bixels, -1, dtype=np.int64)

    bixel_beam_indices = np.asarray(dij.beam_num).ravel().astype(np.int64)
    bixel_ray_indices  = np.asarray(dij.ray_num).ravel().astype(np.int64)
    bins_per_beam = beamlet_h * beamlet_w

    for beam_index in range(n_beams):
        beam_column_mask = (bixel_beam_indices == beam_index)
        if not beam_column_mask.any():
            continue
        rays = stf.beams[beam_index].rays
        # BEV positions of every ray in this beam: (n_rays, 3) -> use (x, z)
        ray_xz = np.array(
            [[float(ray.ray_pos_bev[0]), float(ray.ray_pos_bev[2])]
             for ray in rays],
            dtype=np.float64,
        )
        # Per-bixel position (broadcast via ray index)
        bixel_xz = ray_xz[
            bixel_ray_indices[beam_column_mask]
        ]                                                  # (n_bixels_b, 2)

        min_xz = ray_xz.min(axis=0)
        max_xz = ray_xz.max(axis=0)
        xz_extent = np.maximum(max_xz - min_xz, 1e-6)
        normalised_xz = (bixel_xz - min_xz) / xz_extent    # in [0, 1]
        # x -> w (horizontal), z -> h (vertical)
        width_bin = np.clip(
            (normalised_xz[:, 0] * beamlet_w).astype(np.int64),
            0, beamlet_w - 1,
        )
        height_bin = np.clip(
            (normalised_xz[:, 1] * beamlet_h).astype(np.int64),
            0, beamlet_h - 1,
        )

        local_bin_index = height_bin * beamlet_w + width_bin
        bin_indices[beam_column_mask] = (
            beam_index * bins_per_beam + local_bin_index
        )

    if (bin_indices < 0).any():
        # Bixels not assigned to any of our beams shouldn't happen, but guard anyway.
        raise RuntimeError(
            "Some bixels could not be binned (unknown beam index)."
        )
    return bin_indices


def _downsample_matrix(src_dims_zyx: Tuple[int, int, int],
                       grid: int) -> sp.csr_matrix:
    """Build a (grid^3, V_src) row-stochastic resampling matrix that handles
    both *down*-sampling (N_src > grid) and *up*-sampling (N_src < grid).

    Two-pass construction:

      1. **Source-side mean pool.** Every source voxel s = (sz, sy, sx) is
         routed into target voxel t = (
            floor((sz + 0.5) * grid / NZ),
            floor((sy + 0.5) * grid / NY),
            floor((sx + 0.5) * grid / NX),
         ). This guarantees that all the dose computed by pyRadPlan on the
         native dose grid is preserved when N_src >= grid in every axis.

      2. **Target-side fallback.** If any axis has N_src < grid (the typical
         case for the SVDPB dose grid in z, which is often ~47 slices vs our
         target 64), the source-side pass leaves some target voxels with no
         contributors and they end up identically zero. For each such
         orphaned target voxel we add a single nearest-source contribution
         t -> floor((t + 0.5) * N_src / grid) so every target voxel ends up
         covered by *at least one* source row.

      3. **Row-normalise.** Divide each non-empty target row by the total
         weight of its contributors so the result is row-stochastic (mean
         pool, not sum), matching the original DIM semantics in DoseEnv.

    This fixes the historical bug where the integer-floor map
    ``floor(s * grid / N)`` produced ``N`` distinct target indices regardless
    of ``grid``, so any extra target voxels were silently zero. Empirically
    that left ~25 % of PTV voxels physically unreachable.
    """
    n_source_z, n_source_y, n_source_x = src_dims_zyx
    n_target_voxels = grid ** 3
    n_source_voxels = n_source_z * n_source_y * n_source_x

    # -- 1) source-side aggregation ------------------------------------------
    source_zi, source_yi, source_xi = np.indices(
        (n_source_z, n_source_y, n_source_x),
    )
    source_zi = source_zi.ravel()
    source_yi = source_yi.ravel()
    source_xi = source_xi.ravel()
    target_z = (
        ((source_zi + 0.5) * grid // n_source_z)
        .astype(np.int64).clip(0, grid - 1)
    )
    target_y = (
        ((source_yi + 0.5) * grid // n_source_y)
        .astype(np.int64).clip(0, grid - 1)
    )
    target_x = (
        ((source_xi + 0.5) * grid // n_source_x)
        .astype(np.int64).clip(0, grid - 1)
    )
    target_indices_src = (
        target_z * grid * grid + target_y * grid + target_x
    )
    source_indices_src = np.arange(n_source_voxels, dtype=np.int64)

    rows_list = [target_indices_src]
    cols_list = [source_indices_src]
    vals_list = [np.ones(n_source_voxels, dtype=np.float32)]

    # -- 2) target-side fallback for orphaned target voxels ------------------
    target_covered = np.zeros(n_target_voxels, dtype=bool)
    target_covered[target_indices_src] = True
    if not target_covered.all():
        orphan_target_indices = np.nonzero(~target_covered)[0]
        orphan_tz, remainder_zy = np.divmod(
            orphan_target_indices, grid * grid,
        )
        orphan_ty, orphan_tx = np.divmod(remainder_zy, grid)
        fallback_sz = (
            ((orphan_tz + 0.5) * n_source_z // grid)
            .astype(np.int64).clip(0, n_source_z - 1)
        )
        fallback_sy = (
            ((orphan_ty + 0.5) * n_source_y // grid)
            .astype(np.int64).clip(0, n_source_y - 1)
        )
        fallback_sx = (
            ((orphan_tx + 0.5) * n_source_x // grid)
            .astype(np.int64).clip(0, n_source_x - 1)
        )
        fallback_source_indices = (
            (fallback_sz * n_source_y + fallback_sy) * n_source_x
            + fallback_sx
        )
        rows_list.append(orphan_target_indices.astype(np.int64))
        cols_list.append(fallback_source_indices.astype(np.int64))
        vals_list.append(
            np.ones(orphan_target_indices.size, dtype=np.float32),
        )

    rows_all = np.concatenate(rows_list)
    cols_all = np.concatenate(cols_list)
    vals_all = np.concatenate(vals_list)

    # -- 3) build sparse, then row-normalise to a mean pool ------------------
    pool_matrix = sp.coo_matrix(
        (vals_all, (rows_all, cols_all)),
        shape=(n_target_voxels, n_source_voxels),
    ).tocsr()
    row_totals = np.asarray(pool_matrix.sum(axis=1)).ravel()
    row_totals = np.maximum(row_totals, 1.0)
    inverse_row_totals = sp.diags(
        (1.0 / row_totals).astype(np.float32),
    )
    return (
        (inverse_row_totals @ pool_matrix).tocsr().astype(np.float32)
    )


def _per_beam_dose(sparse_dose_influence_matrix: sp.csr_matrix,
                   bixel_beam_indices: np.ndarray,
                   n_beams: int) -> np.ndarray:
    """Sum bixel contributions per beam -> (V_dose, n_beams) dense float32."""
    n_dose_voxels = sparse_dose_influence_matrix.shape[0]
    per_beam_dose = np.zeros((n_dose_voxels, n_beams), dtype=np.float32)
    for beam_index in range(n_beams):
        beam_columns = np.where(bixel_beam_indices == beam_index)[0]
        if beam_columns.size == 0:
            continue
        per_beam_dose[:, beam_index] = np.asarray(
            sparse_dose_influence_matrix[:, beam_columns].sum(axis=1),
        ).ravel()
    return per_beam_dose


def _dose_to_ct_grid(dose_zyx: np.ndarray,
                     dij,
                     ct_sitk: sitk.Image) -> np.ndarray:
    """Resample a dose volume living on ``dij.dose_grid`` onto the CT grid.

    Returns a (NZ, NY, NX) float32 array aligned with ``ct_sitk``.
    """
    dose_sitk = sitk.GetImageFromArray(dose_zyx.astype(np.float32))
    # dose_grid fields may not always be present — guard each one.
    dose_grid = dij.dose_grid
    if hasattr(dose_grid, "origin"):
        dose_sitk.SetOrigin(
            tuple(float(origin_axis) for origin_axis in dose_grid.origin)
        )
    else:
        dose_sitk.SetOrigin(ct_sitk.GetOrigin())
    if hasattr(dose_grid, "resolution_vector"):
        dose_sitk.SetSpacing(
            tuple(float(spacing_axis)
                  for spacing_axis in dose_grid.resolution_vector)
        )
    else:
        dose_sitk.SetSpacing(ct_sitk.GetSpacing())
    if hasattr(dose_grid, "direction"):
        dose_sitk.SetDirection(
            np.asarray(dose_grid.direction).ravel().tolist()
        )
    else:
        dose_sitk.SetDirection(ct_sitk.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(ct_sitk.GetOrigin())
    resampler.SetOutputSpacing(ct_sitk.GetSpacing())
    resampler.SetOutputDirection(ct_sitk.GetDirection())
    resampler.SetSize(ct_sitk.GetSize())
    return sitk.GetArrayFromImage(
        resampler.Execute(dose_sitk)
    ).astype(np.float32)


def _write_log(out_dir: Path,
               patient: str,
               cfg: Config,
               gantry_angles: list[float],
               shape_zyx: tuple[int, int, int],
               dose_dims_zyx: tuple[int, int, int],
               n_bixels_total: int,
               dose_influence_matrix: np.ndarray,
               per_beam_dose: np.ndarray,
               masks: Dict[str, np.ndarray],
               dose_on_ct: np.ndarray,
               elapsed_s: float) -> Path:
    """Write a human-readable summary at ``<out_dir>/dose_influence_matrix.log``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "dose_influence_matrix.log"
    lines: list[str] = []
    lines.append(f"# Dose-influence-matrix diagnostics — patient {patient}")
    lines.append(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# elapsed:   {elapsed_s:.1f} s")
    lines.append("")
    lines.append("[setup]")
    lines.append(f"native CT shape (z,y,x) = {shape_zyx}")
    lines.append(f"dose-grid shape (z,y,x) = {dose_dims_zyx}")
    lines.append(f"target grid (cfg.grid^3) = {cfg.grid}^3 = {cfg.grid**3}")
    lines.append(f"gantry angles  = {gantry_angles}")
    lines.append(f"prescription   = {cfg.prescription}")
    lines.append(f"n_bixels (raw) = {n_bixels_total}")
    lines.append(f"beamlet grid   = {cfg.n_beams} x {cfg.beamlet_h} x "
                 f"{cfg.beamlet_w} = {cfg.n_beamlets}")
    lines.append("")
    lines.append("[dose_influence_matrix.npz]")
    lines.append(f"shape    = {tuple(dose_influence_matrix.shape)}")
    lines.append(f"dtype    = {dose_influence_matrix.dtype}")
    if sp.issparse(dose_influence_matrix):
        n_bytes = int(
            dose_influence_matrix.data.nbytes
            + dose_influence_matrix.indices.nbytes
            + dose_influence_matrix.indptr.nbytes
        )
        n_nonzero_entries = int(dose_influence_matrix.nnz)
        n_total_entries = (int(dose_influence_matrix.shape[0])
                           * int(dose_influence_matrix.shape[1]))
        dose_min = (float(dose_influence_matrix.min())
                    if n_nonzero_entries else 0.0)
        dose_max = (float(dose_influence_matrix.max())
                    if n_nonzero_entries else 0.0)
        column_sums = np.asarray(
            dose_influence_matrix.sum(axis=0)
        ).ravel()
        row_sums = np.asarray(
            dose_influence_matrix.sum(axis=1)
        ).ravel()
    else:
        n_bytes = int(dose_influence_matrix.nbytes)
        n_nonzero_entries = int((dose_influence_matrix != 0).sum())
        n_total_entries = int(dose_influence_matrix.size)
        dose_min = float(dose_influence_matrix.min())
        dose_max = float(dose_influence_matrix.max())
        column_sums = dose_influence_matrix.sum(0)
        row_sums = dose_influence_matrix.sum(1)
    lines.append(f"size     = {n_bytes / 1e6:.1f} MB")
    lines.append(f"min/max  = {dose_min:.4f} / {dose_max:.4f}")
    lines.append(
        f"non-zero entries = {n_nonzero_entries:,} / {n_total_entries:,} "
        f"({100 * n_nonzero_entries / max(n_total_entries, 1):.2f}%)"
    )
    n_nonzero_columns = int((column_sums > 0).sum())
    n_nonzero_rows = int((row_sums > 0).sum())
    lines.append(
        f"non-zero cols    = {n_nonzero_columns} / "
        f"{dose_influence_matrix.shape[1]}"
    )
    lines.append(
        f"non-zero rows    = {n_nonzero_rows} / "
        f"{dose_influence_matrix.shape[0]}"
    )
    lines.append("")
    lines.append("[per-beam dose @ weight=1, native dose-grid]")
    lines.append(
        f"  {'beam':>4} {'angle':>8} {'max':>10} {'mean':>12} "
        f"{'nonzero':>10}"
    )
    for beam_index, gantry_angle in enumerate(gantry_angles):
        per_beam_dose_column = per_beam_dose[:, beam_index]
        lines.append(
            f"  {beam_index:>4} {gantry_angle:>7.1f}° "
            f"{per_beam_dose_column.max():>10.4f} "
            f"{per_beam_dose_column.mean():>12.6f} "
            f"{int((per_beam_dose_column > 0).sum()):>10,}"
        )
    lines.append("")
    lines.append("[structure mean dose @ uniform action × n_fractions]")
    prescriptions = cfg.prescription
    full_course_dose = dose_on_ct * cfg.n_fractions
    for structure_name, structure_mask in masks.items():
        if structure_name == "BODY" or not structure_mask.any():
            continue
        mean_dose_in_structure = float(
            (full_course_dose * structure_mask).sum() / structure_mask.sum()
        )
        prescription_tag = ""
        if structure_name in prescriptions:
            prescription_tag = (
                f"   (rx = {prescriptions[structure_name]} Gy)"
            )
        lines.append(
            f"  {structure_name:<14}  mean = "
            f"{mean_dose_in_structure:>7.2f} Gy{prescription_tag}"
        )
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def compute_dose_influence_matrix(cfg: Config, processed_pt: Path, *,
                                  charts: bool = True) -> np.ndarray:
    """Returns (cfg.grid**3, cfg.n_beamlets) float32 dose-influence matrix.

    When ``charts=True`` (default) also writes ``dose_influence_matrix.log``
    and six PNG diagnostics into ``<processed_pt>/charts/``.
    """
    try:
        from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence
    except ImportError as import_error:
        raise SystemExit(
            "[error] pyRadPlan is not installed in this environment.\n"
            "        Install it with:\n"
            "            pip install pyRadPlan\n"
            f"        (import error: {import_error})"
        )

    start_time = time.time()

    # Map data/processed/<split>/<pt>/  ->  data/original/<split>/<pt>/
    split = processed_pt.parent.name
    raw_pt = Path(cfg.raw_dir) / split / processed_pt.name
    if not raw_pt.is_dir():
        # Backwards compatibility: flat layout data/original/<pt>/
        flat_layout_raw_pt = Path(cfg.raw_dir) / processed_pt.name
        if flat_layout_raw_pt.is_dir():
            raw_pt = flat_layout_raw_pt
        else:
            raise SystemExit(
                f"[error] raw patient folder not found: {raw_pt}"
            )

    # ---- 1. CT + StructureSet ------------------------------------------------
    from src.config import NX, NY
    ct_obj, cst_obj, ct_sitk, shape_zyx, ct_arr, masks_native = \
        _build_ct_cst(raw_pt, NX, NY)

    # ---- 2. Photon IMRT plan -------------------------------------------------
    gantry_angles = [
        float(angle) for angle in np.linspace(
            0.0, 360.0, cfg.n_beams, endpoint=False,
        )
    ]
    max_prescription_dose = float(max(cfg.prescription.values()))
    photon_plan = PhotonPlan(
        num_of_fractions=cfg.n_fractions,
        prescribed_dose=max_prescription_dose,
        machine="Generic",
        prop_stf={
            "generator":     "photonIMRT",
            "gantry_angles": gantry_angles,
            "couch_angles":  [0.0] * cfg.n_beams,
            "bixel_width":   5.0,
        },
        prop_dose_calc={"engine": "SVDPB"},
    )

    # ---- 3. STF + dose-influence matrix -------------------------------------
    steering_information = generate_stf(ct_obj, cst_obj, photon_plan)
    dose_influence_object = calc_dose_influence(
        ct_obj, cst_obj, steering_information, photon_plan,
    )

    # Sparse (V_dose, total_bixels) in scipy.sparse format
    sparse_dose_influence_matrix = (
        dose_influence_object.physical_dose.flat[0]
    )
    if not sp.issparse(sparse_dose_influence_matrix):
        sparse_dose_influence_matrix = sp.csr_matrix(
            sparse_dose_influence_matrix,
        )
    sparse_dose_influence_matrix = sparse_dose_influence_matrix.tocsr()

    dose_dims_xyz = tuple(
        int(dim) for dim in dose_influence_object.dose_grid.dimensions
    )
    dose_dims_zyx = dose_dims_xyz[::-1]
    n_dose_voxels = int(np.prod(dose_dims_zyx))
    if sparse_dose_influence_matrix.shape[0] != n_dose_voxels:
        raise RuntimeError(
            f"Dose-influence-matrix rows "
            f"({sparse_dose_influence_matrix.shape[0]}) "
            f"!= dose-grid voxels ({n_dose_voxels})."
        )

    bixel_beam_indices = np.asarray(
        dose_influence_object.beam_num
    ).ravel().astype(np.int64)

    # ---- 4. Bixel -> (beam, h, w) bin aggregation ---------------------------
    bin_indices = _bin_bixels_per_beam(
        steering_information, dose_influence_object,
        cfg.n_beams, cfg.beamlet_h, cfg.beamlet_w,
    )
    n_output_columns = cfg.n_beamlets
    n_bixels = sparse_dose_influence_matrix.shape[1]
    # Selection matrix: (n_bixels, n_output_columns) with
    # entry[bixel_index, bin_indices[bixel_index]] = 1
    selection_matrix = sp.csr_matrix(
        (np.ones(n_bixels, dtype=np.float32),
         (np.arange(n_bixels, dtype=np.int64), bin_indices)),
        shape=(n_bixels, n_output_columns),
    )
    binned_dose_influence_matrix = (
        sparse_dose_influence_matrix @ selection_matrix
    ).tocsr()                                       # (V_dose, n_output_columns)

    # ---- 5. Downsample dose grid -> cfg.grid^3 ------------------------------
    downsample_matrix = _downsample_matrix(
        dose_dims_zyx, cfg.grid,
    )                                                       # (G^3, V_dose)
    downsampled_sparse_matrix = (
        downsample_matrix @ binned_dose_influence_matrix
    )                                                       # (G^3, n_output_columns)
    dose_influence_matrix_out = downsampled_sparse_matrix.tocsr().astype(
        np.float32,
    )                                                       # keep sparse

    # ---- 6. Quick sanity print ----------------------------------------------
    column_sums = np.asarray(
        dose_influence_matrix_out.sum(axis=0),
    ).ravel()
    n_nonzero_output_columns = int((column_sums > 0).sum())
    print(
        f"  [{processed_pt.name}] dose_influence_matrix "
        f"{dose_influence_matrix_out.shape}  "
        f"non-zero cols: {n_nonzero_output_columns}/{n_output_columns}  "
        f"max: {dose_influence_matrix_out.max():.4f} Gy/beamlet"
    )

    # ---- 7. Diagnostics (log + charts) --------------------------------------
    elapsed_seconds = time.time() - start_time
    if charts:
        chart_dir = processed_pt / "charts"
        # Aggregated per-beam dose volume on native dose grid (for charts 3/4)
        per_beam_dose = _per_beam_dose(
            sparse_dose_influence_matrix, bixel_beam_indices, cfg.n_beams,
        )

        # Uniform-action dose (sum across beams) resampled to CT grid for 5/6
        uniform_dose_flat = per_beam_dose.sum(axis=1)
        uniform_dose_volume = uniform_dose_flat.reshape(dose_dims_zyx)
        try:
            dose_on_ct = _dose_to_ct_grid(
                uniform_dose_volume, dose_influence_object, ct_sitk,
            )
        except Exception as resample_error:                # noqa: BLE001
            print(
                f"  [{processed_pt.name}] warning: dose resample failed "
                f"({type(resample_error).__name__}: {resample_error}); "
                f"using nearest-block upsample."
            )
            # Fallback: simple integer-broadcast upsample
            zoom_factors = (
                np.array(shape_zyx)
                / np.array(uniform_dose_volume.shape)
            )
            upsample_zi = (
                np.arange(shape_zyx[0]) / zoom_factors[0]
            ).astype(int).clip(0, uniform_dose_volume.shape[0] - 1)
            upsample_yi = (
                np.arange(shape_zyx[1]) / zoom_factors[1]
            ).astype(int).clip(0, uniform_dose_volume.shape[1] - 1)
            upsample_xi = (
                np.arange(shape_zyx[2]) / zoom_factors[2]
            ).astype(int).clip(0, uniform_dose_volume.shape[2] - 1)
            dose_on_ct = uniform_dose_volume[
                np.ix_(upsample_zi, upsample_yi, upsample_xi)
            ].astype(np.float32)

        # --- log ---
        log_path = _write_log(
            chart_dir, processed_pt.name, cfg, gantry_angles, shape_zyx,
            dose_dims_zyx, n_bixels, dose_influence_matrix_out,
            per_beam_dose, masks_native, dose_on_ct, elapsed_seconds,
        )
        print(f"  [{processed_pt.name}] wrote {log_path}")

        # --- charts ---
        try:
            from src.utils import visualize as visualization
            visualization.plot_patient_overview(
                ct_arr, masks_native, chart_dir,
                patient=processed_pt.name,
            )
            visualization.plot_beam_geometry(
                gantry_angles, chart_dir, patient=processed_pt.name,
            )
            visualization.plot_per_beam_dose(
                per_beam_dose, dose_dims_zyx, gantry_angles,
                chart_dir, patient=processed_pt.name,
            )
            visualization.plot_beam_statistics(
                per_beam_dose, gantry_angles, chart_dir,
                patient=processed_pt.name,
            )
            visualization.plot_dose_distribution(
                ct_arr, dose_on_ct, masks_native,
                chart_dir, patient=processed_pt.name,
            )
            visualization.plot_dvh(
                dose_on_ct, masks_native, cfg.prescription,
                chart_dir, patient=processed_pt.name,
                n_fractions=cfg.n_fractions,
            )
            print(
                f"  [{processed_pt.name}] wrote 6 charts -> {chart_dir}"
            )
        except Exception as chart_error:                # noqa: BLE001
            print(
                f"  [{processed_pt.name}] warning: chart generation "
                f"failed ({type(chart_error).__name__}: {chart_error}); "
                f"dose-influence matrix is still saved."
            )

    return dose_influence_matrix_out


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--processed", default="data/processed")
    arg_parser.add_argument("--config", default="configs/default.yaml")
    arg_parser.add_argument(
        "--splits", nargs="*",
        default=["train", "validation", "test"],
        help="split subfolders of --processed to process",
    )
    arg_parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing dose_influence_matrix.npz",
    )
    arg_parser.add_argument(
        "--no-charts", action="store_true",
        help="skip per-patient log & PNG diagnostics",
    )
    args = arg_parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed)

    # Collect patients from <root>/<split>/pt_* if any split exists, else fall
    # back to flat <root>/pt_* layout.
    patient_dirs: list[Path] = []
    for split in args.splits:
        split_dir = processed_root / split
        if split_dir.is_dir():
            patient_dirs.extend(
                sorted(p for p in split_dir.iterdir() if p.is_dir())
            )
    if not patient_dirs:
        patient_dirs = sorted(
            p for p in processed_root.iterdir() if p.is_dir()
        )

    for patient_dir in tqdm(patient_dirs, desc="dose_influence_matrix"):
        output_path = patient_dir / "dose_influence_matrix.npz"
        legacy_output_path = patient_dir / "dim.npz"
        if (output_path.exists() or legacy_output_path.exists()) \
                and not args.force:
            continue
        dose_influence_matrix = compute_dose_influence_matrix(
            cfg, patient_dir, charts=not args.no_charts,
        )
        sp.save_npz(output_path, dose_influence_matrix)


if __name__ == "__main__":
    main()
