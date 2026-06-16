"""Per-patient diagnostic charts for the dose-influence-matrix step.

All figures are written to disk as PNGs (no interactive windows) — the
Matplotlib backend is forced to ``Agg`` on import.

Adapted from Archive/test_pyRadPlan/visualize.py with two changes:
  * configurable output directory (one folder per patient)
  * no plt.show() — purely file-based
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")           # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Distinct colour per structure — extend as new VOIs are added
STRUCT_COLORS: Dict[str, str] = {
    "PTV70":        "red",
    "PTV63":        "orange",
    "PTV56":        "yellow",
    "Brainstem":    "cyan",
    "SpinalCord":   "lime",
    "Mandible":     "deepskyblue",
    "LeftParotid":  "violet",
    "RightParotid": "magenta",
    "BODY":         "white",
}


def _add_explanation(fig, lines: Iterable[str], y_start: float = 0.01) -> None:
    text = "\n".join(f"  * {line}" for line in lines)
    fig.text(
        0.01, y_start, text,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff",
                  edgecolor="#aabbdd", alpha=0.9),
        family="monospace",
    )


def _save(fig, path: Path, *, facecolor=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": 120, "bbox_inches": "tight"}
    if facecolor is not None:
        kwargs["facecolor"] = facecolor
    fig.savefig(path, **kwargs)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Patient overview: CT slices + structure contours
# ---------------------------------------------------------------------------
def plot_patient_overview(ct_array: np.ndarray,
                          masks: Dict[str, np.ndarray],
                          out_dir: Path,
                          patient: str = "") -> Path:
    nz, ny, nx = ct_array.shape
    iz, iy, ix = nz // 2, ny // 2, nx // 2

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Patient overview — {patient}", fontsize=14,
                 fontweight="bold", y=0.98)
    axes = [fig.add_axes([0.02 + i * 0.32, 0.22, 0.29, 0.70]) for i in range(3)]
    views = [
        (axes[0], ct_array[iz, :, :], f"Axial (z={iz})", "axial"),
        (axes[1], ct_array[:, iy, :], f"Coronal (y={iy})", "coronal"),
        (axes[2], ct_array[:, :, ix], f"Sagittal (x={ix})", "sagittal"),
    ]
    for ax, slc, title, kind in views:
        ax.imshow(slc, cmap="gray", vmin=-200, vmax=300,
                  origin="lower", aspect="auto")
        ax.set_title(title, fontsize=10); ax.axis("off")
        for name, m in masks.items():
            if name == "BODY":
                continue
            color = STRUCT_COLORS.get(name, "white")
            sl = (m[iz, :, :] if kind == "axial"
                  else m[:, iy, :] if kind == "coronal"
                  else m[:, :, ix])
            if sl.any():
                ax.contour(sl, levels=[0.5], colors=[color], linewidths=1.2)

    patches = [mpatches.Patch(color=STRUCT_COLORS.get(n, "white"), label=n)
               for n in masks if n != "BODY"]
    if patches:
        axes[2].legend(handles=patches, loc="upper right",
                       fontsize=7, framealpha=0.6)

    _add_explanation(fig, [
        "Axial / coronal / sagittal slices through the patient centre.",
        "Coloured contours = each structure (PTV / OAR) used in planning.",
        "If a contour is missing the corresponding CSV was not provided.",
    ])
    path = out_dir / "1_patient_overview.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 2. Beam geometry polar plot
# ---------------------------------------------------------------------------
def plot_beam_geometry(gantry_angles: Sequence[float],
                       out_dir: Path,
                       patient: str = "") -> Path:
    fig = plt.figure(figsize=(7, 8), facecolor="#1a1a2e")
    fig.suptitle(f"Beam geometry — {patient}", fontsize=13,
                 fontweight="bold", color="white", y=0.97)

    ax = fig.add_axes([0.1, 0.18, 0.8, 0.75], projection="polar",
                      facecolor="#1a1a2e")
    colors = plt.cm.tab10(np.linspace(0, 1, len(gantry_angles)))
    for i, a in enumerate(gantry_angles):
        ang = np.deg2rad(a)
        ax.annotate("", xy=(ang, 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=2))
        ax.text(ang, 1.10, f"{a:.0f}°", ha="center", va="center",
                fontsize=8, color=colors[i])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, [0.25] * 200, "w--", lw=1, alpha=0.5)
    ax.text(0, 0, "Patient", ha="center", va="center", fontsize=9, color="white")
    ax.set_ylim(0, 1.25)
    ax.tick_params(colors="white")
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(labelcolor="white")

    fig.text(0.05, 0.02,
             f"  * {len(gantry_angles)} gantry angles, "
             f"step {360/len(gantry_angles):.0f}°.\n"
             "  * Coplanar (couch=0°), standard IMRT layout.",
             fontsize=8, color="white", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a4e",
                       edgecolor="#6688bb", alpha=0.9),
             family="monospace")
    path = out_dir / "2_beam_geometry.png"
    _save(fig, path, facecolor=fig.get_facecolor())
    return path


# ---------------------------------------------------------------------------
# 3. Per-beam dose maps (MIP)
# ---------------------------------------------------------------------------
def plot_per_beam_dose(per_beam_dose: np.ndarray,
                       dose_shape_zyx: Sequence[int],
                       gantry_angles: Sequence[float],
                       out_dir: Path,
                       patient: str = "") -> Path:
    """per_beam_dose: (V, n_beams) sum-of-bixels per beam reshaped to volumes."""
    nz, ny, nx = dose_shape_zyx
    n_beams = per_beam_dose.shape[1]
    rows = int(np.ceil(n_beams / 3))

    fig = plt.figure(figsize=(13, 4 * rows + 1))
    fig.suptitle(f"Per-beam dose (MIP along z) — {patient}",
                 fontsize=13, fontweight="bold", y=0.99)
    vmax = float(per_beam_dose.max() or 1.0)

    for b in range(n_beams):
        r, c = divmod(b, 3)
        ax = fig.add_axes([0.03 + c * 0.32,
                           0.18 + (rows - 1 - r) * (0.78 / rows),
                           0.29,
                           0.70 / rows])
        vol = per_beam_dose[:, b].reshape(nz, ny, nx)
        ax.imshow(vol.max(axis=0), cmap="hot", origin="lower",
                  vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"Beam {b}  ({gantry_angles[b]:.0f}°)", fontsize=9)
        ax.axis("off")

        # arrow showing beam entry direction
        ang = np.deg2rad(gantry_angles[b])
        cx, cy = nx * 0.5, ny * 0.5
        r_arr = min(nx, ny) * 0.38
        ax.annotate("", xy=(cx, cy),
                    xytext=(cx - np.sin(ang) * r_arr, cy - np.cos(ang) * r_arr),
                    arrowprops=dict(arrowstyle="->", color="cyan",
                                    lw=1.5, mutation_scale=12))

    sm = plt.cm.ScalarMappable(cmap="hot",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.92, 0.22, 0.02, 0.70]))
    cbar.set_label("Dose (Gy / fraction at weight=1)", fontsize=9)

    _add_explanation(fig, [
        "MIP = Maximum Intensity Projection along z; shows beam footprint.",
        "Cyan arrow = beam entry direction (gantry angle).",
        "Bright hot-spots should rotate consistently with the angles.",
    ])
    path = out_dir / "3_per_beam_dose.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 4. Beam statistics
# ---------------------------------------------------------------------------
def plot_beam_statistics(per_beam_dose: np.ndarray,
                         gantry_angles: Sequence[float],
                         out_dir: Path,
                         patient: str = "") -> Path:
    labels = [f"{a:.0f}°" for a in gantry_angles]
    n = len(labels)
    max_doses = per_beam_dose.max(axis=0)
    mean_doses = per_beam_dose.mean(axis=0)
    nonzero = (per_beam_dose > 0).sum(axis=0)

    x = np.arange(n)
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Per-beam statistics — {patient}",
                 fontsize=13, fontweight="bold", y=0.98)

    ax1 = fig.add_axes([0.06, 0.22, 0.40, 0.68])
    ax2 = fig.add_axes([0.55, 0.22, 0.40, 0.68])
    w = 0.35
    ax1.bar(x - w / 2, max_doses, w, label="max",  color="tomato",
            edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, mean_doses, w, label="mean", color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right",
                                            fontsize=9)
    ax1.set_ylabel("Dose (Gy / fraction)", fontsize=10)
    ax1.set_title("Max vs. mean dose per beam", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    ax2.bar(x, nonzero, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right",
                                            fontsize=9)
    ax2.set_ylabel("Irradiated voxels", fontsize=10)
    ax2.set_title("Non-zero voxels per beam", fontsize=11)
    ax2.grid(axis="y", alpha=0.3); ax2.set_facecolor("#f8f9fa")

    _add_explanation(fig, [
        "Max dose ≈ peak per-bixel contribution; should be similar across beams.",
        "Mean dose is small because most voxels receive nothing from a single beam.",
        "Non-zero voxel count indicates anatomical reach of each angle.",
    ])
    path = out_dir / "4_beam_statistics.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 5. Dose distribution on CT (3-plane overlay)
# ---------------------------------------------------------------------------
def plot_dose_distribution(ct_array: np.ndarray,
                           dose_on_ct: np.ndarray,
                           masks: Dict[str, np.ndarray],
                           out_dir: Path,
                           patient: str = "") -> Path:
    nz, ny, nx = ct_array.shape
    ptv = next((m for n, m in masks.items() if n.startswith("PTV") and m.any()),
               None)
    if ptv is not None:
        iz, iy, ix = np.argwhere(ptv).mean(axis=0).astype(int)
    else:
        iz, iy, ix = nz // 2, ny // 2, nx // 2

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Dose distribution on CT (uniform action) — {patient}",
                 fontsize=14, fontweight="bold", y=0.98)
    axes = [fig.add_axes([0.02 + i * 0.32, 0.22, 0.29, 0.70]) for i in range(3)]
    views = [
        (axes[0], ct_array[iz, :, :], dose_on_ct[iz, :, :], f"Axial (z={iz})"),
        (axes[1], ct_array[:, iy, :], dose_on_ct[:, iy, :], f"Coronal (y={iy})"),
        (axes[2], ct_array[:, :, ix], dose_on_ct[:, :, ix], f"Sagittal (x={ix})"),
    ]
    dmax = float(dose_on_ct.max() or 1.0)
    im = None
    for ax, ct_s, d_s, lab in views:
        ax.imshow(ct_s, cmap="gray", vmin=-200, vmax=300,
                  origin="lower", aspect="auto")
        masked = np.ma.masked_where(d_s < 0.05 * dmax, d_s)
        im = ax.imshow(masked, cmap="jet", alpha=0.5,
                       vmin=0, vmax=dmax, origin="lower", aspect="auto")
        ax.set_title(lab, fontsize=10); ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, cax=fig.add_axes([0.96, 0.22, 0.015, 0.70]))
        cbar.set_label("Dose (Gy / fraction)", fontsize=10)

    _add_explanation(fig, [
        "Sanity check: dose volume produced by a UNIFORM beamlet action (all 1).",
        "Hot region should overlap the PTV (red contour in plot 1).",
        "Cold edges = beam entry/exit attenuation — expected.",
    ])
    path = out_dir / "5_dose_distribution.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# 6. DVH
# ---------------------------------------------------------------------------
def plot_dvh(dose_on_ct: np.ndarray,
             masks: Dict[str, np.ndarray],
             prescriptions: Dict[str, float],
             out_dir: Path,
             patient: str = "",
             n_fractions: int = 35) -> Path:
    dose_total = dose_on_ct * n_fractions
    dose_max = float(dose_total.max() or 1.0)
    bins = np.linspace(0, dose_max * 1.05, 300)

    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(f"DVH (uniform action × {n_fractions} fx) — {patient}",
                 fontsize=13, fontweight="bold", y=0.98)
    ax = fig.add_axes([0.09, 0.22, 0.87, 0.70])

    for name, rx in prescriptions.items():
        ax.axvline(rx, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(rx + 0.5, 101, f"{rx:.0f} Gy", fontsize=7.5,
                color="gray", va="bottom")

    for name, m in masks.items():
        if name == "BODY":
            continue
        bool_m = m.astype(bool)
        if not bool_m.any():
            continue
        color = STRUCT_COLORS.get(name, "black")
        d = dose_total[bool_m]
        dvh = (d[:, None] >= bins[None, :]).mean(axis=0) * 100.0
        ax.plot(bins, dvh, color=color, lw=2, label=name)

    ax.set_xlabel("Dose (Gy, full course)", fontsize=11)
    ax.set_ylabel("Volume (%)", fontsize=11)
    ax.set_xlim(0, dose_max * 1.05); ax.set_ylim(0, 107)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3); ax.set_facecolor("#f8f9fa")

    _add_explanation(fig, [
        "DVH = % of structure receiving at least X Gy.",
        "Curves are computed assuming the uniform-action dose × n_fractions.",
        "PTV curves should stay high & right of their prescription line.",
        "OAR curves should drop steeply at low dose.",
    ])
    path = out_dir / "6_dvh.png"
    _save(fig, path)
    return path


# ===========================================================================
# Evaluation charts (added for per-patient post-eval diagnostics)
# ===========================================================================

# ---------------------------------------------------------------------------
# E1. Beamlet fluence maps  (accumulated over all fractions)
# ---------------------------------------------------------------------------
def plot_fluence_maps(accumulated_action: np.ndarray,
                      n_beams: int,
                      beamlet_h: int,
                      beamlet_w: int,
                      out_dir: Path,
                      patient: str = "",
                      title_prefix: str = "Agent fluence") -> Path:
    """Plot one 16×16 intensity heatmap per beam arranged in a grid.

    ``accumulated_action`` shape: (n_beams * beamlet_h * beamlet_w,)
    """
    action_3d = accumulated_action.reshape(n_beams, beamlet_h, beamlet_w)
    cols = min(n_beams, 5)
    rows = int(np.ceil(n_beams / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.8 + 0.5, rows * 2.8 + 1.2))
    axes_flat = np.array(axes).ravel()
    fig.suptitle(f"{title_prefix} — {patient}",
                 fontsize=12, fontweight="bold", y=1.01)

    vmax = float(action_3d.max()) or 1.0
    gantry_step = 360 / n_beams
    for b in range(n_beams):
        ax = axes_flat[b]
        im = ax.imshow(action_3d[b], cmap="inferno",
                       vmin=0, vmax=vmax, aspect="equal", origin="lower")
        angle = round(b * gantry_step)
        ax.set_title(f"Beam {b}  ({angle}°)", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for b in range(n_beams, len(axes_flat)):
        axes_flat[b].axis("off")

    fig.tight_layout()
    import re
    slug = re.sub(r"[^\w\-]", "_", title_prefix.lower().replace(" ", "_"))
    slug = re.sub(r"_+", "_", slug).strip("_")
    path = out_dir / f"eval_{slug}.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# E2. DIM beam-sensitivity maps  (sum of |DIM columns| per beamlet → 16×16)
# ---------------------------------------------------------------------------
def plot_dim_sensitivity(dim,          # scipy.sparse (V, B) or dense ndarray
                         n_beams: int,
                         beamlet_h: int,
                         beamlet_w: int,
                         out_dir: Path,
                         patient: str = "") -> Path:
    """For each beam show the beamlet sensitivity: sum_v |DIM[v, b]| per bixel.

    This reveals which bixels in the 16×16 grid actually deposit dose
    (non-zero rows in the DIM) and how strongly.
    """
    import scipy.sparse as sp

    n_beamlets = n_beams * beamlet_h * beamlet_w
    if sp.issparse(dim):
        col_norms = np.asarray(np.abs(dim).sum(axis=0)).ravel()  # (B,)
    else:
        col_norms = np.abs(dim).sum(axis=0).ravel()
    col_norms = col_norms[:n_beamlets]

    sensitivity = col_norms.reshape(n_beams, beamlet_h, beamlet_w)
    vmax = float(sensitivity.max()) or 1.0

    cols = min(n_beams, 5)
    rows = int(np.ceil(n_beams / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.8 + 0.5, rows * 2.8 + 1.2))
    axes_flat = np.array(axes).ravel()
    fig.suptitle(f"DIM beamlet sensitivity (Σ|DIM| per bixel) — {patient}",
                 fontsize=12, fontweight="bold", y=1.01)

    gantry_step = 360 / n_beams
    for b in range(n_beams):
        ax = axes_flat[b]
        im = ax.imshow(sensitivity[b], cmap="viridis",
                       vmin=0, vmax=vmax, aspect="equal", origin="lower")
        angle = round(b * gantry_step)
        ax.set_title(f"Beam {b}  ({angle}°)", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for b in range(n_beams, len(axes_flat)):
        axes_flat[b].axis("off")

    fig.tight_layout()
    path = out_dir / "eval_dim_sensitivity.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# E3. DVH: predicted vs ground-truth
# ---------------------------------------------------------------------------
def plot_dvh_comparison(predicted_dose: np.ndarray,
                        reference_dose: np.ndarray,
                        masks: Dict[str, np.ndarray],
                        prescriptions: Dict[str, float],
                        oar_tolerances: Dict[str, float],
                        out_dir: Path,
                        patient: str = "",
                        n_bins: int = 300) -> Path:
    """Side-by-side DVH curves for the agent prediction vs ground-truth."""
    dose_max = float(max(predicted_dose.max(), reference_dose.max()) * 1.05) or 80.0
    bins = np.linspace(0, dose_max, n_bins)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"DVH comparison: predicted vs ground-truth — {patient}",
                 fontsize=12, fontweight="bold")

    # prescription / tolerance reference lines
    for name, rx in prescriptions.items():
        ax.axvline(rx, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.text(rx + 0.3, 101.5, f"{rx:.0f}", fontsize=7, color="gray")
    for name, tol in oar_tolerances.items():
        ax.axvline(tol, color="#aaa", lw=0.6, ls=":", alpha=0.4)

    plotted = set()
    for name, m in masks.items():
        if name == "BODY" or not m.astype(bool).any():
            continue
        color = STRUCT_COLORS.get(name, "black")
        bool_m = m.astype(bool)
        d_pred = predicted_dose[bool_m]
        d_ref  = reference_dose[bool_m]
        dvh_pred = (d_pred[:, None] >= bins[None, :]).mean(axis=0) * 100.0
        dvh_ref  = (d_ref[:, None]  >= bins[None, :]).mean(axis=0) * 100.0
        lbl = name if name not in plotted else "_nolegend_"
        ax.plot(bins, dvh_pred, color=color, lw=2,      label=f"{name} (pred)")
        ax.plot(bins, dvh_ref,  color=color, lw=1.5, ls="--", label=f"{name} (GT)")
        plotted.add(name)

    ax.set_xlabel("Dose (Gy, full course)", fontsize=11)
    ax.set_ylabel("Volume (%)", fontsize=11)
    ax.set_xlim(0, dose_max); ax.set_ylim(0, 107)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.3); ax.set_facecolor("#f8f9fa")
    fig.text(0.01, 0.01,
             "  Solid = agent prediction   |   Dashed = ground-truth planner\n"
             "  Gray dashed verticals = PTV prescriptions   |   Dotted = OAR tolerances",
             fontsize=8, va="bottom", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4ff",
                       edgecolor="#aabbdd", alpha=0.9))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = out_dir / "eval_dvh_comparison.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# E4. Dose slice comparison  (predicted vs GT on CT, 3-plane)
# ---------------------------------------------------------------------------
def plot_dose_slice_comparison(ct: np.ndarray,
                               predicted_dose: np.ndarray,
                               reference_dose: np.ndarray,
                               masks: Dict[str, np.ndarray],
                               out_dir: Path,
                               patient: str = "") -> Path:
    """3-plane CT overlay: top row = predicted dose, bottom row = GT."""
    nz, ny, nx = ct.shape
    ptv = next((m for n, m in masks.items() if n.startswith("PTV") and m.any()), None)
    iz, iy, ix = (np.argwhere(ptv).mean(axis=0).astype(int)
                  if ptv is not None else (nz // 2, ny // 2, nx // 2))

    dmax = float(max(predicted_dose.max(), reference_dose.max())) or 1.0
    views = [
        ("Axial",    slice(iz, iz + 1), "z"),
        ("Coronal",  slice(iy, iy + 1), "y"),
        ("Sagittal", slice(ix, ix + 1), "x"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Dose on CT (top: predicted | bottom: ground-truth) — {patient}",
                 fontsize=12, fontweight="bold")

    def _slc(vol, axis, idx):
        if axis == "z":  return vol[idx, :, :]
        if axis == "y":  return vol[:, idx, :]
        return vol[:, :, idx]

    im_last = None
    for col, (label, _, axis) in enumerate(views):
        ct_s  = _slc(ct,               axis, [iz, iy, ix][col])
        dp_s  = _slc(predicted_dose,   axis, [iz, iy, ix][col])
        dg_s  = _slc(reference_dose,   axis, [iz, iy, ix][col])
        for row, (dose_s, row_label) in enumerate([(dp_s, "Predicted"),
                                                   (dg_s, "GT")]):
            ax = axes[row, col]
            ax.imshow(ct_s, cmap="gray", vmin=-200, vmax=300,
                      origin="lower", aspect="auto")
            masked_d = np.ma.masked_where(dose_s < 0.03 * dmax, dose_s)
            im_last = ax.imshow(masked_d, cmap="jet", alpha=0.55,
                                vmin=0, vmax=dmax,
                                origin="lower", aspect="auto")
            ax.set_title(f"{label}  [{row_label}]", fontsize=9)
            ax.axis("off")
            # structure contours on axial slice only
            if axis == "z":
                for name, m in masks.items():
                    if name == "BODY": continue
                    sl = m[iz, :, :]
                    if sl.any():
                        ax.contour(sl, levels=[0.5],
                                   colors=[STRUCT_COLORS.get(name, "white")],
                                   linewidths=0.9)

    if im_last is not None:
        cbar = fig.colorbar(im_last,
                            cax=fig.add_axes([0.93, 0.15, 0.015, 0.70]))
        cbar.set_label("Dose (Gy, full course)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    path = out_dir / "eval_dose_slices.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# E5. Per-fraction reward curve  (oar penalty / ptv reward across 35 fx)
# ---------------------------------------------------------------------------
def plot_fraction_rewards(fraction_records: list[dict],
                          out_dir: Path,
                          patient: str = "") -> Path:
    """Line chart of oar_penalty and ptv_reward over the 35 fractions."""
    fxs   = [r["fraction_index"] for r in fraction_records]
    oar   = [r["oar_penalty"]    for r in fraction_records]
    ptv   = [r["ptv_reward"]     for r in fraction_records]
    total = [-r["oar_penalty"] + r["ptv_reward"] for r in fraction_records]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Per-fraction reward components — {patient}",
                 fontsize=12, fontweight="bold")

    ax.plot(fxs, ptv,   color="steelblue",  lw=2,   label="PTV reward")
    ax.plot(fxs, [-v for v in oar], color="tomato", lw=2, label="-OAR penalty")
    ax.plot(fxs, total, color="green",      lw=1.5, ls="--", label="Net reward")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xlabel("Fraction index", fontsize=11)
    ax.set_ylabel("Reward (a.u.)", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    path = out_dir / "eval_fraction_rewards.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Top-level convenience wrapper called from evaluate.py
# ---------------------------------------------------------------------------
def generate_eval_charts(env,
                         accumulated_action: np.ndarray,
                         predicted_dose: np.ndarray,
                         reference_dose: np.ndarray,
                         fraction_records: list[dict],
                         out_dir: Path) -> list[Path]:
    """Generate all evaluation charts for one patient and return their paths."""
    patient = env._patient
    cfg     = env.cfg
    masks   = env._structure_masks()

    saved: list[Path] = []

    # E1 — agent fluence
    saved.append(plot_fluence_maps(
        accumulated_action, cfg.n_beams, cfg.beamlet_h, cfg.beamlet_w,
        out_dir, patient, title_prefix="Agent fluence (accumulated)"))

    # E2 — warm-start fluence (if the cached file exists)
    ws_path = env.root / patient / "warmstart_action.npy"
    if ws_path.exists():
        ws_action = np.load(ws_path)
        saved.append(plot_fluence_maps(
            ws_action, cfg.n_beams, cfg.beamlet_h, cfg.beamlet_w,
            out_dir, patient, title_prefix="Warm-start fluence (a*)"))

    # E3 — DIM sensitivity
    if env._dose_influence_matrix is not None:
        saved.append(plot_dim_sensitivity(
            env._dose_influence_matrix, cfg.n_beams,
            cfg.beamlet_h, cfg.beamlet_w, out_dir, patient))

    # E4 — DVH comparison
    saved.append(plot_dvh_comparison(
        predicted_dose, reference_dose, masks,
        cfg.prescription, cfg.oar_tolerance, out_dir, patient))

    # E5 — dose slice comparison
    ct = env._data["ct"]
    saved.append(plot_dose_slice_comparison(
        ct, predicted_dose, reference_dose, masks, out_dir, patient))

    # E6 — per-fraction reward curve
    if fraction_records:
        saved.append(plot_fraction_rewards(fraction_records, out_dir, patient))

    return saved
