# measure/pipeline.py
# high-level wrapper that runs detection, platesolve, spike fit, and metrics for a frame

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os
from pathlib import Path
import numpy as np
from astropy.io import fits

from .types import (
    BranchImageResult,
    MeasurementInput,
    MeasurementMetadata,
    MeasurementResult,
)
from .preprocess import prepare_star_branch_input, prepare_stripe_branch_input
from .platesolve import run_platesolve
from .spikes import measure_diffraction_angle
from .metrics import compute_metrics
from .io import make_run_dir, write_branch_fits, write_summary


# ---------- internal helpers ----------

def _load_fits(filepath: str) -> Tuple[np.ndarray, fits.Header]:
    try:
        with fits.open(filepath) as hdul:
            hdu = hdul[0]
            data = hdu.data
            hdr = hdu.header
    except Exception as e:
        raise IOError(f"could not open FITS file '{filepath}': {e}")

    if data is None:
        raise ValueError(f"FITS file '{filepath}' contains no data")
    if data.ndim != 2:
        raise ValueError(f"FITS image must be 2‑D; got ndim={data.ndim}")

    return data.astype(np.float32), hdr


# ---------- public pipeline ----------

def run_measurement_pipeline(
    filepath: str,
    output_dir: str = "out",
    run_name: Optional[str] = None,
    save_outputs: bool = True,
    keep_intermediates: bool = True,
    show: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> MeasurementResult:
    """Run the measurement pipeline on a FITS image.

    This updated flow creates two preprocessing branches, runs the plate
    solver on the star branch and the diffraction-angle measurement on the
    stripe branch, computes derived metrics, and optionally writes outputs.

    Parameters
    ----------
    filepath : str
        Input FITS path.
    output_dir : str
        Root directory for saving outputs (will contain a run subfolder).
    run_name : str or None
        Name for this run; if None one is generated from the input filename
        and timestamp.
    save_outputs : bool
        If True, pipeline will create files under `out/<run_name>`.
    keep_intermediates : bool
        If True, intermediate images/results will be stored in the returned
        result object.
    show : bool
        If True, display branch images during execution.
    overrides, defaults : dict, optional
        Metadata precedence containers.
    verbose : bool
        Print progress messages.
    """
    messages: list[str] = []
    intermediates: Dict[str, Any] = {}
    output_paths: Dict[str, str] = {}

    # 1. load data + metadata
    image, header = _load_fits(filepath)
    meta = MeasurementMetadata.from_header(header, overrides=overrides, defaults=defaults)
    inp = MeasurementInput(filepath=filepath, image=image, header=header, meta=meta)
    messages.append("loaded FITS")

    # optional run directory
    runpath: Optional[Path] = None
    if save_outputs:
        runpath = make_run_dir(output_dir, run_name, filepath)
        output_paths["run_dir"] = str(runpath)
        messages.append(f"created run directory {runpath}")

    # 2. branch preprocessing
    star_branch = prepare_star_branch_input(image, meta)
    stripe_branch = prepare_stripe_branch_input(image, meta)
    messages.extend(star_branch.messages)
    messages.extend(stripe_branch.messages)
    # spike and plate messages are appended after those stages run
    if keep_intermediates:
        intermediates["star_branch_image"] = star_branch.image
        intermediates["stripe_branch_image"] = stripe_branch.image

    if save_outputs and runpath is not None:
        # save branch images to FITS
        p1 = write_branch_fits(star_branch, runpath, header, "star_branch")
        p2 = write_branch_fits(stripe_branch, runpath, header, "stripe_branch")
        output_paths["star_branch_fits"] = str(p1)
        output_paths["stripe_branch_fits"] = str(p2)
        messages.append(f"wrote branch FITS files")

    # 3. stripe measurement (first — so we can mask stripes before platesolving)
    spike_res = measure_diffraction_angle(stripe_branch, meta, save_dir=runpath)
    messages.append("diffraction angle stage completed")
    messages.extend(spike_res.messages)

    # 4. build stripe-masked star image and platesolve
    stripe_mask = spike_res.debug.get("stripe_mask")
    masked_star_branch = _apply_stripe_mask(star_branch, stripe_mask)
    messages.append(
        f"stripe mask applied: "
        f"{int(stripe_mask.sum()) if stripe_mask is not None else 0} px blanked"
    )

    plate_res = run_platesolve(masked_star_branch, meta)
    messages.append("platesolve stage completed")
    messages.extend(plate_res.messages)

    # 5. metrics
    metrics = compute_metrics(plate_res, spike_res, meta)
    messages.append("metrics computed")

    # 6. display — each result shown exactly once
    if show:
        _show_stripe_angle(
            stripe_branch.image, spike_res,
            save_path=runpath / "stripe_angle.png" if runpath else None,
        )
        _show_platesolve(
            masked_star_branch.image, plate_res,
            save_path=runpath / "platesolve.png" if runpath else None,
        )
        _show_final_result(
            masked_star_branch.image, plate_res, spike_res, metrics,
            save_path=runpath / "final_result.png" if runpath else None,
        )

    # determine overall success
    success = bool(metrics.sky_angle_deg is not None)

    result = MeasurementResult(
        input_data=inp,
        star_branch=star_branch,
        stripe_branch=stripe_branch,
        platesolve_result=plate_res,
        spike_result=spike_res,
        metrics=metrics,
        success=success,
        messages=messages,
        output_paths=output_paths,
        intermediates=intermediates if keep_intermediates else {},
    )

    if save_outputs and runpath is not None:
        write_summary(result, runpath)
        messages.append("wrote summary files")

    if verbose:
        for m in messages:
            print(m)

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _apply_stripe_mask(
    star_branch: BranchImageResult,
    stripe_mask: Optional[np.ndarray],
) -> BranchImageResult:
    """Return a copy of star_branch with accepted stripe pixels zeroed out."""
    import copy
    masked = copy.copy(star_branch)
    if stripe_mask is not None and stripe_mask.any():
        img = star_branch.image.copy()
        img[stripe_mask] = 0.0
        masked.image = img
        masked.messages = list(star_branch.messages) + ["stripe pixels masked before platesolve"]
    return masked


def _show_platesolve(
    star_image: np.ndarray,
    plate_res,
    save_path: Optional[Path] = None,
) -> None:
    """Star branch image with WCS sky-grid overlay and detected sources circled."""
    try:
        import matplotlib.pyplot as plt
        import warnings
        from matplotlib.patches import Circle
        from astropy.wcs import WCS
        from astropy.io import fits as _fits
    except ImportError:
        return

    wcs = None
    if plate_res.success and plate_res.wcs_info is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(plate_res.wcs_info) if isinstance(plate_res.wcs_info, _fits.Header) else plate_res.wcs_info

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=subplot_kw)

    finite = star_image[np.isfinite(star_image)]
    vmin, vmax = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1))
    disp = np.arcsinh(np.clip(star_image, vmin, vmax))
    ax.imshow(disp, origin="lower", cmap="gray",
              vmin=np.arcsinh(vmin), vmax=np.arcsinh(vmax))

    if wcs is not None:
        ax.coords.grid(True, color="cyan", alpha=0.4, linestyle="--", linewidth=0.7)
        ax.coords["ra"].set_axislabel("RA")
        ax.coords["dec"].set_axislabel("Dec")

    sources = plate_res.debug.get("sources", {})
    xs, ys  = sources.get("x", np.array([])), sources.get("y", np.array([]))
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), radius=12,
                             edgecolor="lime", facecolor="none",
                             linewidth=0.8, alpha=0.75))

    status = "solved" if plate_res.success else "FAILED"
    title = f"Plate solve ({status})"
    if plate_res.success:
        title += f"  RA={plate_res.ra_deg:.3f}°  Dec={plate_res.dec_deg:.3f}°  rot={plate_res.rot_deg:.2f}°"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _show_stripe_angle(
    stripe_image: np.ndarray,
    spike_res,
    save_path: Optional[Path] = None,
) -> None:
    """Stripe branch image with ensemble angle line overlay."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = stripe_image[np.isfinite(stripe_image)]
    vmin, vmax = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1))
    ax.imshow(stripe_image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    h, w = stripe_image.shape[:2]
    angle = spike_res.image_angle_deg
    if angle is not None:
        cy, cx = h / 2.0, w / 2.0
        L = float(max(h, w))
        tr = np.radians(angle)
        ax.plot(
            [cx - np.cos(tr) * L, cx + np.cos(tr) * L],
            [cy + np.sin(tr) * L, cy - np.sin(tr) * L],
            color="lime", linewidth=1.8, alpha=0.9,
        )
        label = (f"{angle:.2f}° ± {spike_res.sigma_angle_deg:.2f}°"
                 f"   Q={spike_res.quality:.3f}"
                 f"   {spike_res.n_accepted}/{spike_res.n_detected} segs")
        color = "lime"
    else:
        label = "No valid angle solution"
        color = "red"

    # Restore image extents so the line is clipped at the image edge
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)

    ax.text(0.02, 0.97, label, transform=ax.transAxes, color=color, fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=3))
    ax.set_title("Stripe branch — ensemble angle")
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _show_final_result(
    star_image: np.ndarray,
    plate_res,
    spike_res,
    metrics,
    save_path: Optional[Path] = None,
) -> None:
    """Combined summary: WCS grid, source circles, north arrow, spectra orientation."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import warnings
        from matplotlib.patches import FancyArrowPatch, Circle
        from astropy.wcs import WCS
        from astropy.io import fits as _fits
    except ImportError:
        return

    wcs = None
    if plate_res.success and plate_res.wcs_info is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(plate_res.wcs_info) if isinstance(plate_res.wcs_info, _fits.Header) else plate_res.wcs_info

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw=subplot_kw)

    finite = star_image[np.isfinite(star_image)]
    vmin, vmax = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1))
    disp = np.arcsinh(np.clip(star_image, vmin, vmax))
    ax.imshow(disp, origin="lower", cmap="gray",
              vmin=np.arcsinh(vmin), vmax=np.arcsinh(vmax))

    h, w = star_image.shape[:2]

    # WCS grid
    if wcs is not None:
        ax.coords.grid(True, color="cyan", alpha=0.35, linestyle="--", linewidth=0.7)
        ax.coords["ra"].set_axislabel("RA")
        ax.coords["dec"].set_axislabel("Dec")

    # Source circles
    sources = plate_res.debug.get("sources", {})
    xs, ys = sources.get("x", np.array([])), sources.get("y", np.array([]))
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), radius=12,
                             edgecolor="lime", facecolor="none",
                             linewidth=0.8, alpha=0.75))

    cx, cy = w / 2.0, h / 2.0
    arrow_len = min(h, w) * 0.12

    # North arrow — direction of +Y sky axis in pixel frame
    # rot_deg is PA of detector +Y East of North, so North is in the +Y direction when rot=0.
    # In general, the pixel direction toward North makes angle rot_deg with image +Y axis (CCW).
    if plate_res.success and plate_res.rot_deg is not None:
        rot_rad = np.radians(plate_res.rot_deg)
        # pixel vector pointing toward North: rotate +Y by -rot_deg
        ndx = -np.sin(rot_rad)
        ndy = np.cos(rot_rad)
        ax.annotate(
            "", xy=(cx + ndx * arrow_len, cy + ndy * arrow_len),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.0),
        )
        ax.text(cx + ndx * (arrow_len + 8), cy + ndy * (arrow_len + 8),
                "N", color="yellow", fontsize=11, fontweight="bold",
                ha="center", va="center")

    # Spectra orientation line — sky_angle_deg East of North
    if metrics.sky_angle_deg is not None and plate_res.rot_deg is not None:
        sky_rad = np.radians(metrics.sky_angle_deg)
        # Convert sky-frame East-of-North to pixel direction.
        # North pixel direction: angle rot_deg from image +Y.
        # East is 90° CCW from North on sky, but in standard image coords (origin=lower)
        # East is toward decreasing RA, which is +X for typical WCS.
        # The pixel angle for sky PA θ: pixel_angle_from_+X = (90 - rot_deg - θ) ... easier via:
        # sky PA θ EoN: pixel vector = R(-rot_deg) applied to [sin(θ), cos(θ)] in sky (N=+Y, E=+X image)
        rot_rad = np.radians(plate_res.rot_deg)
        sn, cs = np.sin(sky_rad), np.cos(sky_rad)  # East, North components in sky frame
        # pixel +Y is North rotated by rot_deg; pixel +X is East (approx, ignoring RA cos factor)
        # pixel_x = cos(rot_deg)*sn + sin(rot_deg)*cs ... full rotation:
        pdx = np.cos(rot_rad) * sn - np.sin(rot_rad) * cs
        pdy = np.sin(rot_rad) * sn + np.cos(rot_rad) * cs
        # Normalise
        norm = np.hypot(pdx, pdy) + 1e-10
        pdx, pdy = pdx / norm, pdy / norm

        L = min(h, w) * 0.38
        sigma = spike_res.sigma_angle_deg or 0.0
        label_str = (
            f"Spectra: {metrics.sky_angle_deg:.1f}° ± {sigma:.1f}° EoN"
            if spike_res.success else "Spectra: no valid solution"
        )
        ax.plot(
            [cx - pdx * L, cx + pdx * L],
            [cy - pdy * L, cy + pdy * L],
            color="orange", linewidth=2.0, alpha=0.9, label=label_str,
        )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)

    # Annotation text
    lines = []
    if plate_res.success:
        lines.append(f"RA={plate_res.ra_deg:.4f}°  Dec={plate_res.dec_deg:.4f}°  rot={plate_res.rot_deg:.2f}°")
    else:
        lines.append("Platesolve: FAILED")
    if metrics.sky_angle_deg is not None:
        sigma = spike_res.sigma_angle_deg or 0.0
        lines.append(f"Spectra orientation: {metrics.sky_angle_deg:.2f}° ± {sigma:.2f}° East of North")
    else:
        lines.append("Spectra orientation: unavailable")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
            color="white", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.55, pad=4))

    # Legend patches
    handles = [
        mpatches.Patch(color="lime", label=f"Sources ({len(xs)})"),
        mpatches.Patch(color="yellow", label="North"),
        mpatches.Patch(color="orange", label="Spectra orientation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              facecolor="black", labelcolor="white", framealpha=0.6)

    ax.set_title("Final measurement result", fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
