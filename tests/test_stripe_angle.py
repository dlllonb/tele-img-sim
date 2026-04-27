"""
Test 2 — Simulation → stripe angle measurement.

Generates a synthetic astrophotograph using the analytic grating simulator
(parameters taken from the default sim_config.toml), then passes the
result directly through the stripe-branch preprocessing and the ensemble
line-segment angle estimator.

The output figure shows:
  • The stripe branch image (background-subtracted, smoothed) in greyscale.
  • Every accepted diffraction stripe segment, colour-coded and labelled
    with its individual measured angle.
  • The global ensemble solution as a bright green line across the image.
  • Per-segment statistics and the final angle / quality in the title.

Outputs saved to this directory:
    test_stripe_angle.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Simulator
from sim.simulator import run_sim_and_report

# Measurement internals — imported at the level the prompt specifies
from measure.preprocess import prepare_stripe_branch_input
from measure.types import MeasurementMetadata
from measure.spikes import (
    _prepare_feature_image,
    _extract_candidates,
    _filter_candidates,
    _sigma_clip,
    _weighted_axial_mean,
    _weighted_sigma,
    _weight,
    _upscale_mask,
)

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Simulation parameters (default sim_config.toml values, image reduced to
# 1024×768 for practical test runtime while preserving the stripe pattern)
# ---------------------------------------------------------------------------
SIM_KWARGS = dict(
    # I/O — write nothing to disk; we use the in-memory RenderResult directly
    out_dir="out",
    run_name="test_stripe_run",
    stars_csv="sim/physics/starfields/field3.csv",
    overwrite=True,

    # Pointing (default)
    ra0_deg=10.12724197930297,
    dec0_deg=56.53718879378639,
    rot_deg=0.0,

    # Camera (default sensor, reduced size)
    cam_nx=1024,
    cam_ny=768,
    cam_pixel_um=2.4,
    cam_read_noise_e=1.6,
    cam_gain_e_per_adu=1.0,
    cam_qe=0.6,

    # Lens (default)
    lens_focal_mm=25,
    lens_f_number=1.2,
    lens_transmission=0.8,

    # Mask: analytic grating at 23° (default)
    mask_kind="grating",
    mask_grating_model="analytic",
    mask_lines_per_mm=100.0,
    mask_angle_deg=23.0,
    mask_order_max=1,
    mask_duty_cycle=0.5,
    mask_n_lambda=9,
    mask_psf_size_px=513,

    # Render (default)
    exposure_s=5.0,
    sky_mu_mag_per_arcsec2=21.5,
    lambda_eff_nm=550.0,
    band_nm=300.0,
    seeing_fwhm_arcsec=2.0,
    jitter_pointing_rms=20.0,
    seed=42,

    # Toggles: all on
    enable_sky=True,
    enable_stars=True,
    enable_psf=True,
    enable_jitter=True,
    enable_noise=True,

    # Output: suppress files and interactive plots
    show_plots=False,
    save_pngs=False,
    make_rois=False,
)


from astropy.io import fits

USE_EXISTING_IMAGE = True  # ← DEFAULT: use testsim.fit


def run_test():
    # -------------------------------------------------------------------
    # 1. Get image (either load FITS or run simulator)
    # -------------------------------------------------------------------
    if USE_EXISTING_IMAGE:
        fits_path = HERE / "testsim.fit"
        print(f"Loading existing FITS image: {fits_path}")

        with fits.open(fits_path) as hdul:
            image = hdul[0].data.astype(float)

        print(f"  Image shape: {image.shape}  peak: {image.max():.0f}")
    else:
        print("Running simulator …")
        frame, res, _paths = run_sim_and_report(**SIM_KWARGS)
        image = res.final_e
        print(f"  Image shape: {image.shape}  peak: {image.max():.0f} e-")

    # -------------------------------------------------------------------
    # 2. Stripe-branch preprocessing (unchanged)
    # -------------------------------------------------------------------
    meta = MeasurementMetadata()
    stripe_branch = prepare_stripe_branch_input(image, meta)
    stripe_img = stripe_branch.image

    # -------------------------------------------------------------------
    # 3. Ensemble stripe angle measurement (unchanged)
    # -------------------------------------------------------------------
    feature_img, factor = _prepare_feature_image(stripe_img)
    candidates, label_img = _extract_candidates(feature_img)
    accepted = _filter_candidates(candidates)
    if len(accepted) >= 3:
        accepted = _sigma_clip(accepted)

    n_det = len(candidates)
    n_acc = len(accepted)

    mean_angle = sigma_angle = quality = None
    if n_acc >= 2:
        angles  = [c["angle_deg"] for c in accepted]
        weights = [_weight(c)     for c in accepted]
        mean_angle, R = _weighted_axial_mean(angles, weights)
        sigma_angle   = _weighted_sigma(angles, weights, mean_angle)
        quality       = R * min(1.0, n_acc / 5.0)

    print(f"  Segments detected / accepted: {n_det} / {n_acc}")
    if mean_angle is not None:
        print(f"  Ensemble angle: {mean_angle:.2f}° ± {sigma_angle:.2f}°   "
              f"quality={quality:.3f}")
    else:
        print("  No valid angle solution.")


    # -------------------------------------------------------------------
    # 4. Build the per-segment colour overlay in original image coords.
    # -------------------------------------------------------------------
    h, w = stripe_img.shape
    overlay_rgba = np.zeros((h, w, 4), dtype=float)
    tab10 = cm.get_cmap("tab10")

    seg_label_texts = []   # for legend / annotations
    for i, cand in enumerate(accepted):
        color_rgb = tab10(i % 10)[:3]
        seg_mask = _upscale_mask(label_img == cand["label"], factor, (h, w))
        overlay_rgba[seg_mask, :3] = color_rgb
        overlay_rgba[seg_mask,  3] = 0.55   # alpha

        # centroid in original image coords: label_img is in feature-image coords
        row_f, col_f = cand["centroid"]
        cx_orig = col_f * factor
        cy_orig = row_f * factor
        seg_label_texts.append((cx_orig, cy_orig, f"{cand['angle_deg']:.1f}°", color_rgb))

    # -------------------------------------------------------------------
    # 5. Figure.
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9))

    # Background: stripe branch in greyscale (asinh stretch)
    finite = stripe_img[np.isfinite(stripe_img)]
    vmin, vmax = np.percentile(finite, [0.5, 99.8])
    disp = np.arcsinh(np.clip(stripe_img, vmin, vmax) / max(abs(vmax), 1e-30))
    ax.imshow(disp, origin="lower", cmap="gray",
              vmin=float(np.arcsinh(vmin / max(abs(vmax), 1e-30))),
              vmax=float(np.arcsinh(1.0)),
              interpolation="nearest")

    # Accepted segment colour masks
    ax.imshow(overlay_rgba, origin="lower", interpolation="nearest")

    # Per-segment angle annotation
    for cx_a, cy_a, label, color in seg_label_texts:
        ax.text(cx_a, cy_a, label, color=color, fontsize=7, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.5, boxstyle="round"))

    # Global ensemble solution line
    if mean_angle is not None:
        tr  = np.radians(mean_angle)
        L   = min(h, w) * 0.46
        cy0 = h / 2.0
        cx0 = w / 2.0
        ax.plot(
            [cx0 - np.cos(tr) * L, cx0 + np.cos(tr) * L],
            [cy0 + np.sin(tr) * L, cy0 - np.sin(tr) * L],
            color="lime", lw=2.5, alpha=0.95,
            label=f"Ensemble: {mean_angle:.2f}° ± {sigma_angle:.2f}°   Q={quality:.3f}",
        )

    # Dashed reference line at the configured grating angle
    ref_angle_deg = SIM_KWARGS["mask_angle_deg"]
    tr_ref = np.radians(ref_angle_deg)
    L_ref  = min(h, w) * 0.46
    ax.plot(
        [cx0 - np.cos(tr_ref) * L_ref, cx0 + np.cos(tr_ref) * L_ref],
        [cy0 + np.sin(tr_ref) * L_ref, cy0 - np.sin(tr_ref) * L_ref],
        color="yellow", lw=1.5, alpha=0.6, ls="--",
        label=f"Configured grating angle: {ref_angle_deg}°",
    )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.axis("off")

    solve_str = (
        f"{mean_angle:.2f}° ± {sigma_angle:.2f}°   quality={quality:.3f}"
        if mean_angle is not None else "no valid solution"
    )
    ax.set_title(
        f"Test 2 — Stripe angle measurement on simulated grating image\n"
        f"Grating: {SIM_KWARGS['mask_lines_per_mm']:.0f} lp/mm  "
        f"angle={ref_angle_deg}°  analytic model  |  "
        f"segments: {n_det} detected, {n_acc} accepted  |  "
        f"solution: {solve_str}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9,
              facecolor="black", labelcolor="white", framealpha=0.65)

    out_path = HERE / "test_stripe_angle.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path.name}")


if __name__ == "__main__":
    run_test()
