"""
Test 1 — Analytic grating PSF: diffraction order placement.

Verifies that the ±1 diffraction order peaks in the analytic PSF kernel
(`sim.physics.masks._kernel_grating_orders`) are placed at the pixel offset
predicted by the thin-grating small-angle formula:

    sep_px = f_mm * m * λ_mm / (pitch_mm * pix_mm)

Parameters are deliberately chosen so m=1 gives exactly 100 px separation,
making the expected and measured positions trivially comparable.

Outputs saved to this directory:
    test_psf_order_placement.png  — 2D kernel + 1D dispersion cross-section
    test_psf_order_placement.txt  — numerical summary with pass/fail
"""

import sys
from pathlib import Path

# Ensure the project root (one level up from tests/) is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sim.camera import Camera
from sim.lens import Lens
from sim.mask import Mask
from sim.frame import make_blank_frame
from sim.render import RenderConfig
from sim.physics.masks import kernel_for_mask, _diffraction_order_offsets_px

HERE = Path(__file__).resolve().parent
TOLERANCE_PX = 1.5  # maximum acceptable deviation between predicted and measured peak


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
# Chosen so that m=1 sep_px = f_mm * lambda_mm / pitch_mm / pix_mm = 100 px exactly:
#   f=10 mm, lambda=500 nm = 500e-6 mm, pitch=1/100 mm = 0.01 mm, pix=5 um = 5e-3 mm
#   sep = 10 * 500e-6 / 0.01 / 5e-3 = 10 * 0.05 / 0.005 = 100.0 px
FOCAL_MM       = 10.0
F_NUMBER       = 2.0
PIXEL_UM       = 5.0
LINES_PER_MM   = 100.0
LAMBDA_EFF_NM  = 500.0
BAND_NM        = 0.0   # monochromatic → orders at exact predicted position, no smear
MASK_ANGLE_DEG = 0.0   # dispersion along x-axis, simplifies cross-section check
ORDER_MAX      = 1

EXPECTED_SEP_PX = (FOCAL_MM * (LAMBDA_EFF_NM * 1e-6) / (1.0 / LINES_PER_MM) / (PIXEL_UM * 1e-3))


def _build_objects():
    cam   = Camera(nx=200, ny=200, pixel_um=PIXEL_UM)
    lens  = Lens(focal_mm=FOCAL_MM, f_number=F_NUMBER, transmission=1.0)
    frame = make_blank_frame(cam, lens, ra0_deg=0.0, dec0_deg=0.0)
    mask  = Mask(
        kind="grating",
        grating_model="analytic",
        lines_per_mm=LINES_PER_MM,
        angle_deg=MASK_ANGLE_DEG,
        order_max=ORDER_MAX,
        duty_cycle=0.5,
        n_lambda=1,
    )
    cfg = RenderConfig(
        exposure_s=1.0,
        lambda_eff_nm=LAMBDA_EFF_NM,
        band_nm=BAND_NM,
        mask=mask,
    )
    return frame, cfg, mask


def _find_order_peaks(profile: np.ndarray, center: int, min_distance: int = 20) -> np.ndarray:
    """Return x-indices of all significant peaks in the 1-D profile."""
    threshold = 0.04 * float(profile.max())
    idx, _ = find_peaks(profile.astype(float), height=threshold, distance=min_distance)
    return idx


def run_test():
    frame, cfg, mask = _build_objects()

    # -------------------------------------------------------------------
    # 1. Independent prediction using the helper already in the repo.
    # -------------------------------------------------------------------
    offsets = _diffraction_order_offsets_px(frame, cfg, mask)
    # offsets = [(m, dx, dy), ...].  With angle=0, dy=0 and dx=sep.
    assert len(offsets) >= ORDER_MAX, "helper returned no offsets"
    helper_sep_px = float(offsets[0][1])   # dx for m=1

    # -------------------------------------------------------------------
    # 2. Compute the PSF kernel via the analytic model.
    # -------------------------------------------------------------------
    sigma_px = 0.0   # no atmospheric seeing; test only the diffraction placement
    kernel = kernel_for_mask(frame, cfg, sigma_px, mask)

    assert kernel.ndim == 2, "kernel must be 2-D"
    ky, kx = kernel.shape
    cy = ky // 2   # center row
    cx = kx // 2   # center col  (where 0th order sits)

    # -------------------------------------------------------------------
    # 3. Locate peaks in the 1-D cross-section along the dispersion axis.
    # -------------------------------------------------------------------
    profile = kernel[cy, :].astype(float)
    peak_idx = _find_order_peaks(profile, center=cx, min_distance=int(EXPECTED_SEP_PX * 0.4))

    # Measured separations: distance from cx to each off-centre peak.
    measured_seps = [abs(int(p) - cx) for p in peak_idx if abs(int(p) - cx) > 5]
    measured_seps.sort()

    # We expect at least the m=1 order on each side.
    assert len(measured_seps) >= 1, (
        f"No off-centre peaks found.  peak_idx={peak_idx}, cx={cx}"
    )

    # The largest separation peak corresponds to m=1.
    # (Normalised kernel may show sub-peaks from higher orders packed into m=0,
    #  but the dominant outlier is always m=1 for order_max=1.)
    measured_sep_px = float(measured_seps[-1])

    err_vs_formula = abs(measured_sep_px - EXPECTED_SEP_PX)
    err_vs_helper  = abs(measured_sep_px - helper_sep_px)
    kernel_sum     = float(kernel.sum())

    # -------------------------------------------------------------------
    # 4. Assertions.
    # -------------------------------------------------------------------
    assert abs(kernel_sum - 1.0) < 1e-4, (
        f"Kernel does not sum to 1: sum = {kernel_sum:.6f}"
    )
    assert err_vs_formula <= TOLERANCE_PX, (
        f"m=1 peak offset {measured_sep_px:.2f} px deviates from "
        f"formula prediction {EXPECTED_SEP_PX:.2f} px by "
        f"{err_vs_formula:.3f} px  (tolerance {TOLERANCE_PX} px)"
    )
    assert err_vs_helper <= TOLERANCE_PX, (
        f"m=1 peak offset {measured_sep_px:.2f} px deviates from "
        f"helper prediction {helper_sep_px:.2f} px by "
        f"{err_vs_helper:.3f} px  (tolerance {TOLERANCE_PX} px)"
    )

    # -------------------------------------------------------------------
    # 5. Figure: 2-D kernel + 1-D cross-section.
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Test 1 — Analytic grating PSF: diffraction order placement\n"
        f"f={FOCAL_MM} mm  f/{F_NUMBER}  pix={PIXEL_UM} µm  "
        f"{LINES_PER_MM:.0f} lp/mm  λ={LAMBDA_EFF_NM:.0f} nm  "
        f"angle={MASK_ANGLE_DEG}°",
        fontsize=11,
    )

    # --- left: 2-D kernel (asinh stretch) ---
    ax0 = axes[0]
    vmax = float(np.percentile(kernel, 99.9))
    disp = np.arcsinh(kernel / max(vmax, 1e-30))
    im = ax0.imshow(disp, origin="lower", cmap="inferno", interpolation="nearest")
    ax0.set_title("2-D PSF kernel (asinh stretch)")
    ax0.set_xlabel("pixel (x / col)")
    ax0.set_ylabel("pixel (y / row)")

    # mark expected order positions
    for m, dx, dy in offsets:
        for sign in (+1, -1):
            ax0.axvline(cx + sign * dx, color="cyan", lw=1.2, ls="--",
                        label=f"predicted m={m}" if sign == 1 else None)
    ax0.axvline(cx, color="lime", lw=1.2, ls=":", label="m=0 (centre)")
    ax0.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax0, fraction=0.046, label="asinh(kernel / peak)")

    # --- right: 1-D cross-section ---
    ax1 = axes[1]
    x_px = np.arange(kx) - cx   # pixel offset from centre
    ax1.plot(x_px, profile, color="steelblue", lw=1.4, label="kernel row at cy")
    ax1.scatter(peak_idx - cx, profile[peak_idx], color="red", zorder=5,
                s=60, label="detected peaks")
    for m, dx, dy in offsets:
        for sign in (+1, -1):
            ax1.axvline(sign * dx, color="orange", lw=1.2, ls="--",
                        label=f"predicted ±{m} ({sign*dx:+.1f} px)" if sign == 1 else None)
    ax1.axvline(0, color="lime", lw=1.0, ls=":", label="m=0")
    ax1.set_title(
        f"1-D cross-section at row {cy}\n"
        f"predicted sep = {EXPECTED_SEP_PX:.1f} px  |  "
        f"measured sep = {measured_sep_px:.1f} px  |  "
        f"error = {err_vs_formula:.2f} px"
    )
    ax1.set_xlabel("pixel offset from centre")
    ax1.set_ylabel("kernel value")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-int(EXPECTED_SEP_PX * 1.4), int(EXPECTED_SEP_PX * 1.4))

    plt.tight_layout()
    fig_path = HERE / "test_psf_order_placement.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------
    # 6. Text summary.
    # -------------------------------------------------------------------
    lines = [
        "Test 1: PSF Diffraction Order Placement (Analytic Grating)",
        "=" * 60,
        "",
        "Setup",
        f"  focal_mm         = {FOCAL_MM}",
        f"  f_number         = {F_NUMBER}",
        f"  pixel_um         = {PIXEL_UM}",
        f"  lines_per_mm     = {LINES_PER_MM}",
        f"  lambda_eff_nm    = {LAMBDA_EFF_NM}",
        f"  band_nm          = {BAND_NM}  (monochromatic)",
        f"  mask_angle_deg   = {MASK_ANGLE_DEG}",
        f"  order_max        = {ORDER_MAX}",
        "",
        "Prediction",
        f"  Formula:  sep_px = f * m * λ / (pitch * pix)",
        f"          = {FOCAL_MM} * 1 * {LAMBDA_EFF_NM}e-6 / (1/{LINES_PER_MM} * {PIXEL_UM}e-3)",
        f"          = {EXPECTED_SEP_PX:.4f} px",
        f"  Helper (_diffraction_order_offsets_px): {helper_sep_px:.4f} px",
        "",
        "Kernel properties",
        f"  Shape:  {ky} x {kx}  (centre at row {cy}, col {cx})",
        f"  Sum:    {kernel_sum:.8f}  (expected 1.0)",
        "",
        "Peak measurement (1-D cross-section along dispersion axis)",
        f"  Detected peak indices: {list(peak_idx)}",
        f"  Measured m=1 separation: {measured_sep_px:.3f} px",
        "",
        "Results",
        f"  Error vs formula:  {err_vs_formula:.3f} px  (tolerance {TOLERANCE_PX} px)  "
        + ("[PASS]" if err_vs_formula <= TOLERANCE_PX else "[FAIL]"),
        f"  Error vs helper:   {err_vs_helper:.3f} px  (tolerance {TOLERANCE_PX} px)  "
        + ("[PASS]" if err_vs_helper <= TOLERANCE_PX else "[FAIL]"),
        f"  Kernel normalised: |sum - 1| = {abs(kernel_sum - 1.0):.2e}  "
        + ("[PASS]" if abs(kernel_sum - 1.0) < 1e-4 else "[FAIL]"),
        "",
        "ALL ASSERTIONS PASSED",
    ]
    txt_path = HERE / "test_psf_order_placement.txt"
    #txt_path.write_text("\n".join(lines))
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nFigure saved -> {fig_path.name}")
    print(f"Summary saved -> {txt_path.name}")


if __name__ == "__main__":
    run_test()
