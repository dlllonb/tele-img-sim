# measure/spikes.py
# diffraction stripe angle measurement via Radon, Hough, and Fourier methods
#
# Angle convention: 0° = horizontal (left-right), positive = CCW, range (-90°, 90°].
# Measured in image display coordinates (origin='lower', y-axis up).

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .types import BranchImageResult, MeasurementMetadata, SpikeMeasurementResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_angle(a: float) -> float:
    """Wrap angle to (-90°, 90°]."""
    a = float(a) % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def _downsample(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Downsample by integer factor so the longest side <= max_side."""
    factor = max(1, max(img.shape) // max_side)
    return img[::factor, ::factor].copy() if factor > 1 else img


def _peak_sigma(values: np.ndarray, x: np.ndarray, peak_idx: int) -> float:
    """Estimate 1-sigma peak width via HWHM."""
    half = (values[peak_idx] + values.mean()) * 0.5
    left, right = peak_idx, peak_idx
    while left > 0 and values[left] >= half:
        left -= 1
    while right < len(values) - 1 and values[right] >= half:
        right += 1
    fwhm = float(x[right] - x[left]) if right > left else float(x[1] - x[0]) * 2
    return fwhm / 2.355


def _draw_angle_plot(
    image: np.ndarray,
    angle_deg: float,
    sigma_deg: float,
    method: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Display the stripe image with a red line at the measured angle."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = image[np.isfinite(image)]
    vmin, vmax = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    cy, cx = image.shape[0] / 2.0, image.shape[1] / 2.0
    L = float(max(image.shape))
    tr = np.radians(angle_deg)
    dx, dy = np.cos(tr) * L, np.sin(tr) * L
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            color="red", linewidth=1.5, alpha=0.9)

    ax.text(0.02, 0.97,
            f"{method}:  {angle_deg:.2f}° ± {sigma_deg:.2f}°",
            transform=ax.transAxes, color="red", fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=3))
    ax.set_title(f"Stripe angle — {method}")
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── three methods ─────────────────────────────────────────────────────────────
# Each stub returns (angle_deg, sigma_deg) = (0.0, 0.0) as a placeholder.
# Real implementations to be developed interactively in the notebook.

def _measure_radon(image: np.ndarray) -> Tuple[float, float]:  # noqa: ARG001
    """Stub — Radon transform angle estimate (not yet implemented)."""
    del image
    return 0.0, 0.0


def _measure_hough(image: np.ndarray) -> Tuple[float, float]:  # noqa: ARG001
    """Stub — Hough transform angle estimate (not yet implemented)."""
    del image
    return 0.0, 0.0


def _measure_fourier(image: np.ndarray) -> Tuple[float, float]:  # noqa: ARG001
    """Stub — Fourier angular power spectrum angle estimate (not yet implemented)."""
    del image
    return 0.0, 0.0


# ── public entry point ────────────────────────────────────────────────────────

def measure_diffraction_angle(
    branch: BranchImageResult,
    meta: MeasurementMetadata,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> SpikeMeasurementResult:
    """Run Radon, Hough, and Fourier angle estimates on the stripe branch.

    Consensus angle is the median of all successful method estimates.
    Combined sigma combines inter-method spread with per-method uncertainties.
    Plots with a red angle overlay are saved to save_dir and/or shown.
    """
    res = SpikeMeasurementResult()

    if branch.image is None or branch.image.max() == 0:
        res.messages.append("stripe image empty; skipping angle measurement")
        return res

    image = branch.image

    methods = [
        ("radon",   _measure_radon),
        ("hough",   _measure_hough),
        ("fourier", _measure_fourier),
    ]

    angles: list[float] = []
    sigmas: list[float] = []

    for name, fn in methods:
        try:
            angle, sigma = fn(image)
            setattr(res, f"{name}_angle_deg", angle)
            setattr(res, f"{name}_sigma_deg", sigma)
            angles.append(angle)
            sigmas.append(sigma)
            res.messages.append(f"{name}: {angle:.2f}° ± {sigma:.2f}°")
            _draw_angle_plot(
                image, angle, sigma, name,
                save_path=save_dir / f"spikes_{name}.png" if save_dir else None,
                show=show,
            )
        except Exception as e:
            res.messages.append(f"{name} failed: {e}")

    if angles:
        res.image_angle_deg = float(np.median(angles))
        spread = float(np.std(angles)) if len(angles) > 1 else 0.0
        res.sigma_angle_deg = float(np.sqrt(spread ** 2 + np.mean(sigmas) ** 2))
        res.success = True
        res.messages.append(
            f"consensus: {res.image_angle_deg:.2f}° ± {res.sigma_angle_deg:.2f}°"
        )

    return res
