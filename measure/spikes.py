# measure/spikes.py
# Diffraction stripe angle measurement via ensemble line-segment fitting.
#
# Angle convention: 0° = horizontal, positive = CCW, range (-90°, 90°].
# Measured in image display coordinates (origin='lower', y-axis up).
#
# Ported from notebooks/dev/angle_dev.py.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from .types import BranchImageResult, MeasurementMetadata, SpikeMeasurementResult


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def _norm_angle(a: float) -> float:
    a = float(a) % 180.0
    return a - 180.0 if a > 90.0 else a


def _wrap_diff(a: float, ref: float) -> float:
    return float((a - ref + 90.0) % 180.0 - 90.0)


def _downsample(img: np.ndarray, max_side: int = 1200) -> np.ndarray:
    factor = max(1, max(img.shape) // max_side)
    return img[::factor, ::factor].copy() if factor > 1 else img


# ---------------------------------------------------------------------------
# Feature image preparation
# ---------------------------------------------------------------------------

def _suppress_stars(image: np.ndarray, size: int = 7) -> np.ndarray:
    return median_filter(image.astype(np.float32), size=size)


def _prepare_feature_image(
    stripe_image: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Downsample → star suppress → local background subtract.

    Returns (feature_img, downsample_factor).  factor is needed to scale
    the stripe mask back to original image coordinates.
    """
    factor = max(1, max(stripe_image.shape) // 1200)
    img = stripe_image.astype(np.float32)[::factor, ::factor].copy() if factor > 1 else stripe_image.astype(np.float32)
    star_suppressed = _suppress_stars(img, size=7)
    bg = gaussian_filter(star_suppressed, sigma=30.0)
    return np.clip(star_suppressed - bg, 0.0, None), factor


def _upscale_mask(
    mask_small: np.ndarray,
    factor: int,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Scale a boolean mask from feature-image coords back to original image shape."""
    if factor == 1:
        return mask_small[:target_shape[0], :target_shape[1]]
    big = np.kron(mask_small, np.ones((factor, factor), dtype=bool))
    h, w = target_shape
    return big[:h, :w]


# ---------------------------------------------------------------------------
# Ensemble estimator internals
# ---------------------------------------------------------------------------

def _orientation_to_angle(orientation_rad: float) -> float:
    """skimage regionprops orientation → our angle convention."""
    return _norm_angle(float(np.degrees(orientation_rad)) - 90.0)


def _extract_candidates(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    from skimage.measure import label, regionprops

    nz = feature_img[feature_img > 0]
    thresh = float(np.percentile(nz, threshold_pct)) if nz.size else 0.5
    label_img = label(feature_img >= thresh)

    candidates: List[Dict[str, Any]] = []
    for region in regionprops(label_img, intensity_image=feature_img):
        if region.num_pixels < min_pixels:
            continue
        candidates.append({
            "label":           region.label,
            "angle_deg":       _orientation_to_angle(region.orientation),
            "major_length":    float(region.major_axis_length),
            "eccentricity":    float(region.eccentricity),
            "mean_brightness": float(region.mean_intensity),
            "n_pixels":        region.num_pixels,
            "centroid":        region.centroid,
        })
    return candidates, label_img


def _filter_candidates(
    candidates: List[Dict[str, Any]],
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
) -> List[Dict[str, Any]]:
    return [
        c for c in candidates
        if c["major_length"] >= min_major_length
        and c["eccentricity"] >= min_eccentricity
    ]


def _weight(c: Dict[str, Any]) -> float:
    return float(c["major_length"] * c["eccentricity"] * c["mean_brightness"])


def _weighted_axial_mean(
    angles_deg: List[float], weights: List[float]
) -> Tuple[float, float]:
    theta = 2.0 * np.radians(angles_deg)
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum() + 1e-10
    s, c = float(np.dot(w, np.sin(theta))), float(np.dot(w, np.cos(theta)))
    return _norm_angle(float(np.degrees(np.arctan2(s, c))) / 2.0), float(np.sqrt(s**2 + c**2))


def _weighted_sigma(
    angles_deg: List[float], weights: List[float], mean_deg: float
) -> float:
    diffs = (np.asarray(angles_deg) - mean_deg + 90.0) % 180.0 - 90.0
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum() + 1e-10
    return float(np.sqrt(np.dot(w, diffs**2)))


def _sigma_clip(
    accepted: List[Dict[str, Any]],
    sigma_clip: float = 2.5,
    max_iter: int = 3,
    min_keep: int = 3,
) -> List[Dict[str, Any]]:
    kept = list(accepted)
    for _ in range(max_iter):
        if len(kept) < min_keep:
            break
        angles  = [c["angle_deg"] for c in kept]
        weights = [_weight(c) for c in kept]
        mean, _ = _weighted_axial_mean(angles, weights)
        sigma   = max(_weighted_sigma(angles, weights, mean), 0.5)
        new = [c for c in kept if abs(_wrap_diff(c["angle_deg"], mean)) <= sigma_clip * sigma]
        if len(new) == len(kept) or len(new) < min_keep:
            break
        kept = new
    return kept


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def measure_ensemble_lines(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
    min_accepted: int = 2,
    sigma_clip: float = 2.5,
    clip_max_iter: int = 3,
) -> Tuple[float, float, float, bool, int, int]:
    """Run ensemble estimator and return (angle_deg, sigma_deg, quality, valid, n_accepted, n_detected)."""
    candidates, _ = _extract_candidates(feature_img, threshold_pct, min_pixels)
    accepted = _filter_candidates(candidates, min_major_length, min_eccentricity)

    if sigma_clip > 0 and len(accepted) >= max(min_accepted, 3):
        accepted = _sigma_clip(accepted, sigma_clip, clip_max_iter, max(min_accepted, 3))

    n_det, n_acc = len(candidates), len(accepted)
    if n_acc < min_accepted:
        return 0.0, 180.0, 0.0, False, n_acc, n_det

    angles  = [c["angle_deg"] for c in accepted]
    weights = [_weight(c) for c in accepted]
    mean, R = _weighted_axial_mean(angles, weights)
    sigma   = _weighted_sigma(angles, weights, mean)
    quality = R * min(1.0, n_acc / 5.0)
    valid   = quality > 0.3 and n_acc >= min_accepted
    return mean, sigma, quality, valid, n_acc, n_det


def measure_diffraction_angle(
    branch: BranchImageResult,
    meta: MeasurementMetadata,
    save_dir: Optional[Path] = None,
) -> SpikeMeasurementResult:
    """Estimate diffraction stripe angle and build a stripe pixel mask.

    Uses the ensemble line-segment method: threshold → connected components
    → PCA per component → weighted circular mean with sigma-clipping.

    Stores ``res.debug["stripe_mask"]`` — a boolean array in the original
    image coordinate frame marking pixels that belong to accepted stripe
    segments.  The pipeline uses this to blank stripes before platesolving.
    """
    res = SpikeMeasurementResult()

    if branch.image is None or branch.image.max() == 0:
        res.messages.append("stripe image empty; skipping angle measurement")
        res.debug["stripe_mask"] = np.zeros(branch.image.shape if branch.image is not None else (1, 1), dtype=bool)
        return res

    feature_img, factor = _prepare_feature_image(branch.image)

    # Run internals directly so we keep label_img for mask building
    candidates, label_img = _extract_candidates(feature_img)
    accepted = _filter_candidates(candidates)
    if len(accepted) >= 3:
        accepted = _sigma_clip(accepted)

    n_det, n_acc = len(candidates), len(accepted)

    # Build stripe mask in original image coordinates
    mask_small = np.zeros(feature_img.shape, dtype=bool)
    for c in accepted:
        mask_small[label_img == c["label"]] = True
    res.debug["stripe_mask"] = _upscale_mask(mask_small, factor, branch.image.shape)
    res.debug["feature_img"] = feature_img

    angle, sigma, quality, valid = 0.0, 180.0, 0.0, False
    if n_acc >= 2:
        angles  = [c["angle_deg"] for c in accepted]
        weights = [_weight(c) for c in accepted]
        angle, R = _weighted_axial_mean(angles, weights)
        sigma    = _weighted_sigma(angles, weights, angle)
        quality  = R * min(1.0, n_acc / 5.0)
        valid    = quality > 0.3

    res.n_detected = n_det
    res.n_accepted = n_acc
    res.quality    = quality
    res.messages.append(
        f"ensemble: {n_acc}/{n_det} segments accepted, "
        f"angle={angle:.2f}° ± {sigma:.2f}°, quality={quality:.3f}, "
        f"{'valid' if valid else 'INVALID'}"
    )

    if valid:
        res.success         = True
        res.image_angle_deg = angle
        res.sigma_angle_deg = sigma

    if save_dir is not None:
        _save_angle_plot(branch.image, angle if valid else None, sigma, quality,
                         save_path=save_dir / "stripe_angle.png")

    return res


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _save_angle_plot(
    image: np.ndarray,
    angle_deg: Optional[float],
    sigma_deg: float,
    quality: float,
    save_path: Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = image[np.isfinite(image)]
    vmin, vmax = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    if angle_deg is not None:
        cy, cx = image.shape[0] / 2.0, image.shape[1] / 2.0
        L = float(max(image.shape))
        tr = np.radians(angle_deg)
        ax.plot(
            [cx - np.cos(tr) * L, cx + np.cos(tr) * L],
            [cy + np.sin(tr) * L, cy - np.sin(tr) * L],
            color="lime", linewidth=1.8, alpha=0.9,
        )
        label = f"{angle_deg:.2f}° ± {sigma_deg:.2f}°   Q={quality:.3f}"
        color = "lime"
    else:
        label = f"No valid solution  (Q={quality:.3f})"
        color = "red"

    ax.text(0.02, 0.97, label, transform=ax.transAxes, color=color, fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=3))
    ax.set_title("Stripe branch — ensemble angle")
    ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
