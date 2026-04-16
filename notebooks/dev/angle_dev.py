# notebooks/dev/angle_dev.py
#
# Interactive development of diffraction stripe angle extraction.
# Designed for portability into measure/preprocess.py and measure/spikes.py.
#
# Angle convention: 0° = horizontal, positive = CCW, range (-90°, 90°].
# Measured in image display coordinates (origin='lower', y-axis up).

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


# ── result type ───────────────────────────────────────────────────────────────

class AngleEstimate(NamedTuple):
    """Return type for measurement methods.

    Supports unpacking as `angle, sigma = est` for backward compat, and
    carries quality metadata for reporting / pipeline quality flags.
    """
    angle_deg: float
    sigma_deg: float
    quality: float   # confidence metric; higher = better
    valid: bool      # True when quality >= method's min_quality threshold
    method: str = ""


# ── path bootstrap ─────────────────────────────────────────────────────────────

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "sim").is_dir():
            return p
    raise RuntimeError("Could not locate project root (no sim/ directory found)")


# ── data loading ───────────────────────────────────────────────────────────────

def load_and_preprocess(fits_path: "str | Path") -> Tuple[np.ndarray, np.ndarray]:
    """Load FITS → run pipeline stripe preprocessing → return (raw, stripe_image)."""
    import sys
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from astropy.io import fits as astrofits
    from measure.preprocess import prepare_stripe_branch_input
    from measure.types import MeasurementMetadata

    fits_path = Path(fits_path)
    if not fits_path.is_absolute():
        fits_path = root / fits_path

    with astrofits.open(fits_path) as hdul:
        raw = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    meta = MeasurementMetadata.from_header(header)
    branch = prepare_stripe_branch_input(raw, meta)
    return raw, branch.image


# ── shared helpers ─────────────────────────────────────────────────────────────

def _norm_angle(a: float) -> float:
    """Wrap angle to (-90°, 90°]."""
    a = float(a) % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def _downsample(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Integer-stride downsample so the longest side <= max_side."""
    factor = max(1, max(img.shape) // max_side)
    return img[::factor, ::factor].copy() if factor > 1 else img


def _make_soft_mask(shape: Tuple[int, int],
                    roi_fraction: float = 0.80,
                    apodize_frac: float = 0.10) -> np.ndarray:
    """Circular soft mask: 1 inside roi, cosine taper at the edge, 0 outside."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    r_max = min(cy, cx) * roi_fraction
    r_inner = r_max * (1.0 - apodize_frac)
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    mask = np.ones((h, w), dtype=np.float32)
    taper = (r >= r_inner) & (r <= r_max)
    t = (r[taper] - r_inner) / (r_max - r_inner)
    mask[taper] = 0.5 * (1.0 + np.cos(np.pi * t))
    mask[r > r_max] = 0.0
    return mask


def _draw_line(ax, image: np.ndarray, angle_deg: float,
               color: str = "red", lw: float = 1.5,
               cx: Optional[float] = None, cy: Optional[float] = None,
               length: Optional[float] = None) -> None:
    """Draw a line at angle_deg through (cx,cy) with given length."""
    if cy is None:
        cy = image.shape[0] / 2.0
    if cx is None:
        cx = image.shape[1] / 2.0
    if length is None:
        length = float(max(image.shape))
    tr = np.radians(angle_deg)
    ax.plot([cx - np.cos(tr) * length, cx + np.cos(tr) * length],
        [cy + np.sin(tr) * length, cy - np.sin(tr) * length],
        color=color, linewidth=lw, alpha=0.9)


def _imshow_pct(ax, img: np.ndarray, cmap: str = "gray",
                lo: float = 0.5, hi: float = 99.5) -> None:
    finite = img[np.isfinite(img)]
    vmin, vmax = (np.percentile(finite, [lo, hi]) if finite.size else (0.0, 1.0))
    ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")


# ── shared preprocessing ───────────────────────────────────────────────────────

def _suppress_stars(image: np.ndarray, star_size: int) -> np.ndarray:
    """Median filter: replaces compact bright stars with local background."""
    return median_filter(image.astype(np.float32), size=star_size)


def _enhance_ridges(image: np.ndarray,
                    sigmas: Tuple[float, ...] = (2.0, 4.0, 8.0)) -> np.ndarray:
    """Frangi vesselness: amplifies elongated ridges, suppresses blobs."""
    from skimage.filters import frangi
    img = image.astype(np.float64)
    denom = img.max() - img.min()
    img = (img - img.min()) / (denom + 1e-10)
    return frangi(img, sigmas=sigmas, black_ridges=False).astype(np.float32)


def prepare_angle_feature_image(
    image: np.ndarray,
    roi_fraction: float = 0.80,
    apodize_frac: float = 0.10,
    star_size: int = 7,
    bg_sigma: float = 30.0,
    ridge_sigmas: Tuple[float, ...] = (2.0, 4.0, 8.0),
    max_side: int = 1200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Isolate stripe-like features from the pipeline stripe image.

    Pipeline: downsample → median star-suppression → Gaussian bg removal
              → Frangi ridge enhancement → soft circular ROI mask.

    Returns (feature_img, debug_info). Portable to measure/preprocess.py.
    """
    img = _downsample(image.astype(np.float32), max_side=max_side)
    star_suppressed = _suppress_stars(img, star_size)
    bg = gaussian_filter(star_suppressed, sigma=bg_sigma)
    bg_subtracted = np.clip(star_suppressed - bg, 0.0, None)
    ridge_response = _enhance_ridges(bg_subtracted, sigmas=ridge_sigmas)
    soft_mask = _make_soft_mask(img.shape, roi_fraction=roi_fraction,
                                apodize_frac=apodize_frac)
    feature_img = (ridge_response * soft_mask).astype(np.float32)

    debug_info = {
        "downsampled":     img,
        "soft_mask":       soft_mask,
        "star_suppressed": star_suppressed,
        "bg_subtracted":   bg_subtracted,
        "ridge_response":  ridge_response,
        "feature_img":     feature_img,
    }
    return feature_img, debug_info


def show_feature_debug(stripe: np.ndarray, debug_info: Dict[str, Any]) -> None:
    """Show all preprocessing pipeline stages side by side."""
    import matplotlib.pyplot as plt

    stages = [
        ("soft_mask",      "Soft ROI mask",          "gray",    (0.0, 1.0)),
        ("downsampled",    "Downsampled stripe",      "gray",    None),
        ("star_suppressed","Star-suppressed\n(median)","gray",   None),
        ("bg_subtracted",  "BG-subtracted",           "gray",    None),
        ("ridge_response", "Ridge response\n(Frangi)","inferno", None),
        ("feature_img",    "Feature image\n(masked)", "inferno", None),
    ]
    fig, axes = plt.subplots(1, len(stages), figsize=(24, 4))
    for ax, (key, title, cmap, fixed_range) in zip(axes, stages):
        img = debug_info[key]
        if fixed_range is not None:
            vmin, vmax = fixed_range
            ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            _imshow_pct(ax, img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle("Preprocessing stages", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── ensemble line estimator ────────────────────────────────────────────────────
#
# Strategy: threshold feature image → connected components → fit each component
# as a local line (PCA on pixel coords) → filter by elongation / length →
# combine accepted angles via weighted circular mean (axial statistics).
#
# This supports many short spectra, not just one dominant long stripe.


def _orientation_to_angle(orientation_rad: float) -> float:
    """Convert skimage regionprops orientation to our angle convention.

    skimage orientation: angle between major axis and 0th axis (rows), CCW,
    range [-π/2, π/2].  Our convention: 0° = horizontal, CCW positive.

    Derivation: our_angle = orientation_deg - 90°, then normalize.
    """
    return _norm_angle(float(np.degrees(orientation_rad)) - 90.0)


def _extract_candidates(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Threshold feature image, label connected components, fit each as a line.

    Parameters
    ----------
    threshold_pct : percentile of nonzero feature values used as binary threshold
    min_pixels    : discard components with fewer pixels than this

    Returns
    -------
    candidates : list of dicts with geometry, angle, and brightness
    label_img  : integer label array (0 = background)
    """
    from skimage.measure import label, regionprops

    nz = feature_img[feature_img > 0]
    thresh = float(np.percentile(nz, threshold_pct)) if nz.size else 0.5
    binary = (feature_img >= thresh)

    label_img = label(binary)

    candidates: List[Dict[str, Any]] = []
    for region in regionprops(label_img, intensity_image=feature_img):
        if region.num_pixels < min_pixels:
            continue
        candidates.append({
            "label":          region.label,
            "angle_deg":      _orientation_to_angle(region.orientation),
            "major_length":   float(region.major_axis_length),
            "minor_length":   float(region.minor_axis_length),
            "eccentricity":   float(region.eccentricity),
            "mean_brightness":float(region.mean_intensity),
            "n_pixels":       region.num_pixels,
            "centroid":       region.centroid,   # (row, col)
        })

    return candidates, label_img


def _filter_candidates(
    candidates: List[Dict[str, Any]],
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
    min_brightness: float = 0.0,
) -> List[Dict[str, Any]]:
    """Keep only sufficiently elongated, long, and bright candidates.

    Parameters
    ----------
    min_major_length  : minimum major axis length (px in feature_img coords)
    min_eccentricity  : minimum eccentricity (0=circle, 1=line); rejects blobs
    min_brightness    : minimum mean feature brightness; 0.0 = no brightness filter
    """
    accepted = []
    for c in candidates:
        if c["major_length"] < min_major_length:
            continue
        if c["eccentricity"] < min_eccentricity:
            continue
        if c["mean_brightness"] < min_brightness:
            continue
        accepted.append(c)
    return accepted


def _candidate_weight(c: Dict[str, Any]) -> float:
    """Scalar weight for one accepted candidate: longer, more elongated, brighter = higher."""
    return float(c["major_length"] * c["eccentricity"] * c["mean_brightness"])


def _weighted_axial_mean(
    angles_deg: List[float],
    weights: List[float],
) -> Tuple[float, float]:
    """Weighted circular mean for line/axial angles (modulo 180°).

    Uses the doubled-angle trick: map each angle θ to 2θ on the unit circle,
    compute weighted circular mean, then halve the result.

    Returns (mean_angle_deg, R) where R ∈ [0, 1] is the resultant length
    (R≈1 = highly concentrated, R≈0 = highly dispersed).
    """
    theta = 2.0 * np.radians(angles_deg)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-10)
    sin_w = float(np.dot(w, np.sin(theta)))
    cos_w = float(np.dot(w, np.cos(theta)))
    mean_angle = _norm_angle(float(np.degrees(np.arctan2(sin_w, cos_w))) / 2.0)
    R = float(np.sqrt(sin_w ** 2 + cos_w ** 2))
    return mean_angle, R


def _weighted_angle_sigma(
    angles_deg: List[float],
    weights: List[float],
    mean_angle_deg: float,
) -> float:
    """Weighted angular standard deviation around mean_angle_deg (degrees).

    Differences are wrapped to (-90°, 90°] before squaring so wrap-around
    near ±90° is handled correctly.
    """
    diffs = np.asarray(angles_deg) - mean_angle_deg
    # wrap to (-90°, 90°]
    diffs = (diffs + 90.0) % 180.0 - 90.0
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-10)
    return float(np.sqrt(np.dot(w, diffs ** 2)))


def measure_ensemble_lines(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
    min_accepted: int = 2,
) -> AngleEstimate:
    """Estimate stripe angle from an ensemble of locally-fitted line segments.

    1. Threshold → connected components → fit each as a local line (skimage PCA).
    2. Filter by elongation (eccentricity) and length.
    3. Combine accepted angles via weighted circular mean (axial statistics).
    4. Uncertainty = weighted angular dispersion of the ensemble.

    Quality = resultant length R ∈ [0,1] (concentration of the ensemble);
    R close to 1 means all accepted segments agree on orientation.
    Valid when at least min_accepted segments pass the filter.

    Portable to measure/spikes.py — no notebook-specific state.
    """
    candidates, _ = _extract_candidates(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels)

    accepted = _filter_candidates(
        candidates, min_major_length=min_major_length,
        min_eccentricity=min_eccentricity)

    n_det = len(candidates)
    n_acc = len(accepted)

    if n_acc < min_accepted:
        return AngleEstimate(0.0, 180.0, 0.0, False, "ensemble",)

    angles = [c["angle_deg"] for c in accepted]
    weights = [_candidate_weight(c) for c in accepted]

    mean_angle, R = _weighted_axial_mean(angles, weights)
    sigma = _weighted_angle_sigma(angles, weights, mean_angle)

    # quality = resultant length; penalty when very few segments
    quality = R * min(1.0, n_acc / 5.0)
    valid = (quality > 0.3) and (n_acc >= min_accepted)

    return AngleEstimate(mean_angle, sigma, quality, valid, "ensemble")


# ── ensemble debug plot ────────────────────────────────────────────────────────

def show_ensemble_debug(
    stripe: np.ndarray,
    feature_img: np.ndarray,
    debug_info: Dict[str, Any],
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
) -> None:
    """Six-panel debug: preprocessing stages + ensemble candidate details.

    Panels:
    Row 1: stripe+final_overlay | feature_img+overlay | all components (colored by angle)
    Row 2: accepted candidates with fitted lines | angle histogram | text summary
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # --- recompute to get full debug artifacts ---
    candidates, label_img = _extract_candidates(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels)
    accepted = _filter_candidates(
        candidates, min_major_length=min_major_length,
        min_eccentricity=min_eccentricity)

    est = measure_ensemble_lines(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels,
        min_major_length=min_major_length, min_eccentricity=min_eccentricity)

    # colormap: angle -90°…90° → color
    angle_cmap = cm.get_cmap("hsv")
    def angle_color(a_deg):
        return angle_cmap((a_deg + 90.0) / 180.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── panel (0,0): stripe with final angle overlay ──
    _imshow_pct(axes[0, 0], stripe)
    if est.valid:
        _draw_line(axes[0, 0], stripe, est.angle_deg, color="lime", lw=2.0)
    axes[0, 0].set_title(
        f"Stripe + result  {est.angle_deg:.2f}°\n"
        f"σ={est.sigma_deg:.2f}°  Q={est.quality:.2f}  "
        f"{'valid' if est.valid else 'INVALID'}", fontsize=9)

    # ── panel (0,1): feature image with final overlay ──
    _imshow_pct(axes[0, 1], feature_img, cmap="inferno")
    if est.valid:
        _draw_line(axes[0, 1], feature_img, est.angle_deg, color="lime", lw=1.5)
    axes[0, 1].set_title("Feature image", fontsize=9)

    # ── panel (0,2): all components colored by angle ──
    # build a pseudo-color image: each component pixel gets its angle as hue
    rgb = np.zeros((*feature_img.shape, 3), dtype=np.float32)
    for c in candidates:
        mask = label_img == c["label"]
        col = angle_color(c["angle_deg"])[:3]
        for ch in range(3):
            rgb[:, :, ch][mask] = col[ch]
    # darken background using feature magnitude
    alpha = (feature_img / (feature_img.max() + 1e-10))[:, :, np.newaxis]
    bg_gray = debug_info["downsampled"]
    bg_norm = (bg_gray / (bg_gray.max() + 1e-10))[:, :, np.newaxis]
    display = np.where(rgb.sum(axis=2, keepdims=True) > 0, rgb * alpha + bg_norm * (1 - alpha), bg_norm * np.ones(3))
    display = np.clip(display, 0, 1)
    axes[0, 2].imshow(display, origin="lower")
    axes[0, 2].set_title(
        f"All components ({len(candidates)}) — colored by angle", fontsize=9)
    axes[0, 2].axis("off")
    # colorbar for angle
    sm = cm.ScalarMappable(cmap="hsv", norm=mcolors.Normalize(-90, 90))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0, 2], label="angle (°)", fraction=0.046, pad=0.04)

    # ── panel (1,0): accepted candidates with fitted segment lines ──
    _imshow_pct(axes[1, 0], feature_img, cmap="gray")
    rejected_labels = {c["label"] for c in candidates} - {c["label"] for c in accepted}
    # draw rejected faintly
    for c in candidates:
        if c["label"] in rejected_labels:
            row, col = c["centroid"]
            axes[1, 0].scatter(col, row, s=4, c="gray", alpha=0.3, linewidths=0)
    # draw accepted with colored lines
    for c in accepted:
        row, col = c["centroid"]
        col_rgb = angle_color(c["angle_deg"])
        half = c["major_length"] / 2.0
        _draw_line(axes[1, 0], feature_img, c["angle_deg"],
                   color=col_rgb, lw=1.5,
                   cx=col, cy=row, length=half * 2)
        axes[1, 0].scatter(col, row, s=10, c=[col_rgb], zorder=5, linewidths=0)
    if est.valid:
        _draw_line(axes[1, 0], feature_img, est.angle_deg,
                   color="lime", lw=2.5)
    axes[1, 0].set_title(
        f"Accepted candidates ({len(accepted)}/{len(candidates)})\n"
        "— colored lines = local fits, lime = ensemble result", fontsize=9)

    # ── panel (1,1): weighted angle histogram ──
    if accepted:
        a_vals = [c["angle_deg"] for c in accepted]
        w_vals = [_candidate_weight(c) for c in accepted]
        w_norm = np.asarray(w_vals) / (max(w_vals) + 1e-10)
        colors = [angle_color(a) for a in a_vals]
        axes[1, 1].barh(
            range(len(a_vals)), a_vals,
            left=0,
            height=0.8,
            color=colors, alpha=0.85)
        # replot as horizontal: x=angle, y=rank sorted by angle
        # actually use a vertical bar chart with angle on x-axis
        axes[1, 1].cla()
        order = np.argsort(a_vals)
        for i, idx in enumerate(order):
            axes[1, 1].bar(a_vals[idx], w_norm[idx], width=1.5,
                           color=angle_color(a_vals[idx]), alpha=0.85)
        if est.valid:
            axes[1, 1].axvline(est.angle_deg, color="lime", lw=2.0,
                               label=f"mean={est.angle_deg:.1f}°")
            axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_xlabel("Angle (°)")
        axes[1, 1].set_ylabel("Normalised weight")
        axes[1, 1].set_title("Accepted segment angles\n(height ∝ weight)", fontsize=9)
        axes[1, 1].set_xlim(-95, 95)
    else:
        axes[1, 1].text(0.5, 0.5, "No accepted candidates",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Angle distribution", fontsize=9)

    # ── panel (1,2): text summary ──
    axes[1, 2].axis("off")
    summary = (
        f"Detected components:  {len(candidates)}\n"
        f"Accepted (filtered):  {len(accepted)}\n"
        f"\n"
        f"Filter criteria:\n"
        f"  min_major_length:   {min_major_length:.0f} px\n"
        f"  min_eccentricity:   {min_eccentricity:.2f}\n"
        f"\n"
        f"Result:\n"
        f"  angle:    {est.angle_deg:>8.3f} °\n"
        f"  σ:        {est.sigma_deg:>8.3f} °\n"
        f"  quality:  {est.quality:>8.3f}  (R·scale)\n"
        f"  valid:    {'YES' if est.valid else 'NO'}\n"
    )
    if accepted:
        angles_arr = np.array([c["angle_deg"] for c in accepted])
        summary += (
            f"\n"
            f"Ensemble spread:\n"
            f"  min angle:  {angles_arr.min():.2f}°\n"
            f"  max angle:  {angles_arr.max():.2f}°\n"
            f"  raw std:    {angles_arr.std():.2f}°\n"
        )
    axes[1, 2].text(0.05, 0.95, summary,
                    transform=axes[1, 2].transAxes,
                    fontsize=11, va="top", family="monospace")

    plt.suptitle(
        f"Ensemble estimator — {est.angle_deg:.2f}° ± {est.sigma_deg:.2f}°   "
        f"quality={est.quality:.3f}   {'VALID' if est.valid else 'INVALID'}",
        fontsize=12)
    plt.tight_layout()
    plt.show()
