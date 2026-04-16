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


# ── internal full-run helper ───────────────────────────────────────────────────

def _run_ensemble(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
    min_accepted: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], AngleEstimate]:
    """Extract + filter + fit: return (all_candidates, accepted, estimate).

    Used by sweep and field-analysis functions so we don't repeat the
    extraction logic.  Public API stays measure_ensemble_lines().
    """
    candidates, _ = _extract_candidates(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels)
    accepted = _filter_candidates(
        candidates, min_major_length=min_major_length,
        min_eccentricity=min_eccentricity)

    n_acc = len(accepted)
    if n_acc < min_accepted:
        return candidates, accepted, AngleEstimate(0.0, 180.0, 0.0, False, "ensemble")

    angles  = [c["angle_deg"] for c in accepted]
    weights = [_candidate_weight(c) for c in accepted]
    mean_angle, R = _weighted_axial_mean(angles, weights)
    sigma         = _weighted_angle_sigma(angles, weights, mean_angle)
    quality       = R * min(1.0, n_acc / 5.0)
    valid         = (quality > 0.3) and (n_acc >= min_accepted)
    return candidates, accepted, AngleEstimate(mean_angle, sigma, quality, valid, "ensemble")


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
    """Display ensemble debug output as separate figures:

    1. Stripe image with final angle overlay
    2. Feature image with final angle overlay
    3. All components colored by fitted angle
    4. Accepted candidates with local fitted lines
    5. Weighted angle histogram
    6. Text summary (printed to stdout)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # --- recompute debug artifacts ---
    candidates, label_img = _extract_candidates(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels)
    accepted = _filter_candidates(
        candidates, min_major_length=min_major_length,
        min_eccentricity=min_eccentricity)
    est = measure_ensemble_lines(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels,
        min_major_length=min_major_length, min_eccentricity=min_eccentricity)

    angle_cmap = cm.get_cmap("hsv")
    def angle_color(a_deg):
        return angle_cmap((a_deg + 90.0) / 180.0)

    title_suffix = (f"{est.angle_deg:.2f}° ± {est.sigma_deg:.2f}°  "
                    f"Q={est.quality:.2f}  "
                    f"{'VALID' if est.valid else 'INVALID'}")

    # ── figure 1: stripe with final angle overlay ──
    fig, ax = plt.subplots(figsize=(10, 7))
    _imshow_pct(ax, stripe)
    if est.valid:
        _draw_line(ax, stripe, est.angle_deg, color="lime", lw=2.0)
    ax.set_title(f"Stripe + result  {title_suffix}", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── figure 2: feature image with final overlay ──
    fig, ax = plt.subplots(figsize=(10, 7))
    _imshow_pct(ax, feature_img, cmap="inferno")
    if est.valid:
        _draw_line(ax, feature_img, est.angle_deg, color="lime", lw=1.5)
    ax.set_title(f"Feature image  {title_suffix}", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── figure 3: all components colored by angle ──
    rgb = np.zeros((*feature_img.shape, 3), dtype=np.float32)
    for c in candidates:
        mask = label_img == c["label"]
        col = angle_color(c["angle_deg"])[:3]
        for ch in range(3):
            rgb[:, :, ch][mask] = col[ch]
    alpha = (feature_img / (feature_img.max() + 1e-10))[:, :, np.newaxis]
    bg_gray = debug_info["downsampled"]
    bg_norm = (bg_gray / (bg_gray.max() + 1e-10))[:, :, np.newaxis]
    display = np.where(
        rgb.sum(axis=2, keepdims=True) > 0,
        rgb * alpha + bg_norm * (1 - alpha),
        bg_norm * np.ones(3))
    display = np.clip(display, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(display, origin="lower")
    ax.set_title(f"All components ({len(candidates)}) — colored by fitted angle", fontsize=11)
    ax.axis("off")
    sm = cm.ScalarMappable(cmap="hsv", norm=mcolors.Normalize(-90, 90))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="angle (°)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # ── figure 4: accepted candidates with fitted lines ──
    fig, ax = plt.subplots(figsize=(10, 7))
    _imshow_pct(ax, feature_img, cmap="gray")
    rejected_labels = {c["label"] for c in candidates} - {c["label"] for c in accepted}
    for c in candidates:
        if c["label"] in rejected_labels:
            row, col = c["centroid"]
            ax.scatter(col, row, s=4, c="gray", alpha=0.3, linewidths=0)
    for c in accepted:
        row, col = c["centroid"]
        col_rgb = angle_color(c["angle_deg"])
        _draw_line(ax, feature_img, c["angle_deg"],
                   color=col_rgb, lw=1.5, cx=col, cy=row,
                   length=c["major_length"])
        ax.scatter(col, row, s=10, c=[col_rgb], zorder=5, linewidths=0)
    if est.valid:
        _draw_line(ax, feature_img, est.angle_deg, color="lime", lw=2.5)
    ax.set_title(
        f"Accepted candidates ({len(accepted)}/{len(candidates)})  "
        f"colored=local fits  lime=ensemble result", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── figure 5: weighted angle histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    if accepted:
        a_vals = [c["angle_deg"] for c in accepted]
        w_vals = [_candidate_weight(c) for c in accepted]
        w_norm = np.asarray(w_vals) / (max(w_vals) + 1e-10)
        for a, w in sorted(zip(a_vals, w_norm)):
            ax.bar(a, w, width=1.5, color=angle_color(a), alpha=0.85)
        if est.valid:
            ax.axvline(est.angle_deg, color="lime", lw=2.0,
                       label=f"ensemble mean = {est.angle_deg:.1f}°")
            ax.legend(fontsize=10)
        ax.set_xlabel("Angle (°)", fontsize=11)
        ax.set_ylabel("Normalised weight", fontsize=11)
        ax.set_xlim(-95, 95)
    else:
        ax.text(0.5, 0.5, "No accepted candidates",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Accepted segment angles  (bar height ∝ weight)", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── output 6: text summary ──
    angles_arr = np.array([c["angle_deg"] for c in accepted]) if accepted else np.array([])
    print(
        f"─── Ensemble estimator ───────────────────────────\n"
        f"  Detected components : {len(candidates)}\n"
        f"  Accepted (filtered) : {len(accepted)}\n"
        f"  Filter criteria     : min_major_length={min_major_length:.0f} px"
        f",  min_eccentricity={min_eccentricity:.2f}\n"
        f"\n"
        f"  angle   : {est.angle_deg:.3f} °\n"
        f"  σ       : {est.sigma_deg:.3f} °\n"
        f"  quality : {est.quality:.3f}  (R · scale)\n"
        f"  valid   : {'YES' if est.valid else 'NO'}\n"
        + (f"\n"
           f"  Ensemble spread:\n"
           f"    min={angles_arr.min():.2f}°  max={angles_arr.max():.2f}°"
           f"  raw std={angles_arr.std():.2f}°\n"
           if angles_arr.size else "")
        + f"──────────────────────────────────────────────────"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — Parameter sweep
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_ensemble_params(
    feature_img: np.ndarray,
    threshold_pcts: Tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0),
    min_pixels_vals: Tuple[int, ...] = (20, 40, 60),
    min_major_lengths: Tuple[float, ...] = (15.0, 25.0, 40.0),
    min_eccentricities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    min_accepted_for_ranking: int = 3,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate many ensemble-parameter combinations and rank by a composite score.

    Optimization: candidate extraction (the expensive label+regionprops step)
    is cached per unique (threshold_pct, min_pixels) pair, so filtering over
    different length/eccentricity thresholds reuses the same components.

    Ranking score:
        score = quality × sqrt(n_accepted) / (sigma_deg + 0.5)
    Rewards: high quality, enough accepted candidates, low sigma.
    Invalid results and those with < min_accepted_for_ranking get score = 0.

    Returns a list of dicts sorted best-first.  Each dict contains:
      threshold_pct, min_pixels, min_major_length, min_eccentricity,
      angle_deg, sigma_deg, quality, valid,
      n_detected, n_accepted, acceptance_frac, raw_std_deg, rank_score.
    """
    import itertools

    combos = list(itertools.product(
        threshold_pcts, min_pixels_vals, min_major_lengths, min_eccentricities))
    if verbose:
        print(f"Sweeping {len(combos)} combinations "
              f"({len(threshold_pcts)}×{len(min_pixels_vals)}×"
              f"{len(min_major_lengths)}×{len(min_eccentricities)})…")

    # Cache extraction results for each (threshold_pct, min_pixels) pair
    cache: Dict[Tuple, List[Dict]] = {}
    for thresh, mpix in itertools.product(threshold_pcts, min_pixels_vals):
        key = (thresh, mpix)
        if key not in cache:
            cands, _ = _extract_candidates(
                feature_img, threshold_pct=thresh, min_pixels=mpix)
            cache[key] = cands

    rows = []
    for thresh, mpix, mlen, mecc in combos:
        all_c    = cache[(thresh, mpix)]
        accepted = _filter_candidates(all_c, min_major_length=mlen,
                                      min_eccentricity=mecc)
        n_det = len(all_c)
        n_acc = len(accepted)

        # ensemble angle from accepted candidates
        if n_acc >= 2:
            angles  = [c["angle_deg"] for c in accepted]
            weights = [_candidate_weight(c) for c in accepted]
            mean_angle, R = _weighted_axial_mean(angles, weights)
            sigma         = _weighted_angle_sigma(angles, weights, mean_angle)
            quality       = R * min(1.0, n_acc / 5.0)
            valid         = (quality > 0.3)
            raw_std       = float(np.std(angles))
        else:
            mean_angle = 0.0
            sigma      = 180.0
            quality    = 0.0
            valid      = False
            raw_std    = float("nan")

        acc_frac   = n_acc / n_det if n_det > 0 else 0.0
        rank_score = (
            quality * (n_acc ** 0.5) / (sigma + 0.5)
            if valid and n_acc >= min_accepted_for_ranking else 0.0
        )

        rows.append({
            "threshold_pct":    thresh,
            "min_pixels":       mpix,
            "min_major_length": mlen,
            "min_eccentricity": mecc,
            "angle_deg":        mean_angle,
            "sigma_deg":        sigma,
            "quality":          quality,
            "valid":            valid,
            "n_detected":       n_det,
            "n_accepted":       n_acc,
            "acceptance_frac":  acc_frac,
            "raw_std_deg":      raw_std,
            "rank_score":       rank_score,
        })

    rows.sort(key=lambda r: r["rank_score"], reverse=True)

    if verbose:
        n_valid = sum(r["valid"] for r in rows)
        print(f"Done.  {n_valid}/{len(rows)} valid.  "
              f"Best score: {rows[0]['rank_score']:.3f}  "
              f"→ angle={rows[0]['angle_deg']:.2f}°  σ={rows[0]['sigma_deg']:.2f}°")

    return rows


def print_sweep_results(results: List[Dict[str, Any]], top_n: int = 20) -> None:
    """Pretty-print the top_n rows from sweep_ensemble_params."""
    hdr = (f"{'#':>3}  {'thr':>4}  {'mpx':>3}  {'mlen':>4}  {'mecc':>4}  "
           f"{'angle':>7}  {'sigma':>6}  {'qual':>5}  {'V':>1}  "
           f"{'nacc':>4}  {'ndet':>4}  {'acc%':>4}  {'score':>7}")
    print(hdr)
    print("─" * len(hdr))
    for i, r in enumerate(results[:top_n]):
        print(
            f"{i+1:>3}  {r['threshold_pct']:>4.0f}  {r['min_pixels']:>3}  "
            f"{r['min_major_length']:>4.0f}  {r['min_eccentricity']:>4.2f}  "
            f"{r['angle_deg']:>7.2f}  {r['sigma_deg']:>6.3f}  "
            f"{r['quality']:>5.3f}  {'Y' if r['valid'] else 'n':>1}  "
            f"{r['n_accepted']:>4}  {r['n_detected']:>4}  "
            f"{r['acceptance_frac']*100:>4.1f}  {r['rank_score']:>7.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — Field angle variation analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_field_angle_variation(
    feature_img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_major_length: float = 15.0,
    min_eccentricity: float = 0.85,
) -> Tuple[List[Dict[str, Any]], AngleEstimate]:
    """Return a per-candidate table annotated with spatial position and residuals.

    Centroids are in feature_img pixel coords (col=x increases right,
    row=y with origin='lower' so row 0 is display-bottom).
    Normalized coords scale to [-1, 1] over the smaller half-dimension.

    Each row in the returned list contains:
      cx_px, cy_px          : centroid in feature_img array pixels
      cx_norm, cy_norm       : normalized position (0 = centre, ±1 = roi edge)
      radius_norm            : radial distance from image centre
      azimuth_deg            : azimuthal angle around field centre (E-of-N)
      angle_deg              : locally fitted line angle
      residual_deg           : angle − global ensemble mean (wrapped to (−90°, 90°])
      major_length           : major axis length (px)
      eccentricity
      mean_brightness
      weight                 : _candidate_weight value
    """
    all_c, accepted, est = _run_ensemble(
        feature_img, threshold_pct=threshold_pct, min_pixels=min_pixels,
        min_major_length=min_major_length, min_eccentricity=min_eccentricity)

    if not accepted:
        return [], est

    H, W = feature_img.shape
    cy_img, cx_img = H / 2.0, W / 2.0
    r_scale = min(cy_img, cx_img)

    # global mean: use ensemble result if valid, else unweighted mean
    global_angle = (est.angle_deg if est.valid
                    else float(np.mean([c["angle_deg"] for c in accepted])))

    rows = []
    for c in accepted:
        row_arr, col_arr = c["centroid"]   # array (row, col); row 0 = top of array

        # In display with origin='lower', row 0 is bottom: cy_px = row_arr directly
        cx_px = float(col_arr)
        cy_px = float(row_arr)

        x_norm = (cx_px - cx_img) / r_scale
        # y_norm: positive = upward in display = smaller row index in array → negate
        y_norm = -(cy_px - cy_img) / r_scale
        radius_norm  = float(np.sqrt(x_norm**2 + y_norm**2))
        azimuth_deg  = float(np.degrees(np.arctan2(y_norm, x_norm)))

        raw_residual = c["angle_deg"] - global_angle
        residual_deg = float((raw_residual + 90.0) % 180.0 - 90.0)

        rows.append({
            "cx_px":          cx_px,
            "cy_px":          cy_px,
            "cx_norm":        float(x_norm),
            "cy_norm":        float(y_norm),
            "radius_norm":    radius_norm,
            "azimuth_deg":    azimuth_deg,
            "angle_deg":      c["angle_deg"],
            "residual_deg":   residual_deg,
            "major_length":   c["major_length"],
            "eccentricity":   c["eccentricity"],
            "mean_brightness":c["mean_brightness"],
            "weight":         _candidate_weight(c),
        })

    return rows, est


def show_field_angle_overlay(
    feature_img: np.ndarray,
    field_data: List[Dict[str, Any]],
    est: Optional[AngleEstimate] = None,
    background: Optional[np.ndarray] = None,
    residual_vmax: float = 5.0,
) -> None:
    """Overlay fitted segment lines on the image, coloured by residual angle.

    Each accepted segment is drawn at its centroid with length = major_length.
    Colour = residual angle on a RdBu_r diverging scale (blue=negative,
    white=0, red=positive).  Line width scales with normalised weight so
    stronger candidates are more prominent.

    Parameters
    ----------
    feature_img    : image that defines the coordinate frame (for centroid coords)
    field_data     : output of analyze_field_angle_variation()
    est            : AngleEstimate for the global mean line overlay (optional)
    background     : image to show as grayscale background; defaults to feature_img.
                     Pass dbg["downsampled"] for a more natural astronomical look.
    residual_vmax  : colour scale ±limit in degrees
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if not field_data:
        print("No field data to display.")
        return

    bg = background if background is not None else feature_img

    fig, ax = plt.subplots(figsize=(12, 8))
    finite = bg[np.isfinite(bg)]
    vmin, vmax = np.percentile(finite, [1.0, 99.5]) if finite.size else (0.0, 1.0)
    ax.imshow(bg, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    cmap_res = cm.get_cmap("RdBu_r")
    norm_res = mcolors.Normalize(vmin=-residual_vmax, vmax=residual_vmax)

    weights = np.array([d["weight"] for d in field_data])
    w_max   = weights.max() + 1e-10

    for d in field_data:
        color = cmap_res(norm_res(d["residual_deg"]))
        lw    = 0.8 + 2.5 * (d["weight"] / w_max)
        _draw_line(ax, feature_img, d["angle_deg"],
                   color=color, lw=lw,
                   cx=d["cx_px"], cy=d["cy_px"],
                   length=d["major_length"])

    if est is not None and est.valid:
        _draw_line(ax, feature_img, est.angle_deg, color="lime", lw=2.5)
        ax.set_title(
            f"Local fits coloured by residual  "
            f"(lime = global mean {est.angle_deg:.2f}°)\n"
            f"RdBu_r: blue = −{residual_vmax}°  white = 0°  red = +{residual_vmax}°",
            fontsize=11)
    else:
        ax.set_title("Local fits coloured by residual angle", fontsize=11)

    sm = cm.ScalarMappable(cmap="RdBu_r", norm=norm_res)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Residual angle (°)", fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.show()


def show_field_angle_trends(
    field_data: List[Dict[str, Any]],
    feature_img_shape: Tuple[int, int],
) -> None:
    """Spatial trend plots for local angle residuals.

    Shows four separate figures:
      1. Residual vs X (column position) with weighted linear fit
      2. Residual vs Y (row position, display-up) with weighted linear fit
      3. Residual vs radius from centre with binned trend + fit
      4. 2D spatial scatter: each candidate dot at (cx_px, cy_px) coloured by residual

    Prints Pearson r and weighted linear slope for each 1D trend.
    These let you judge whether residuals are random scatter or a coherent
    spatial pattern (e.g. from lens distortion).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if not field_data:
        print("No field data.")
        return

    cx   = np.array([d["cx_norm"]    for d in field_data])
    cy   = np.array([d["cy_norm"]    for d in field_data])
    rad  = np.array([d["radius_norm"]for d in field_data])
    res  = np.array([d["residual_deg"]for d in field_data])
    wts  = np.array([d["weight"]     for d in field_data])
    wts_norm = wts / (wts.sum() + 1e-10)

    def _wlinfit(x, y, w):
        """Weighted linear fit y = m*x + b. Returns (m, b, r_pearson)."""
        sw   = np.sqrt(w / (w.sum() + 1e-10))
        coef = np.polyfit(x, y, 1, w=sw)
        # weighted Pearson r
        xm = np.dot(w, x) / w.sum()
        ym = np.dot(w, y) / w.sum()
        cov = np.dot(w, (x - xm) * (y - ym))
        sx  = np.sqrt(np.dot(w, (x - xm)**2))
        sy  = np.sqrt(np.dot(w, (y - ym)**2))
        r   = float(cov / (sx * sy + 1e-10))
        return float(coef[0]), float(coef[1]), r

    cmap_res = cm.get_cmap("RdBu_r")
    res_vmax = max(abs(res).max(), 1.0)
    norm_res = mcolors.Normalize(vmin=-res_vmax, vmax=res_vmax)
    colors   = [cmap_res(norm_res(r)) for r in res]
    sizes    = 20 + 80 * (wts / (wts.max() + 1e-10))

    x_fit = np.linspace(-1.1, 1.1, 200)

    # ── figure 1: residual vs X ──
    m, b, r = _wlinfit(cx, res, wts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cx, res, c=colors, s=sizes, alpha=0.85, edgecolors="none")
    ax.plot(x_fit, m * x_fit + b, "k--", lw=1.5,
            label=f"fit: {m:+.3f}·x + {b:+.3f}°  (r={r:+.3f})")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("X position (normalised, 0=centre)", fontsize=11)
    ax.set_ylabel("Residual angle (°)", fontsize=11)
    ax.set_title("Residual angle vs X", fontsize=12)
    ax.legend(fontsize=10)
    print(f"Residual vs X :  slope={m:+.4f} °/unit   r={r:+.4f}")
    plt.tight_layout(); plt.show()

    # ── figure 2: residual vs Y ──
    m, b, r = _wlinfit(cy, res, wts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cy, res, c=colors, s=sizes, alpha=0.85, edgecolors="none")
    ax.plot(x_fit, m * x_fit + b, "k--", lw=1.5,
            label=f"fit: {m:+.3f}·y + {b:+.3f}°  (r={r:+.3f})")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Y position (normalised, 0=centre, +up)", fontsize=11)
    ax.set_ylabel("Residual angle (°)", fontsize=11)
    ax.set_title("Residual angle vs Y", fontsize=12)
    ax.legend(fontsize=10)
    print(f"Residual vs Y :  slope={m:+.4f} °/unit   r={r:+.4f}")
    plt.tight_layout(); plt.show()

    # ── figure 3: residual vs radius ──
    m, b, r = _wlinfit(rad, res, wts)
    # binned radial trend
    n_bins  = min(8, max(3, len(field_data) // 5))
    bin_edges = np.linspace(rad.min(), rad.max(), n_bins + 1)
    bin_means, bin_centers = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (rad >= lo) & (rad < hi)
        if mask.sum() >= 2:
            w_bin = wts[mask]
            bin_means.append(np.dot(w_bin, res[mask]) / w_bin.sum())
            bin_centers.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(rad, res, c=colors, s=sizes, alpha=0.85, edgecolors="none")
    r_fit = np.linspace(0, rad.max() * 1.05, 200)
    ax.plot(r_fit, m * r_fit + b, "k--", lw=1.5,
            label=f"fit: {m:+.3f}·r + {b:+.3f}°  (r={r:+.3f})")
    if bin_centers:
        ax.plot(bin_centers, bin_means, "s-", color="orange",
                ms=7, lw=1.5, label="binned mean")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Radius from centre (normalised)", fontsize=11)
    ax.set_ylabel("Residual angle (°)", fontsize=11)
    ax.set_title("Residual angle vs Radius", fontsize=12)
    ax.legend(fontsize=10)
    print(f"Residual vs R :  slope={m:+.4f} °/unit   r={r:+.4f}")
    plt.tight_layout(); plt.show()

    # ── figure 4: 2D spatial scatter coloured by residual ──
    H, W = feature_img_shape
    cx_px = np.array([d["cx_px"] for d in field_data])
    cy_px = np.array([d["cy_px"] for d in field_data])

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(cx_px, cy_px, c=res, s=sizes, cmap="RdBu_r",
                    vmin=-res_vmax, vmax=res_vmax,
                    alpha=0.9, edgecolors="gray", linewidths=0.3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xlabel("Column (px)", fontsize=11)
    ax.set_ylabel("Row (px, 0=top of array)", fontsize=11)
    ax.set_title("2D spatial distribution of angle residuals\n"
                 "(dot size ∝ weight,  colour = residual °)", fontsize=11)
    ax.invert_yaxis()   # match array convention: row 0 at top
    plt.colorbar(sc, ax=ax, label="Residual angle (°)", fraction=0.03, pad=0.02)
    plt.tight_layout(); plt.show()

    # ── summary ──
    print(f"\nField variation summary ({len(field_data)} accepted candidates):")
    print(f"  residual range:  [{res.min():.2f}°, {res.max():.2f}°]")
    print(f"  weighted rms:    {np.sqrt(np.dot(wts_norm, res**2)):.3f}°")
