# measure/preprocess.py
# branch-specific image preprocessing

from __future__ import annotations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .types import MeasurementMetadata, BranchImageResult

# Sigma (px) for the large-scale Gaussian used to estimate and remove the sky
# background gradient.  Should be much larger than any stellar PSF (~few px)
# but comparable to or larger than the grating spectrum length (~50-200 px).
_BG_SIGMA_PX = 50.0

# Additional smoothing applied to the stripe branch to reduce the prominence
# of point sources (narrow stars) relative to extended spectral features.
_STRIPE_SMOOTH_SIGMA_PX = 2.5


def _subtract_background(image: np.ndarray, sigma: float) -> np.ndarray:
    """Estimate and subtract a large-scale Gaussian background.

    Computes a heavily blurred version of the image (sigma >> star PSF) and
    subtracts it, leaving stars and extended features on a near-zero
    background.  Negative residuals are clipped to zero.
    """
    bg = gaussian_filter(image.astype(np.float32), sigma=sigma)
    residual = image.astype(np.float32) - bg
    np.clip(residual, 0.0, None, out=residual)
    return residual


def prepare_star_branch_input(
    raw_image: np.ndarray,
    meta: MeasurementMetadata,
    **kwargs: Any,
) -> BranchImageResult:
    """Prepare an image optimised for star detection and plate-solving.

    Processing:
      1. Subtract a large-scale Gaussian background (sigma=50 px) to remove
         smooth sky and vignetting gradients, leaving point sources and
         extended features on a flat background.

    Stars are left sharp; no blurring is applied so centroids remain
    well-localised for the plate solver.
    """
    res = BranchImageResult(success=True, branch_name="star")
    star_img = _subtract_background(raw_image, sigma=_BG_SIGMA_PX)
    res.image = star_img
    res.messages.append(
        f"star branch: background subtracted (Gaussian sigma={_BG_SIGMA_PX:.0f} px)"
    )
    return res


def prepare_stripe_branch_input(
    raw_image: np.ndarray,
    meta: MeasurementMetadata,
    **kwargs: Any,
) -> BranchImageResult:
    """Prepare an image optimised for diffraction-stripe angle measurement.

    Processing:
      1. Subtract the same large-scale Gaussian background as the star branch.
      2. Apply a moderate Gaussian smooth (sigma=2.5 px).  This reduces the
         peak brightness of narrow point sources (stars, whose PSF is ~1-3 px
         wide) more than it does for broader, elongated spectral features,
         making the stripes relatively more prominent.

    The resulting image is suitable for Radon / Hough-based angle extraction.
    Star suppression will be improved in a later step once source detection
    is available.
    """
    res = BranchImageResult(success=True, branch_name="stripe")
    bg_subtracted = _subtract_background(raw_image, sigma=_BG_SIGMA_PX)
    stripe_img = gaussian_filter(bg_subtracted, sigma=_STRIPE_SMOOTH_SIGMA_PX)
    res.image = stripe_img
    res.messages.append(
        f"stripe branch: background subtracted (Gaussian sigma={_BG_SIGMA_PX:.0f} px), "
        f"smoothed (sigma={_STRIPE_SMOOTH_SIGMA_PX:.1f} px)"
    )
    return res
