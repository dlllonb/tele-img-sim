# measure/preprocess.py
# branch-specific image preprocessing stubs

from __future__ import annotations
from typing import Any

import numpy as np

from .types import MeasurementMetadata, BranchImageResult


def prepare_star_branch_input(
    raw_image: np.ndarray,
    meta: MeasurementMetadata,
    **kwargs: Any,
) -> BranchImageResult:
    """Prepare an image optimized for star/platesolve branch.

    The real implementation will mask or suppress elongated diffraction
    features, sharpen stellar cores, and possibly rescale/denoise.  It
    receives the raw image and normalized metadata.

    Returns a BranchImageResult containing the preprocessed image.
    """
    res = BranchImageResult(success=True, branch_name="star")
    # copy for now; later we may apply filtering
    res.image = np.array(raw_image, copy=True)
    res.messages.append("star branch stub: passthrough")
    return res


def prepare_stripe_branch_input(
    raw_image: np.ndarray,
    meta: MeasurementMetadata,
    **kwargs: Any,
) -> BranchImageResult:
    """Prepare an image optimized for diffraction/stripe branch.

    The real implementation will suppress point sources and enhance
    elongated orders/spectra so that angle measurement is easier.
    """
    res = BranchImageResult(success=True, branch_name="stripe")
    res.image = np.array(raw_image, copy=True)
    res.messages.append("stripe branch stub: passthrough")
    return res
