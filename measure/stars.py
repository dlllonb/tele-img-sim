# measure/stars.py
# functions for star detection and centroiding from an image

from __future__ import annotations
from typing import Any
import numpy as np

from .types import MeasurementInput, StarDetectionResult


def detect_stars(inp: MeasurementInput) -> StarDetectionResult:
    """Stub star finder.

    Real implementation will locate bright sources in `inp.image`,
    compute centroids & fluxes, and populate the result arrays.

    The stub returns an empty result (success=True) and records the
    image shape in `extra` for debugging.
    """
    res = StarDetectionResult()
    res.success = True
    res.n_stars_detected = 0
    res.extra["image_shape"] = inp.image.shape
    return res