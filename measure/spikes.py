# measure/spikes.py
# routines to detect and fit diffraction spike or spectrum orientations

from __future__ import annotations
import numpy as np

from .types import (
    BranchImageResult,
    MeasurementMetadata,
    PlateSolveResult,
    SpikeMeasurementResult,
)


def measure_diffraction_angle(
    branch: BranchImageResult,
    meta: MeasurementMetadata,
) -> SpikeMeasurementResult:
    """Stub diffraction orientation measurement on stripe branch image.

    Parameters
    ----------
    branch : BranchImageResult
        Preprocessed image optimized for stripe detection.
    meta : MeasurementMetadata
        Normalized metadata; may contain mask angle hints.

    Returns
    -------
    SpikeMeasurementResult
        Stub result.  Currently success=False and image_angle_deg set
        only if metadata contains a mask angle, as a placeholder.
    """
    res = SpikeMeasurementResult()
    res.messages.append("diffraction-angle stub invoked; no real fit performed")
    if meta.mask_angle_deg is not None:
        res.image_angle_deg = meta.mask_angle_deg
        res.messages.append("echoed mask_angle_deg from metadata (stub)")
    # success left False to remind user this is not a measurement
    return res