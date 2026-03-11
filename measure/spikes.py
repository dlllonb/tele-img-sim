# measure/spikes.py
# routines to detect and fit diffraction spike or spectrum orientations

from __future__ import annotations
import numpy as np

from .types import (
    MeasurementInput,
    StarDetectionResult,
    PlateSolveResult,
    SpikeMeasurementResult,
)


def measure_spike_angle(
    inp: MeasurementInput,
    star_res: StarDetectionResult,
    plate_res: PlateSolveResult,
) -> SpikeMeasurementResult:
    """Stub spike/feature orientation measurement.

    The real function will analyze `inp.image` (optionally using the
    star/plate results to mask out stars or transform coordinates) and
    return the best-fit diffraction orientation and uncertainty.

    Here we simply return a placeholder that echos the mask angle from
    metadata if available.
    """
    res = SpikeMeasurementResult()
    res.success = True
    if inp.meta.mask_angle_deg is not None:
        res.angle_deg = inp.meta.mask_angle_deg
    return res