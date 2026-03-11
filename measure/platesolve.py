# measure/platesolve_proxy.py
# thin wrapper or proxy to a plate-solving service or algorithm

from __future__ import annotations
from typing import Any

from .types import MeasurementInput, StarDetectionResult, PlateSolveResult


def run_platesolve(
    inp: MeasurementInput,
    star_res: StarDetectionResult,
) -> PlateSolveResult:
    """Stub platesolver.

    A future implementation would call an external solver (e.g. astrometry.net)
    or a local library, providing detected star positions to derive a WCS and
    astrometric pointing/rotation.

    This placeholder echoes metadata RA/Dec/rot if available, and otherwise
    returns a mostly-empty result.
    """
    res = PlateSolveResult()
    if inp.meta.ra_deg is not None:
        res.success = True
        res.ra_deg = inp.meta.ra_deg
        res.dec_deg = inp.meta.dec_deg
        res.rot_deg = inp.meta.rot_deg
    return res