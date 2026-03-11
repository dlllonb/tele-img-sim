# measure/platesolve_proxy.py
# thin wrapper or proxy to a plate-solving service or algorithm

from __future__ import annotations
from typing import Any

from .types import BranchImageResult, MeasurementMetadata, PlateSolveResult


def run_platesolve(
    branch: BranchImageResult,
    meta: MeasurementMetadata,
) -> PlateSolveResult:
    """Placeholder platesolve operating on a preprocessed star branch image.

    Parameters
    ----------
    branch : BranchImageResult
        Preprocessed image intended for astrometry/plate solving.
    meta : MeasurementMetadata
        Normalized metadata from the FITS header.

    Returns
    -------
    PlateSolveResult
        Stub result; currently not a real solve.

    The stub behaviour is explicit: success is False and the result contains
    a message indicating that no platesolver backend is configured.  If
    the header provided RA/Dec/rot, the stub will echo them but will still
    mark success=False to emphasise that this is not a real solution.
    """
    res = PlateSolveResult()
    res.messages.append("platesolve stub invoked; no backend implemented")
    if meta.ra_deg is not None:
        res.ra_deg = meta.ra_deg
        res.dec_deg = meta.dec_deg
        res.rot_deg = meta.rot_deg
        res.messages.append("echoed header RA/Dec/rot into result (stub)")
    # success remains False until a real solver is hooked up
    return res