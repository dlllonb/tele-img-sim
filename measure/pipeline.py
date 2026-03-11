# measure/pipeline.py
# high-level wrapper that runs detection, platesolve, spike fit, and metrics for a frame

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from .types import (
    MeasurementInput,
    MeasurementMetadata,
    MeasurementResult,
    MeasurementMetrics,
)
from .stars import detect_stars
from .platesolve import run_platesolve
from .spikes import measure_spike_angle
from .metrics import compute_measurement_metrics


# ---------- internal helpers ----------

def _load_fits(filepath: str) -> Tuple[np.ndarray, fits.Header]:
    try:
        with fits.open(filepath) as hdul:
            hdu = hdul[0]
            data = hdu.data
            hdr = hdu.header
    except Exception as e:
        raise IOError(f"could not open FITS file '{filepath}': {e}")

    if data is None:
        raise ValueError(f"FITS file '{filepath}' contains no data")
    if data.ndim != 2:
        raise ValueError(f"FITS image must be 2‑D; got ndim={data.ndim}")

    return data.astype(np.float32), hdr


# ---------- public pipeline ----------

def run_measurement_pipeline(
    filepath: str,
    overrides: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> MeasurementResult:
    """Run the full measurement chain on a FITS image at `filepath`.

    Parameters
    ----------
    filepath : str
        Path to a FITS file containing a 2‑D image in the primary HDU.
    overrides : dict, optional
        Metadata values that take precedence over FITS header contents.
    defaults : dict, optional
        Fallback metadata if neither header nor overrides supply them.
    verbose : bool
        If True, accumulate simple messages and (optionally) print them.

    Returns
    -------
    MeasurementResult
        Structured results including all intermediate outputs and metrics.
    """
    messages: list[str] = []

    # load data + header
    image, header = _load_fits(filepath)
    meta = MeasurementMetadata.from_header(header, overrides=overrides, defaults=defaults)
    inp = MeasurementInput(filepath=filepath, image=image, header=header, meta=meta)
    messages.append("loaded FITS")

    # stage 1: star detection
    star_res = detect_stars(inp)
    messages.append(f"star detection: success={star_res.success}, n={star_res.n_stars_detected}")

    # stage 2: platesolve
    plate_res = run_platesolve(inp, star_res)
    messages.append(f"platesolve: success={plate_res.success}")

    # stage 3: spike measurement
    spike_res = measure_spike_angle(inp, star_res, plate_res)
    messages.append(f"spike measurement: success={spike_res.success}")

    # stage 4: metrics
    metrics = compute_measurement_metrics(inp, star_res, plate_res, spike_res)
    messages.append("computed metrics")

    success = plate_res.success and spike_res.success

    result = MeasurementResult(
        input_data=inp,
        star_result=star_res,
        platesolve_result=plate_res,
        spike_result=spike_res,
        metrics=metrics,
        success=success,
        messages=messages,
    )

    if verbose:
        for m in messages:
            print(m)

    return result
