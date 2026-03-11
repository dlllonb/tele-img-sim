# measure/pipeline.py
# high-level wrapper that runs detection, platesolve, spike fit, and metrics for a frame

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os
from pathlib import Path
import numpy as np
from astropy.io import fits

from .types import (
    MeasurementInput,
    MeasurementMetadata,
    MeasurementResult,
)
from .preprocess import prepare_star_branch_input, prepare_stripe_branch_input
from .platesolve import run_platesolve
from .spikes import measure_diffraction_angle
from .metrics import compute_metrics
from .io import make_run_dir, write_branch_fits, write_summary


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
    output_dir: str = "out",
    run_name: Optional[str] = None,
    save_outputs: bool = True,
    keep_intermediates: bool = True,
    show: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> MeasurementResult:
    """Run the measurement pipeline on a FITS image.

    This updated flow creates two preprocessing branches, runs the plate
    solver on the star branch and the diffraction-angle measurement on the
    stripe branch, computes derived metrics, and optionally writes outputs.

    Parameters
    ----------
    filepath : str
        Input FITS path.
    output_dir : str
        Root directory for saving outputs (will contain a run subfolder).
    run_name : str or None
        Name for this run; if None one is generated from the input filename
        and timestamp.
    save_outputs : bool
        If True, pipeline will create files under `out/<run_name>`.
    keep_intermediates : bool
        If True, intermediate images/results will be stored in the returned
        result object.
    show : bool
        If True, display branch images during execution.
    overrides, defaults : dict, optional
        Metadata precedence containers.
    verbose : bool
        Print progress messages.
    """
    messages: list[str] = []
    intermediates: Dict[str, Any] = {}
    output_paths: Dict[str, str] = {}

    # 1. load data + metadata
    image, header = _load_fits(filepath)
    meta = MeasurementMetadata.from_header(header, overrides=overrides, defaults=defaults)
    inp = MeasurementInput(filepath=filepath, image=image, header=header, meta=meta)
    messages.append("loaded FITS")

    # optional run directory
    runpath: Optional[Path] = None
    if save_outputs:
        runpath = make_run_dir(output_dir, run_name, filepath)
        output_paths["run_dir"] = str(runpath)
        messages.append(f"created run directory {runpath}")

    # 2. branch preprocessing
    star_branch = prepare_star_branch_input(image, meta)
    stripe_branch = prepare_stripe_branch_input(image, meta)
    messages.extend(star_branch.messages)
    messages.extend(stripe_branch.messages)
    if keep_intermediates:
        intermediates["star_branch_image"] = star_branch.image
        intermediates["stripe_branch_image"] = stripe_branch.image

    if save_outputs and runpath is not None:
        # save branch images to FITS
        p1 = write_branch_fits(star_branch, runpath, header, "star_branch")
        p2 = write_branch_fits(stripe_branch, runpath, header, "stripe_branch")
        output_paths["star_branch_fits"] = str(p1)
        output_paths["stripe_branch_fits"] = str(p2)
        messages.append(f"wrote branch FITS files")

    # optional display
    if show:
        try:
            import matplotlib.pyplot as plt

            def _display_with_percentile(ax, img, title, qlo=0.5, qhi=99.5):
                # compute robust limits
                finite = img[np.isfinite(img)]
                if finite.size > 0:
                    vmin = float(np.percentile(finite, qlo))
                    vmax = float(np.percentile(finite, qhi))
                else:
                    vmin, vmax = 0.0, 1.0
                if verbose:
                    messages.append(f"{title} display vmin={vmin:.3g} vmax={vmax:.3g}")
                ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            _display_with_percentile(ax[0], star_branch.image, "star branch")
            _display_with_percentile(ax[1], stripe_branch.image, "stripe branch")
            plt.show()
        except ImportError:
            messages.append("matplotlib not available; cannot show images")

    # 3. platesolve
    plate_res = run_platesolve(star_branch, meta)
    messages.append("platesolve stage completed")

    # 4. stripe measurement
    spike_res = measure_diffraction_angle(stripe_branch, meta)
    messages.append("diffraction angle stage completed")

    # 5. metrics
    metrics = compute_metrics(plate_res, spike_res, meta)
    messages.append("metrics computed")

    # determine overall success
    success = bool(metrics.sky_angle_deg is not None)

    result = MeasurementResult(
        input_data=inp,
        star_branch=star_branch,
        stripe_branch=stripe_branch,
        platesolve_result=plate_res,
        spike_result=spike_res,
        metrics=metrics,
        success=success,
        messages=messages,
        output_paths=output_paths,
        intermediates=intermediates if keep_intermediates else {},
    )

    if save_outputs and runpath is not None:
        write_summary(result, runpath)
        messages.append("wrote summary files")

    if verbose:
        for m in messages:
            print(m)

    return result
