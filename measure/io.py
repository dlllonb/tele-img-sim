# measure/io.py
# utilities for FITS I/O and output directory management

from __future__ import annotations
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .types import BranchImageResult, MeasurementResult

import numpy as np
from astropy.io import fits


def make_run_dir(output_dir: str, run_name: Optional[str], input_filepath: str) -> Path:
    outroot = Path(output_dir)
    outroot.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        stem = Path(input_filepath).stem
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{stem}_{timestamp}"
    runpath = outroot / run_name
    runpath.mkdir(parents=True, exist_ok=True)
    return runpath


def write_branch_fits(
    branch: "BranchImageResult",
    runpath: Path,
    orig_header: fits.Header,
    suffix: str,
) -> Path:
    """Write preprocessed branch image to FITS with provenance header.

    suffix is a short name e.g. 'star_branch' or 'stripe_branch'.
    """
    fname = runpath / f"{suffix}.fits"
    hdr = orig_header.copy()
    hdr["HIERARCH"] = True  # allow long keys
    hdr["BRANCH"] = branch.branch_name
    hdr["PROVENIN"] = orig_header.get("FILENAME", "?")
    hdr["DATE"] = datetime.utcnow().isoformat()
    # add any header updates
    for k, v in branch.header_updates.items():
        try:
            hdr[k] = v
        except Exception:
            pass
    fits.writeto(fname, branch.image, hdr, overwrite=True)
    return fname


def write_summary(result: "MeasurementResult", runpath: Path) -> Path:
    """Dump a machine-readable JSON summary and a human text log."""
    # core information
    summary: Dict[str, Any] = {
        "input_filepath": result.input_data.filepath,
        "timestamp": datetime.utcnow().isoformat(),
        "image_shape": None,
        "success": result.success,
        "paths": result.output_paths,
    }

    # image shape if available
    try:
        summary["image_shape"] = tuple(result.input_data.image.shape)
    except Exception:
        pass

    # metadata fields of interest
    md = result.input_data.meta
    summary["metadata"] = {
        "ra_deg": md.ra_deg,
        "dec_deg": md.dec_deg,
        "rot_deg": md.rot_deg,
        "plate_scale_arcsec_px": md.plate_scale_arcsec_per_px,
        "exposure_s": md.exposure_s,
        "focal_length_mm": md.focal_length_mm,
        "f_number": md.f_number,
        "pixel_size_um": md.pixel_size_um,
        "grating_lines_per_mm": md.grating_lines_per_mm,
        "mask_angle_deg": md.mask_angle_deg,
    }

    # platesolve results
    pr = result.platesolve_result
    summary["platesolve"] = {
        "success": pr.success,
        "ra_deg": pr.ra_deg,
        "dec_deg": pr.dec_deg,
        "rot_deg": pr.rot_deg,
    }

    # stripe measurement results
    sr = result.spike_result
    summary["stripe"] = {
        "success": sr.success,
        "image_angle_deg": sr.image_angle_deg,
        "sigma_angle_deg": sr.sigma_angle_deg,
    }

    # derived metrics
    summary["metrics"] = {
        "sky_angle_deg": result.metrics.sky_angle_deg,
        "final_sigma_deg": result.metrics.final_sigma_deg,
        "quality_flags": result.metrics.quality_flags,
    }

    # diagnostics placeholders
    summary["diagnostics"] = {
        "n_detected_stars": None,
        "n_used_for_solve": None,
        "n_stripe_features": None,
    }

    # messages/log
    summary["messages"] = result.messages
    jpath = runpath / "summary.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tpath = runpath / "summary.txt"
    with open(tpath, "w", encoding="utf-8") as f:
        for line in result.messages:
            f.write(line + "\n")
    return jpath
