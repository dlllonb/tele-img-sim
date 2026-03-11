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
    summary = {
        "input": result.input_data.filepath,
        "success": result.success,
        "metrics": {
            "sky_angle_deg": result.metrics.sky_angle_deg,
            "final_sigma_deg": result.metrics.final_sigma_deg,
        },
        "paths": result.output_paths,
        "messages": result.messages,
    }
    jpath = runpath / "summary.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tpath = runpath / "summary.txt"
    with open(tpath, "w", encoding="utf-8") as f:
        for line in result.messages:
            f.write(line + "\n")
    return jpath
