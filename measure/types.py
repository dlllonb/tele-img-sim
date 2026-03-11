# measure/types.py
# dataclasses and header parsing utilities for measurement pipeline

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits


# ---------- dataclasses ----------

@dataclass
class MeasurementMetadata:
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    rot_deg: Optional[float] = None
    plate_scale_arcsec_per_px: Optional[float] = None

    nx: Optional[int] = None
    ny: Optional[int] = None
    pixel_size_um: Optional[float] = None
    focal_length_mm: Optional[float] = None
    f_number: Optional[float] = None

    exposure_s: Optional[float] = None
    read_noise_e: Optional[float] = None
    gain_e_per_adu: Optional[float] = None
    qe: Optional[float] = None

    lambda_nm: Optional[float] = None
    bandwidth_nm: Optional[float] = None

    jitter_rms_arcsec: Optional[float] = None

    mask_kind: Optional[str] = None
    mask_angle_deg: Optional[float] = None
    grating_lines_per_mm: Optional[float] = None

    # store all header keys that were accessed
    _header_used: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_header(
        cls,
        header: fits.Header,
        overrides: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "MeasurementMetadata":
        """Create metadata by applying override/header/default precedence.

        """
        overrides = overrides or {}
        defaults = defaults or {}

        def _pick(keys: Tuple[str, ...], cast=None, default=None):
            # overrides first
            for k in keys:
                if k in overrides:
                    return overrides[k]
            # then header
            for k in keys:
                if k in header:
                    val = header[k]
                    return cast(val) if cast and val is not None else val
            # fall back to defaults
            if default is not None:
                return default
            return None

        md = cls()
        # simple helpers to try multiple keywords
        md.ra_deg = _pick(("RA0", "CRVAL1", "RA"), cast=float)
        md.dec_deg = _pick(("DEC0", "CRVAL2", "DEC"), cast=float)
        md.rot_deg = _pick(("ROT", "ROLL"), cast=float, default=0.0)
        md.plate_scale_arcsec_per_px = _pick(
            ("PLTSCL", "PIXSCL", "PIXSCALE"), cast=float
        )

        md.nx = _pick(("NAXIS1",), cast=int)
        md.ny = _pick(("NAXIS2",), cast=int)
        md.pixel_size_um = _pick(("PIXSIZE", "PIXEL", "PIXELSIZE"), cast=float)
        md.focal_length_mm = _pick(("FOCALLEN", "FOCAL", "FL"), cast=float)
        md.f_number = _pick(("FNUMBER", "FNO", "F/#"), cast=float)

        md.exposure_s = _pick(("EXPTIME", "EXPOSURE", "TEXPS"), cast=float)
        md.read_noise_e = _pick(("RDNOISE", "READNOI"), cast=float)
        md.gain_e_per_adu = _pick(("GAIN", "GAINE"), cast=float)
        md.qe = _pick(("QE",), cast=float)

        md.lambda_nm = _pick(("LAMBDAN", "WAVELEN"), cast=float)
        md.bandwidth_nm = _pick(("BANDWID", "BAND"), cast=float)

        md.jitter_rms_arcsec = _pick(("JITRMS", "JITTER"), cast=float)

        md.mask_kind = _pick(("MASKKIND", "MASK"), cast=str)
        md.mask_angle_deg = _pick(("MASKANG",), cast=float)
        md.grating_lines_per_mm = _pick(("GRATLIN",), cast=float)

        # record which header keys we consumed (approximate)
        for k in header.keys():
            if k in overrides or k in defaults:
                continue
            if k.upper() in (
                "RA0",
                "DEC0",
                "ROT",
                "PLTSCL",
                "NAXIS1",
                "PIXSIZE",
                "FOCALLEN",
                "EXPTIME",
                "RDNOISE",
                "GAIN",
                "QE",
                "LAMBDAN",
                "JITRMS",
                "MASKKIND",
            ):
                md._header_used[k] = header[k]
        return md


@dataclass
class MeasurementInput:
    filepath: str
    image: np.ndarray
    header: fits.Header
    meta: MeasurementMetadata


@dataclass
class StarDetectionResult:
    success: bool = False
    n_stars_detected: int = 0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    flux: np.ndarray = field(default_factory=lambda: np.array([]))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlateSolveResult:
    success: bool = False
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    rot_deg: Optional[float] = None
    wcs: Any = None  # placeholder for a WCS object or dict
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeMeasurementResult:
    success: bool = False
    angle_deg: Optional[float] = None
    sigma_angle_deg: Optional[float] = None
    feature_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementMetrics:
    solve_success: bool = False
    spike_success: bool = False
    final_angle_deg: Optional[float] = None
    final_sigma_deg: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementResult:
    input_data: MeasurementInput
    star_result: StarDetectionResult
    platesolve_result: PlateSolveResult
    spike_result: SpikeMeasurementResult
    metrics: MeasurementMetrics
    success: bool = False
    messages: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MeasurementResult(success={self.success}, "
            f"stars={self.star_result.n_stars_detected}, "
            f"solve={self.platesolve_result.success}, "
            f"spike={self.spike_result.success})"
        )
