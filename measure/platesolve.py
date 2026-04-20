# measure/platesolve.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .types import BranchImageResult, MeasurementMetadata, PlateSolveResult

# API key lives one level above this package (project root).
_API_KEY_FILE = Path(__file__).parent.parent / "astrometry_api.txt"

_MAX_SOURCES = 300          # cap sent to nova.astrometry.net
_SEARCH_RADIUS_DEG = 5.0


def run_platesolve(
    branch: BranchImageResult,
    meta: MeasurementMetadata,
) -> PlateSolveResult:
    """Plate-solve the star-branch image via nova.astrometry.net.

    Reads the API key from ``astrometry_api.txt`` at the project root.
    Detects sources from ``branch.image``, submits the brightest
    ``_MAX_SOURCES`` to ``solve_from_source_list``, and parses the
    returned WCS into a ``PlateSolveResult``.
    """
    res = PlateSolveResult()

    api_key = _load_api_key()
    if not api_key:
        res.messages.append(
            f"no API key found — create {_API_KEY_FILE} containing your "
            "nova.astrometry.net key"
        )
        return res

    sources = _detect_sources(branch.image)
    if sources is None:
        res.messages.append("photutils not installed — cannot detect sources")
        return res
    if len(sources["x"]) == 0:
        res.messages.append("no sources detected in star branch image")
        return res

    # Top N by flux
    fluxes = sources["flux"]
    if len(fluxes) > _MAX_SOURCES:
        top = np.argsort(fluxes)[::-1][:_MAX_SOURCES]
        xs, ys = sources["x"][top], sources["y"][top]
    else:
        xs, ys = sources["x"], sources["y"]

    res.messages.append(f"submitting {len(xs)} sources to nova.astrometry.net")

    try:
        from astroquery.astrometry_net import AstrometryNet  # type: ignore
    except ImportError:
        res.messages.append("astroquery not installed — pip install astroquery")
        return res

    ny, nx = branch.image.shape[-2], branch.image.shape[-1]

    kwargs = {"center_ra": meta.ra_deg, "center_dec": meta.dec_deg,
              "radius": _SEARCH_RADIUS_DEG} if meta.ra_deg is not None else {}

    if meta.plate_scale_arcsec_per_px is not None:
        ps = meta.plate_scale_arcsec_per_px
        kwargs.update(scale_lower=ps * 0.75, scale_upper=ps * 1.25,
                      scale_units="arcsecperpix")

    try:
        an = AstrometryNet()
        an.api_key = api_key
        wcs_header = an.solve_from_source_list(xs, ys, nx, ny, **kwargs)
    except Exception as exc:
        res.messages.append(f"astroquery error: {exc}")
        return res

    if wcs_header is None:
        res.messages.append("nova.astrometry.net returned no solution")
        return res

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fits.verify.VerifyWarning)
        wcs = WCS(wcs_header)

    ra  = float(wcs.wcs.crval[0])
    dec = float(wcs.wcs.crval[1])
    rot = _extract_rot(wcs)

    res.success  = True
    res.ra_deg   = ra
    res.dec_deg  = dec
    res.rot_deg  = rot
    res.wcs_info = wcs_header
    res.messages.append(f"solved: RA={ra:.5f}° Dec={dec:.5f}° rot={rot:.3f}°")
    return res


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_api_key() -> Optional[str]:
    try:
        key = _API_KEY_FILE.read_text().strip()
        return key if key else None
    except FileNotFoundError:
        return None


def _detect_sources(image: np.ndarray) -> Optional[dict]:
    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder, find_peaks
    except ImportError:
        return None

    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    sub = image - median

    finder = DAOStarFinder(
        fwhm=3.0, threshold=5.0 * std,
        sharplo=0.05, sharphi=2.0, roundlo=-2.0, roundhi=2.0,
        peakmax=None,
    )
    table = finder(sub)

    if table is None or len(table) == 0:
        table = find_peaks(sub, threshold=5.0 * std, box_size=7)
        if table is None or len(table) == 0:
            return {"x": np.array([]), "y": np.array([]), "flux": np.array([])}
        return {
            "x":    table["x_peak"].data.astype(float),
            "y":    table["y_peak"].data.astype(float),
            "flux": table["peak_value"].data.astype(float),
        }

    return {
        "x":    table["xcentroid"].data.astype(float),
        "y":    table["ycentroid"].data.astype(float),
        "flux": table["flux"].data.astype(float),
    }


def _extract_rot(wcs: WCS) -> float:
    shape = wcs.pixel_shape or (1000, 1000)
    cx, cy = shape[0] / 2.0, shape[1] / 2.0
    pts  = wcs.all_pix2world([[cx, cy], [cx, cy + 1.0]], 0)
    dra  = (pts[1, 0] - pts[0, 0]) * np.cos(np.radians(pts[0, 1]))
    ddec = pts[1, 1] - pts[0, 1]
    return float(np.degrees(np.arctan2(-dra, ddec)))
