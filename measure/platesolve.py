# measure/platesolve.py
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

from .types import BranchImageResult, MeasurementMetadata, PlateSolveResult

# API key lives one level above this package (project root).
_API_KEY_FILE = Path(__file__).parent.parent / "astrometry_api.txt"

_MAX_SOURCES      = 300
_SEARCH_RADIUS_DEG = 5.0
_API_URL          = "https://nova.astrometry.net/api"
_SOLVE_TIMEOUT    = 180   # seconds to wait for a solution
_POLL_INTERVAL    = 4     # seconds between status polls


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

    # Top N by flux, then deduplicate overlapping detections
    fluxes = sources["flux"]
    if len(fluxes) > _MAX_SOURCES:
        top = np.argsort(fluxes)[::-1][:_MAX_SOURCES]
        xs, ys, fs = sources["x"][top], sources["y"][top], fluxes[top]
    else:
        xs, ys, fs = sources["x"], sources["y"], fluxes

    xs, ys = _deduplicate_sources(xs, ys, fs)

    res.debug["sources"] = {"x": xs, "y": ys}
    res.messages.append(f"submitting {len(xs)} sources to nova.astrometry.net")

    ny, nx = branch.image.shape[-2], branch.image.shape[-1]

    hints: dict = {}
    if meta.ra_deg is not None:
        hints.update(center_ra=meta.ra_deg, center_dec=meta.dec_deg,
                     radius=_SEARCH_RADIUS_DEG)
    if meta.plate_scale_arcsec_per_px is not None:
        ps = meta.plate_scale_arcsec_per_px
        hints.update(scale_lower=ps * 0.75, scale_upper=ps * 1.25,
                     scale_units="arcsecperpix")

    try:
        wcs_header = _submit_source_list(api_key, xs, ys, nx, ny, hints)
    except Exception as exc:
        res.messages.append(f"astrometry.net error: {exc}")
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
# Direct nova.astrometry.net HTTP client (no astroquery dependency)
# ---------------------------------------------------------------------------

def _api_post(endpoint: str, payload: dict, timeout: int = 30) -> dict:
    r = requests.post(
        f"{_API_URL}/{endpoint}",
        data={"request-json": json.dumps(payload)},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _submit_source_list(
    api_key: str,
    xs: np.ndarray,
    ys: np.ndarray,
    image_width: int,
    image_height: int,
    hints: dict,
) -> Optional[fits.Header]:
    """Login, submit source list, poll until solved, return WCS Header or None."""

    # 1. Login
    resp = _api_post("login", {"apikey": api_key})
    if resp.get("status") != "success":
        raise RuntimeError(f"login failed: {resp}")
    session = resp["session"]

    # 2. Submit source list via url_upload endpoint
    payload = {
        "session":      session,
        "x":            [float(v) for v in xs],
        "y":            [float(v) for v in ys],
        "image_width":  int(image_width),
        "image_height": int(image_height),
    }
    payload.update(hints)

    resp = _api_post("url_upload", payload, timeout=60)
    if resp.get("status") != "success":
        raise RuntimeError(f"submission rejected: {resp}")
    sub_id = resp["subid"]

    # 3. Poll until a job is assigned to this submission
    job_id = None
    t0 = time.time()
    while time.time() - t0 < _SOLVE_TIMEOUT:
        time.sleep(_POLL_INTERVAL)
        print(".", end="", flush=True)
        r = requests.get(f"{_API_URL}/submissions/{sub_id}", timeout=15)
        r.raise_for_status()
        jobs = [j for j in r.json().get("jobs", []) if j is not None]
        if jobs:
            job_id = jobs[0]
            break

    if job_id is None:
        raise RuntimeError(
            f"submission {sub_id} created no jobs within {_SOLVE_TIMEOUT}s — "
            "source list may have been rejected by the server"
        )

    # 4. Poll job status
    status = ""
    while time.time() - t0 < _SOLVE_TIMEOUT:
        time.sleep(_POLL_INTERVAL)
        print(".", end="", flush=True)
        r = requests.get(f"{_API_URL}/jobs/{job_id}/info", timeout=15)
        r.raise_for_status()
        status = r.json().get("status", "")
        if status in ("success", "failure"):
            break
    print()  # newline after dots

    if status != "success":
        return None  # failure or timeout → caller treats as unsolved

    # 5. Fetch WCS FITS header
    r = requests.get(f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=30)
    r.raise_for_status()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = fits.Header.fromstring(r.text)
    return header


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

    # Build DAOStarFinder with version-safe parameter names
    import photutils
    _phot_ver = tuple(int(x) for x in photutils.__version__.split(".")[:2])
    if _phot_ver >= (3, 0):
        finder = DAOStarFinder(
            fwhm=3.0, threshold=5.0 * std,
            sharpness_range=(0.05, 2.0), roundness_range=(-2.0, 2.0),
            peak_max=None,
        )
    else:
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

    # Column names changed in photutils 3.0
    x_col = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    y_col = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
    return {
        "x":    table[x_col].data.astype(float),
        "y":    table[y_col].data.astype(float),
        "flux": table["flux"].data.astype(float),
    }


def _deduplicate_sources(
    xs: np.ndarray,
    ys: np.ndarray,
    fluxes: np.ndarray,
    min_separation: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS: keep only one source per cluster of overlapping detections.

    Iterates brightest-first (fluxes assumed pre-sorted descending).  A
    candidate is accepted only if no already-accepted source is within
    ``min_separation`` pixels.  Default 24 px = 2 × the 12 px display-circle
    radius, i.e. circles that touch or overlap are collapsed to the brighter one.
    """
    if len(xs) == 0:
        return xs, ys

    kept_x: list[float] = []
    kept_y: list[float] = []

    for x, y in zip(xs, ys):
        if kept_x:
            dx = np.asarray(kept_x) - x
            dy = np.asarray(kept_y) - y
            if np.min(dx * dx + dy * dy) < min_separation ** 2:
                continue
        kept_x.append(float(x))
        kept_y.append(float(y))

    return np.asarray(kept_x), np.asarray(kept_y)


def _extract_rot(wcs: WCS) -> float:
    shape = wcs.pixel_shape or (1000, 1000)
    cx, cy = shape[0] / 2.0, shape[1] / 2.0
    pts  = wcs.all_pix2world([[cx, cy], [cx, cy + 1.0]], 0)
    dra  = (pts[1, 0] - pts[0, 0]) * np.cos(np.radians(pts[0, 1]))
    ddec = pts[1, 1] - pts[0, 1]
    return float(np.degrees(np.arctan2(-dra, ddec)))
