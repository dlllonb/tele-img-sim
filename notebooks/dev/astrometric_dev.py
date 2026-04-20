"""
astrometric_dev.py — development twin of measure/platesolve.py

Wraps an external plate-solver and exposes clean functions that mirror the
eventual production interface.  Notebook display helpers live at the bottom,
clearly separated from core logic, so the top half can be ported to
measure/platesolve.py with minimal editing.

Backend assumptions
-------------------
  "local"       : astrometry.net ``solve-field`` binary on PATH (or a path you
                  configure).  Install via ``apt install astrometry.net`` on
                  Linux / WSL2.  On Windows, run under WSL2 and set
                  ``PlateSolveConfig.solve_field_bin = "wsl solve-field"``
                  (paths inside the subprocess must be Linux-style; see the
                  notebook for a helper that converts them).
  "astroquery"  : submits to nova.astrometry.net via the astroquery library.
                  Requires a free API key and internet access.  Slower (~1 min)
                  but needs no local installation.  ``pip install astroquery``.

Porting notes
-------------
  When moving to measure/platesolve.py:
    1. Remove the fallback ``PlateSolveResult`` definition below — import from
       measure.types directly.
    2. Replace the ``PlateSolveConfig`` definition here with one in
       measure/types.py (or keep it in platesolve.py and re-export it).
    3. Keep all the functions below ``# --- notebook helpers ---`` in this dev
       module or move them to a separate measure/display.py.
    4. The ``solve_field`` signature matches the stub in measure/platesolve.py —
       swap the stub body with the real one.
"""
from __future__ import annotations

import copy
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Re-use the canonical result type from the real package so the dataclass shape
# stays in sync.  The fallback lets this module load without the measure package
# (e.g. when running a quick standalone test outside the repo).
try:
    from measure.types import PlateSolveResult  # type: ignore
except ImportError:
    @dataclass
    class PlateSolveResult:  # type: ignore[no-redef]
        success: bool = False
        ra_deg: Optional[float] = None
        dec_deg: Optional[float] = None
        rot_deg: Optional[float] = None
        wcs_info: Any = None
        messages: List[str] = field(default_factory=list)
        debug: Dict[str, Any] = field(default_factory=dict)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlateSolveConfig:
    """All tuneable parameters for the plate-solving step.

    This will live in measure/types.py once ported.
    """

    backend: str = "local"          # "local" | "astroquery"

    # --- local backend (astrometry.net solve-field) ---
    solve_field_bin: str = "solve-field"  # full path or "wsl solve-field" for WSL2
    timeout_s: int = 120

    # --- search hints (override what's in the FITS header) ---
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    search_radius_deg: float = 5.0

    scale_low_arcsec_per_px: Optional[float] = None
    scale_high_arcsec_per_px: Optional[float] = None

    # --- astroquery backend ---
    api_key: Optional[str] = None   # nova.astrometry.net API key

    # --- misc ---
    downsample: int = 2             # passed to solve-field --downsample
    extra_args: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API  (these functions port directly into measure/platesolve.py)
# ---------------------------------------------------------------------------

def load_fits(path: Path | str) -> Tuple[np.ndarray, fits.Header]:
    """Load the primary HDU of a FITS file and return (float32 array, header)."""
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    return data, header


def solve_field(
    fits_path: Path | str,
    config: Optional[PlateSolveConfig] = None,
    *,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    scale_arcsec_per_px: Optional[float] = None,
    scale_tolerance: float = 0.25,
) -> PlateSolveResult:
    """Plate-solve a FITS image and return a structured result.

    Parameters
    ----------
    fits_path :
        Path to the FITS image.
    config :
        Solver settings.  Defaults are used when omitted.
    ra_deg, dec_deg :
        Per-call hint coordinates (override ``config`` values).
    scale_arcsec_per_px :
        Per-call pixel scale hint.  Sets symmetric ±``scale_tolerance``
        bounds around this value (e.g. 0.25 → ±25 %).
    scale_tolerance :
        Fractional half-range around ``scale_arcsec_per_px``.

    Returns
    -------
    PlateSolveResult
        ``success=True`` iff the solver converged and the WCS was parsed.
        ``result.debug["sources"]`` always contains the detected source
        positions (x, y, flux arrays) for use by ``summarize_result``.
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        res = PlateSolveResult()
        res.messages.append(f"FITS file not found: {fits_path}")
        return res

    cfg = copy.copy(config) if config is not None else PlateSolveConfig()

    # Per-call overrides
    if ra_deg is not None:
        cfg = _dc_replace(cfg, ra_deg=ra_deg)
    if dec_deg is not None:
        cfg = _dc_replace(cfg, dec_deg=dec_deg)
    if scale_arcsec_per_px is not None:
        lo = scale_arcsec_per_px * (1.0 - scale_tolerance)
        hi = scale_arcsec_per_px * (1.0 + scale_tolerance)
        cfg = _dc_replace(cfg, scale_low_arcsec_per_px=lo, scale_high_arcsec_per_px=hi)

    # Fill missing hints from the FITS header before dispatching
    cfg = _hints_from_header(fits_path, cfg)

    # Detect sources now so both backends and summarize_result can use them.
    # We load the image once here rather than inside each backend.
    data, _ = load_fits(fits_path)
    sources = detect_sources(data)

    if cfg.backend == "local":
        result = _solve_local(fits_path, cfg)
    elif cfg.backend == "astroquery":
        result = _solve_astroquery(fits_path, cfg, sources=sources)
    else:
        result = PlateSolveResult()
        result.messages.append(f"Unknown backend: {cfg.backend!r}")

    # Store lightweight debug info for notebook helpers
    result.debug["fits_path"] = str(fits_path)
    result.debug["sources"] = sources
    return result


def load_wcs(result: PlateSolveResult) -> Optional[WCS]:
    """Return an astropy WCS object from a solved PlateSolveResult.

    Returns None if the solve failed or no WCS header is stored.
    """
    if not result.success or result.wcs_info is None:
        return None
    if isinstance(result.wcs_info, fits.Header):
        return WCS(result.wcs_info)
    if isinstance(result.wcs_info, WCS):
        return result.wcs_info
    return None


def extract_rot_deg(wcs: WCS) -> float:
    """Extract the detector rotation angle (degrees, N-up = 0, E-left = 90).

    Projects the image +Y axis onto the sky and returns its position angle
    East of North.  This is the quantity that matches ``RenderConfig.rot_deg``
    in the simulator.
    """
    return _extract_rot_from_wcs(wcs)


def detect_sources(
    image: np.ndarray,
    *,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    sharplo: float = 0.05,
    sharphi: float = 2.0,
    roundlo: float = -2.0,
    roundhi: float = 2.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Detect point sources in an image and return pixel coordinates.

    Uses DAOStarFinder with relaxed sharpness/roundness bounds (important for
    diffraction-modified PSFs).  Falls back to simple peak-finding if
    DAOStarFinder returns nothing.

    Parameters
    ----------
    image :
        2-D float array (background not subtracted).
    fwhm :
        Expected FWHM of the PSF in pixels.
    threshold_sigma :
        Detection threshold in units of background RMS.
    sharplo, sharphi, roundlo, roundhi :
        DAOStarFinder morphology filters — deliberately wide defaults so
        diffraction spikes and elongated PSFs are not rejected.

    Returns
    -------
    dict with keys ``x``, ``y``, ``flux`` (numpy arrays, pixel coords),
    or ``None`` if photutils is not installed.
    """
    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder, find_peaks
    except ImportError:
        log.warning("photutils not installed — source detection skipped.  pip install photutils")
        return None

    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    sub = image - median

    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_sigma * std,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
        peakmax=None,
    )
    table = finder(sub)

    if table is None or len(table) == 0:
        # Fall back to raw peak-finding (no morphology filtering)
        table = find_peaks(sub, threshold=threshold_sigma * std, box_size=max(3, int(fwhm * 2)))
        if table is None or len(table) == 0:
            return {"x": np.array([]), "y": np.array([]), "flux": np.array([])}
        x = table["x_peak"].data.astype(float)
        y = table["y_peak"].data.astype(float)
        flux = table["peak_value"].data.astype(float)
    else:
        x    = table["xcentroid"].data.astype(float)
        y    = table["ycentroid"].data.astype(float)
        flux = table["flux"].data.astype(float)

    return {"x": x, "y": y, "flux": flux}


# ---------------------------------------------------------------------------
# Local backend — astrometry.net solve-field
# ---------------------------------------------------------------------------

def _solve_local(fits_path: Path, cfg: PlateSolveConfig) -> PlateSolveResult:
    res = PlateSolveResult()
    res.debug["backend"] = "local"

    if not _binary_available(cfg.solve_field_bin):
        res.messages.append(
            f"solve-field not found: {cfg.solve_field_bin!r}\n"
            "  Linux/WSL: apt install astrometry.net && apt install astrometry.net-data-*\n"
            "  macOS: brew install astrometry-net\n"
            "  Windows: use backend='astroquery' or run via WSL2 and set\n"
            "           PlateSolveConfig(solve_field_bin='wsl solve-field')"
        )
        return res

    with tempfile.TemporaryDirectory(prefix="platesolve_") as _tmp:
        tmpdir = Path(_tmp)
        args = _build_solve_field_args(fits_path, cfg, tmpdir)
        res.debug["cmd"] = " ".join(str(a) for a in args)

        stdout, stderr, returncode = _run_subprocess(args, cfg.timeout_s)
        res.debug["stdout"] = stdout
        res.debug["stderr"] = stderr
        res.debug["returncode"] = returncode

        stem = fits_path.stem
        solved_file = tmpdir / f"{stem}.solved"
        wcs_file    = tmpdir / f"{stem}.wcs"

        if not solved_file.exists():
            res.messages.append(
                "solve-field produced no .solved file — field not solved.\n"
                "  Common causes: no index files, bad hints, faint stars.\n"
                "  Run summarize_result(res) and show_debug_info(res) to inspect output."
            )
            return res

        if not wcs_file.exists():
            res.messages.append(".solved exists but .wcs is missing (unexpected).")
            return res

        _parse_wcs_into(wcs_file, res)
    return res


def _build_solve_field_args(
    fits_path: Path, cfg: PlateSolveConfig, out_dir: Path
) -> List[str]:
    args: List[str] = [cfg.solve_field_bin]

    # Suppress files we don't need
    args += [
        "--no-plots",
        "--overwrite",
        "--new-fits", "none",
        "--rdls",     "none",
        "--match",    "none",
        "--corr",     "none",
    ]

    args += ["-D", str(out_dir)]
    args += ["--downsample", str(cfg.downsample)]

    if cfg.ra_deg is not None and cfg.dec_deg is not None:
        args += [
            "--ra",     f"{cfg.ra_deg:.6f}",
            "--dec",    f"{cfg.dec_deg:.6f}",
            "--radius", f"{cfg.search_radius_deg:.3f}",
        ]

    if cfg.scale_low_arcsec_per_px is not None:
        args += [
            "--scale-units", "arcsecperpix",
            "--scale-low",   f"{cfg.scale_low_arcsec_per_px:.4f}",
            "--scale-high",  f"{cfg.scale_high_arcsec_per_px:.4f}",
        ]

    args += cfg.extra_args
    args.append(str(fits_path))
    return args


def _run_subprocess(
    args: List[str], timeout_s: int
) -> Tuple[str, str, int]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timed out after {timeout_s}s", -1
    except FileNotFoundError as exc:
        return "", str(exc), -2


def _parse_wcs_into(wcs_file: Path, res: PlateSolveResult) -> None:
    """Parse a .wcs FITS file from solve-field into res (mutates in place)."""
    try:
        with fits.open(wcs_file) as hdul:
            wcs_header = hdul[0].header.copy()
        wcs = WCS(wcs_header)
        ra, dec = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
        rot = _extract_rot_from_wcs(wcs)
        res.success = True
        res.ra_deg  = ra
        res.dec_deg = dec
        res.rot_deg = rot
        res.wcs_info = wcs_header   # raw header; use load_wcs() to get WCS object
        res.messages.append(
            f"Solved: RA={ra:.5f}°  Dec={dec:.5f}°  rot={rot:.3f}°"
        )
    except Exception as exc:
        res.messages.append(f"WCS parse error: {exc}")


# ---------------------------------------------------------------------------
# astroquery backend — nova.astrometry.net online API
# ---------------------------------------------------------------------------

def _solve_astroquery(
    fits_path: Path,
    cfg: PlateSolveConfig,
    sources: Optional[Dict[str, np.ndarray]] = None,
) -> PlateSolveResult:
    res = PlateSolveResult()
    res.debug["backend"] = "astroquery"

    try:
        from astroquery.astrometry_net import AstrometryNet  # type: ignore
    except ImportError:
        res.messages.append(
            "astroquery is not installed.  Run: pip install astroquery"
        )
        return res

    if not cfg.api_key:
        res.messages.append(
            "PlateSolveConfig.api_key is required for the astroquery backend.\n"
            "  Get a free key at https://nova.astrometry.net/api/profile"
        )
        return res

    try:
        an = AstrometryNet()
        an.api_key = cfg.api_key

        kwargs: Dict[str, Any] = {}
        if cfg.ra_deg is not None:
            kwargs["center_ra"]  = cfg.ra_deg
            kwargs["center_dec"] = cfg.dec_deg
            kwargs["radius"]     = cfg.search_radius_deg
        if cfg.scale_low_arcsec_per_px is not None:
            kwargs["scale_lower"] = cfg.scale_low_arcsec_per_px
            kwargs["scale_upper"] = cfg.scale_high_arcsec_per_px
            kwargs["scale_units"] = "arcsecperpix"

        # Pass our own source table to bypass astroquery's internal DAOStarFinder,
        # which crashes when it returns None on unusual PSFs (diffraction images).
        if sources is not None and len(sources.get("x", [])) > 0:
            with fits.open(fits_path) as hdul:
                ny, nx = hdul[0].data.shape[-2], hdul[0].data.shape[-1]
            xs, ys, fluxes = sources["x"], sources["y"], sources["flux"]
            if len(xs) > 300:
                top = np.argsort(fluxes)[::-1][:300]
                xs, ys = xs[top], ys[top]
            res.messages.append(f"Submitting {len(xs)} pre-detected sources to nova.astrometry.net")
            wcs_header = an.solve_from_source_list(
                xs, ys,
                nx, ny,
                **kwargs,
            )
        else:
            res.messages.append("No sources detected — submitting raw image to nova.astrometry.net")
            wcs_header = an.solve_from_image(str(fits_path), **kwargs)

        if wcs_header is None:
            res.messages.append("astroquery returned no solution.")
            return res

        wcs = WCS(wcs_header)
        ra  = float(wcs.wcs.crval[0])
        dec = float(wcs.wcs.crval[1])
        rot = _extract_rot_from_wcs(wcs)
        res.success  = True
        res.ra_deg   = ra
        res.dec_deg  = dec
        res.rot_deg  = rot
        res.wcs_info = wcs_header
        res.messages.append(
            f"Solved: RA={ra:.5f}°  Dec={dec:.5f}°  rot={rot:.3f}°"
        )
    except Exception as exc:
        res.messages.append(f"astroquery error: {exc}")
    return res


# ---------------------------------------------------------------------------
# WCS internals
# ---------------------------------------------------------------------------

def _extract_rot_from_wcs(wcs: WCS) -> float:
    """Position angle of the detector's +Y axis, East of North (degrees)."""
    shape = wcs.pixel_shape or (1000, 1000)
    cx, cy = shape[0] / 2.0, shape[1] / 2.0
    pts = wcs.all_pix2world([[cx, cy], [cx, cy + 1.0]], 0)
    ra0, dec0 = pts[0, 0], pts[0, 1]
    ra1, dec1 = pts[1, 0], pts[1, 1]
    dra  = (ra1 - ra0) * np.cos(np.radians(dec0))
    ddec = dec1 - dec0
    return float(np.degrees(np.arctan2(-dra, ddec)))


# ---------------------------------------------------------------------------
# Header-hint extraction
# ---------------------------------------------------------------------------

def _hints_from_header(fits_path: Path, cfg: PlateSolveConfig) -> PlateSolveConfig:
    """Promote FITS header values to cfg fields only when cfg has no value."""
    if cfg.ra_deg is not None and cfg.dec_deg is not None and cfg.scale_low_arcsec_per_px is not None:
        return cfg
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
        updates: Dict[str, Any] = {}
        if cfg.ra_deg is None:
            ra = hdr.get("RA0") or hdr.get("CRVAL1") or hdr.get("RA")
            if ra is not None:
                updates["ra_deg"] = float(ra)
        if cfg.dec_deg is None:
            dec = hdr.get("DEC0") or hdr.get("CRVAL2") or hdr.get("DEC")
            if dec is not None:
                updates["dec_deg"] = float(dec)
        if cfg.scale_low_arcsec_per_px is None:
            ps = hdr.get("PLTSCL") or hdr.get("PIXSCL") or hdr.get("PIXSCALE")
            if ps is not None:
                ps = float(ps)
                updates["scale_low_arcsec_per_px"]  = ps * 0.75
                updates["scale_high_arcsec_per_px"] = ps * 1.25
        return _dc_replace(cfg, **updates) if updates else cfg
    except Exception:
        return cfg


def _binary_available(name: str) -> bool:
    # handles "wsl solve-field" as well as plain names / absolute paths
    parts = name.split()
    return shutil.which(parts[0]) is not None or Path(parts[0]).is_file()


# ---------------------------------------------------------------------------
# Notebook helpers  (display / debug — keep below this line when porting)
# ---------------------------------------------------------------------------

def summarize_result(
    result: PlateSolveResult,
    *,
    stretch: str = "asinh",
    figsize: Tuple[int, int] = (10, 8),
    circle_radius: int = 12,
) -> None:
    """Print a solve summary and display the image with detected sources circled.

    Source positions are taken from ``result.debug["sources"]`` which is
    populated automatically by ``solve_field``.  The original FITS image is
    reloaded from ``result.debug["fits_path"]``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # --- text summary ---
    status = "SOLVED" if result.success else "FAILED"
    print(f"Status : {status}")
    if result.success:
        print(f"RA     : {result.ra_deg:.5f}°")
        print(f"Dec    : {result.dec_deg:.5f}°")
        print(f"Rot    : {result.rot_deg:.4f}°")
    if result.messages:
        print("Messages:")
        for m in result.messages:
            for line in m.splitlines():
                print(f"  {line}")
    dbg_cmd = result.debug.get("cmd")
    if dbg_cmd:
        print(f"Command: {dbg_cmd}")

    # --- image display with source overlay ---
    fits_path = result.debug.get("fits_path")
    sources   = result.debug.get("sources")
    if fits_path is None:
        return  # result was not produced by solve_field (e.g. stub)

    try:
        data, _ = load_fits(fits_path)
    except Exception as exc:
        print(f"(Could not reload image for display: {exc})")
        return

    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [0.5, 99.5])
    if stretch == "asinh":
        disp = np.arcsinh(np.clip(data, vmin, vmax))
        lo, hi = np.arcsinh(vmin), np.arcsinh(vmax)
    elif stretch == "log":
        floor = max(float(vmin), 1e-6)
        disp = np.log10(np.clip(data, floor, vmax))
        lo, hi = np.log10(floor), np.log10(vmax)
    else:
        disp, lo, hi = data, vmin, vmax

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(disp, origin="lower", cmap="gray", vmin=lo, vmax=hi)

    n_src = 0
    if sources is not None:
        xs, ys, fluxes = sources.get("x", []), sources.get("y", []), sources.get("flux", [])
        n_src = len(xs)
        if n_src > 0:
            order = np.argsort(fluxes)[::-1][:100]
            for i in order:
                ax.add_patch(Circle(
                    (xs[i], ys[i]), radius=circle_radius,
                    edgecolor="lime", facecolor="none", linewidth=0.8, alpha=0.75,
                ))

    suffix = f" — {n_src} sources detected" if n_src else " — no sources detected"
    ax.set_title(f"{Path(fits_path).name}{suffix}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_debug_info(result: PlateSolveResult) -> None:
    """Dump raw subprocess stdout/stderr for diagnosing a failed solve."""
    print("── stdout " + "─" * 60)
    print(result.debug.get("stdout") or "(empty)")
    print("── stderr " + "─" * 60)
    print(result.debug.get("stderr") or "(empty)")
    print(f"── returncode: {result.debug.get('returncode')} " + "─" * 47)


def show_image(
    image: np.ndarray,
    *,
    title: str = "Image",
    stretch: str = "asinh",
    figsize: Tuple[int, int] = (10, 8),
    wcs: Optional[WCS] = None,
) -> None:
    """Display a FITS image with optional WCS sky-coordinate grid."""
    import matplotlib.pyplot as plt

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=subplot_kw)

    finite = image[np.isfinite(image)]
    vmin, vmax = np.percentile(finite, [0.5, 99.5])

    if stretch == "asinh":
        disp = np.arcsinh(np.clip(image, vmin, vmax))
        lo, hi = np.arcsinh(vmin), np.arcsinh(vmax)
    elif stretch == "log":
        floor = max(vmin, 1e-6)
        disp = np.log10(np.clip(image, floor, vmax))
        lo, hi = np.log10(floor), np.log10(vmax)
    else:
        disp, lo, hi = image, vmin, vmax

    ax.imshow(disp, origin="lower", cmap="gray", vmin=lo, vmax=hi)

    if wcs is not None:
        ax.coords.grid(True, color="cyan", alpha=0.45, linestyle="--", linewidth=0.7)
        ax.coords["ra"].set_axislabel("RA")
        ax.coords["dec"].set_axislabel("Dec")

    ax.set_title(title)
    ax.axis("off") if wcs is None else None
    plt.tight_layout()
    plt.show()


def compare_header_vs_solved(
    result: PlateSolveResult,
    header: fits.Header,
) -> None:
    """Print side-by-side comparison of header hints vs solved values."""
    hdr_ra  = header.get("RA0") or header.get("CRVAL1") or header.get("RA")
    hdr_dec = header.get("DEC0") or header.get("CRVAL2") or header.get("DEC")
    hdr_rot = header.get("ROT") or header.get("ROLL")
    hdr_ps  = header.get("PLTSCL") or header.get("PIXSCL")

    def _fmt(v, unit="°"):
        return f"{float(v):.5f}{unit}" if v is not None else "—"

    print(f"{'':18s} {'Header':>14s}   {'Solved':>14s}")
    print("─" * 50)
    print(f"{'RA':18s} {_fmt(hdr_ra):>14s}   {_fmt(result.ra_deg):>14s}")
    print(f"{'Dec':18s} {_fmt(hdr_dec):>14s}   {_fmt(result.dec_deg):>14s}")
    print(f"{'Rot':18s} {_fmt(hdr_rot):>14s}   {_fmt(result.rot_deg):>14s}")
    print(f"{'Plate scale':18s} {_fmt(hdr_ps, ' \"/px'):>14s}   {'(from WCS)':>14s}")
