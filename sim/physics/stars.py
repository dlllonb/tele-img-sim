# sim/physics/stars.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import math


@dataclass(frozen=True)
class StarField:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray
    meta: dict | None = None

    def __post_init__(self):
        n = len(self.ra_deg)
        if len(self.dec_deg) != n or len(self.mag) != n:
            raise ValueError("StarField arrays must have the same length.")


def load_star_field(path: str | Path) -> StarField:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Star field file not found: {path}")

    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    cols = set(arr.dtype.names or ())

    if {"ra_deg", "dec_deg", "mag"}.issubset(cols):
        ra = np.array(arr["ra_deg"], dtype=float)
        dec = np.array(arr["dec_deg"], dtype=float)
        mag = np.array(arr["mag"], dtype=float)
    elif {"ra", "dec", "phot_g_mean_mag"}.issubset(cols):
        ra = np.array(arr["ra"], dtype=float)
        dec = np.array(arr["dec"], dtype=float)
        mag = np.array(arr["phot_g_mean_mag"], dtype=float)
    else:
        raise ValueError(f"Unrecognized star field columns: {arr.dtype.names}")

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
    ra, dec, mag = ra[m], dec[m], mag[m]

    return StarField(ra_deg=ra, dec_deg=dec, mag=mag, meta={"source": str(path)})


def derive_zeropoint_e_per_s(lens, qe: float, lambda_eff_nm: float, band_nm: float) -> float:
    """
    Crude broadband photometric zeropoint: mag=0 -> detected e-/s.

    Scales with aperture area (via f_number), lens transmission, and QE.
    Assumes a V-ish mag=0 flux density at ~550 nm. Engineering approximation.
    """
    h = 6.62607015e-34  # J*s
    c = 2.99792458e8    # m/s

    lam = lambda_eff_nm * 1e-9
    dlam = band_nm * 1e-9

    # Rough mag=0 spectral irradiance near 550 nm:
    # ~3.6e-11 W/m^2/nm => 3.6e-2 W/m^2/m
    F0_W_m2_m = 3.6e-2

    f_m = lens.focal_mm * 1e-3
    D_m = f_m / lens.f_number
    A_m2 = math.pi * (0.5 * D_m) ** 2

    E_ph = h * c / lam
    photons_s_m2 = (F0_W_m2_m * dlam) / E_ph

    zp = photons_s_m2 * A_m2 * lens.transmission * qe
    return float(zp)


StarsInput = Union[None, str, Path, StarField]


def stars_layer(frame, stars: StarsInput, cfg, rng=None):
    """
    Stars-only expected electrons image (float), pre-PSF.

    Steps:
      - resolve StarField (file or object)
      - compute per-star expected electrons from mag using derived or provided zeropoint
      - map RA/Dec -> (x_px, y_px)
      - cut to frame (+margin)
      - bilinear splat into pixels
    """
    if rng is None:
        rng = np.random.default_rng(getattr(cfg, "seed", 0))

    if stars is None:
        return np.zeros_like(frame.image, dtype=np.float32)

    if isinstance(stars, StarField):
        sf = stars
    else:
        sf = load_star_field(stars)

    # Zeropoint: if cfg provides override use it, else derive from optics
    if getattr(cfg, "zeropoint_e_per_s", 0.0) and cfg.zeropoint_e_per_s > 0.0:
        zp_e_s = float(cfg.zeropoint_e_per_s)
    else:
        zp_e_s = derive_zeropoint_e_per_s(
            frame.lens,
            qe = float(getattr(frame.camera, "qe", 1.0)),
            lambda_eff_nm=float(getattr(cfg, "lambda_eff_nm", 550.0)),
            band_nm=float(getattr(cfg, "band_nm", 90.0)),
        )

    t = float(cfg.exposure_s)

    # mag -> expected electrons (mean, not noisy)
    # N_e = t * ZP * 10^(-0.4 m)
    mag = sf.mag.astype(float, copy=False)
    flux_e = (t * zp_e_s) * np.power(10.0, -0.4 * mag)

    # Map to pixel coords (floats)
    x_px, y_px = frame.radec_to_pixel(sf.ra_deg, sf.dec_deg)

    ny, nx = frame.image.shape

    # Cut to on-sensor with a small margin (for PSF wings later)
    margin = 2.0  # px (small for now)
    keep = (x_px >= -margin) & (x_px <= (nx - 1) + margin) & (y_px >= -margin) & (y_px <= (ny - 1) + margin)
    x_px = x_px[keep]
    y_px = y_px[keep]
    flux_e = flux_e[keep]

    stars_e = np.zeros((ny, nx), dtype=np.float32)

    if x_px.size == 0:
        return stars_e

    # Bilinear splatting into 4 nearest pixels
    ix = np.floor(x_px).astype(int)
    iy = np.floor(y_px).astype(int)

    fx = (x_px - ix).astype(np.float32)
    fy = (y_px - iy).astype(np.float32)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    # Helper to add safely (bounds check)
    def add_weighted(dx: int, dy: int, w: np.ndarray):
        x = ix + dx
        y = iy + dy
        m = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        if not np.any(m):
            return
        np.add.at(stars_e, (y[m], x[m]), (flux_e[m] * w[m]).astype(np.float32))

    add_weighted(0, 0, w00)
    add_weighted(1, 0, w10)
    add_weighted(0, 1, w01)
    add_weighted(1, 1, w11)

    return stars_e