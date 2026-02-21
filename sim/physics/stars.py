# sim/physics/stars.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class StarField:
    """
    Minimal star catalog slice.

    Arrays are 1D and same length.
    Units:
      ra_deg, dec_deg: degrees (ICRS)
      mag: dimensionless (Gaia G if fetched via fetch_gaia_field)
    """
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray
    meta: dict | None = None

    def __post_init__(self):
        n = len(self.ra_deg)
        if len(self.dec_deg) != n or len(self.mag) != n:
            raise ValueError("StarField arrays must have the same length.")


def load_star_field(path: str | Path) -> StarField:
    """
    Load a star field CSV.

    Supports:
      - ra_deg, dec_deg, mag
      - ra, dec, phot_g_mean_mag (Gaia default)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Star field file not found: {path}")

    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)

    colnames = arr.dtype.names

    # Determine column mapping
    if {"ra_deg", "dec_deg", "mag"}.issubset(colnames):
        ra = np.array(arr["ra_deg"], dtype=float)
        dec = np.array(arr["dec_deg"], dtype=float)
        mag = np.array(arr["mag"], dtype=float)

    elif {"ra", "dec", "phot_g_mean_mag"}.issubset(colnames):
        ra = np.array(arr["ra"], dtype=float)
        dec = np.array(arr["dec"], dtype=float)
        mag = np.array(arr["phot_g_mean_mag"], dtype=float)

    else:
        raise ValueError(
            f"Unrecognized star field format. Columns found: {colnames}"
        )

    return StarField(
        ra_deg=ra,
        dec_deg=dec,
        mag=mag,
        meta={"source": str(path)}
    )


StarsInput = Union[None, str, Path, StarField]
def stars_layer(frame, stars: StarsInput, cfg, rng=None):
    """
    Produce a stars-only image layer in electrons (float), typically pre-PSF.

    Input handling only (no placement/cutting yet):
      - None -> no stars
      - str/Path -> loads CSV
      - StarField -> uses directly
    """
    if rng is None:
        rng = np.random.default_rng(getattr(cfg, "seed", 0))

    if stars is None:
        starfield = None
    elif isinstance(stars, StarField):
        starfield = stars
    else:
        starfield = load_star_field(stars)

    # Next steps (intentionally not implemented yet):
    # - filter to frame footprint using radec_to_pixel
    # - mag -> electrons using cfg.zeropoint_e_per_s and cfg.exposure_s
    # - subpixel splatting (bilinear) into a delta image
    # - return that delta image (pre-PSF)

    return np.zeros_like(frame.image, dtype=np.float32)