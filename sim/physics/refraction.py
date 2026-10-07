# sim/physics/refraction.py
"""
Atmospheric refraction for the vector-grating renderer (opt-in).

Light from a star is refracted by the atmosphere BEFORE it reaches the payload's grating,
so for a diffracted sample at wavelength lambda the grating acts on the star's apparent
direction AT THAT WAVELENGTH. The renderer therefore refracts:
  - every star's incident direction per wavelength sample, before the vector grating
    equation, for the diffracted orders;
  - the zeroth-order (broadband) star at a single reference wavelength.

Model (deliberately simple; it is the geometry that matters for testing the estimator):
  * refractivity of dry air, Edlen (1966) at 15 C / 101325 Pa:
        (n - 1) 1e8 = 8342.13 + 2406030 / (130 - s^2) + 15997 / (38.9 - s^2),  s = 1 / lambda[um]
    scaled to the given pressure and temperature by the ideal-gas density ratio;
  * plane-parallel refraction: an apparent direction is the true one rotated toward the zenith
    by R = (n - 1) tan z, where z is the true zenith distance.
At z = 45 deg this gives R ~ 58" at sea level and ~30" at a 5 km site (530 hPa, 0 C); the
dispersion between 400 and 700 nm is ~1.5" and ~0.8" respectively.
"""
from __future__ import annotations

import numpy as np


def n_air_minus_1(lam_nm, pressure_hpa: float = 1013.25, temperature_c: float = 15.0,
                  relative_humidity: float = 0.0):
    """Edlen (1966) dry air, ideal-gas scaled, plus (optionally) Edlen's water-vapour term
    -f (3.7345 - 0.0401 s^2) 1e-10, with f the partial pressure of water vapour in Pa from the
    relative humidity (Magnus saturation pressure). Humidity is a simulator-side detail that the
    estimator does not model."""
    s2 = (1.0e3 / np.asarray(lam_nm, float)) ** 2
    nm1 = 1e-8 * (8342.13 + 2406030.0 / (130.0 - s2) + 15997.0 / (38.9 - s2))
    nm1 = nm1 * (pressure_hpa / 1013.25) * (288.15 / (273.15 + temperature_c))
    if relative_humidity:
        e_sat_hpa = 6.1094 * np.exp(17.625 * temperature_c / (temperature_c + 243.04))
        f_pa = float(relative_humidity) * e_sat_hpa * 100.0
        nm1 = nm1 - f_pa * (3.7345 - 0.0401 * s2) * 1e-10
    return nm1


def refract(d, zenith, lam_nm, pressure_hpa: float = 1013.25, temperature_c: float = 15.0,
            relative_humidity: float = 0.0):
    """True directions d (..., 3) -> apparent directions, rotated toward `zenith` (3,) by
    (n(lam) - 1) tan z. lam_nm broadcasts against d[..., 0]."""
    d = np.asarray(d, float)
    Z = np.asarray(zenith, float) / np.linalg.norm(zenith)
    cz = np.clip(d @ Z, -1.0, 1.0)
    R = n_air_minus_1(lam_nm, pressure_hpa, temperature_c, relative_humidity) \
        * np.sqrt(1.0 - cz ** 2) / np.maximum(cz, 1e-6)
    t = Z - cz[..., None] * d
    tn = np.linalg.norm(t, axis=-1, keepdims=True)
    t = np.where(tn > 1e-15, t / np.maximum(tn, 1e-300), 0.0)
    R = R[..., None]
    out = d * np.cos(R) + t * np.sin(R)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def zenith_from_field(b_icrs, zenith_distance_deg: float, zenith_pa_deg: float):
    """ICRS zenith vector for a field whose centre b_icrs sits at `zenith_distance_deg` from
    the zenith, with the zenith lying toward position angle `zenith_pa_deg` (east of north)
    as seen from the field centre."""
    b = np.asarray(b_icrs, float) / np.linalg.norm(b_icrs)
    z = np.array([0.0, 0.0, 1.0])
    e_E = np.cross(z, b)
    e_E = e_E / np.linalg.norm(e_E) if np.linalg.norm(e_E) > 1e-12 else np.array([0.0, 1.0, 0.0])
    e_N = np.cross(b, e_E)
    a, q = np.radians(zenith_distance_deg), np.radians(zenith_pa_deg)
    return np.cos(a) * b + np.sin(a) * (np.cos(q) * e_N + np.sin(q) * e_E)
