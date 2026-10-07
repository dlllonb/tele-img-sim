# sim/physics/stellar_colour.py
"""
Stellar colours for the vector-grating renderer (opt-in).

Each star gets an effective temperature from its Gaia BP-RP colour and a blackbody PHOTON
spectrum. This changes three things relative to the colourless default, in which every star has
the same spectral weighting across the band:
  1. the distribution of a star's diffracted light along its spectrum (the per-wavelength
     weights of its order samples);
  2. its flux in the camera band relative to its Gaia G magnitude (a colour term, normalised so
     a solar-colour star is unchanged);
  3. the effective wavelength of its broadband zeroth-order image, which matters once
     wavelength-dependent effects (refraction, lateral colour) are on: the zeroth-order image
     then sits at the flux-weighted mean of its per-wavelength positions.

Approximations, which are deliberate (the purpose is a realistic SPREAD of stellar spectra, not
precise stellar physics):
  * Teff from BP-RP with the Mucciarelli & Bellazzini (2020) dwarf relation at solar
    metallicity, theta = 5040 / Teff = 0.4929 + 0.5092 C - 0.0353 C^2, clipped to 2500-40000 K;
    stars without a colour are treated as solar (BP-RP = 0.82);
  * blackbody spectra; no extinction;
  * Gaia G approximated as a flat photon passband over 330-1050 nm.
"""
from __future__ import annotations

import numpy as np

BPRP_SOLAR = 0.82
_H, _C, _K = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def teff_from_bprp(bp_rp):
    c = np.asarray(bp_rp, dtype=float)
    c = np.where(np.isfinite(c), c, BPRP_SOLAR)
    c = np.clip(c, -0.3, 4.0)
    theta = 0.4929 + 0.5092 * c - 0.0353 * c ** 2
    return np.clip(5040.0 / theta, 2500.0, 40000.0)


def photon_sed(lam_nm, teff):
    """Blackbody photon spectral radiance, arbitrary units (broadcasts lam_nm against teff)."""
    lam = np.asarray(lam_nm, dtype=float) * 1e-9
    x = _H * _C / (lam * _K * np.asarray(teff, dtype=float))
    return lam ** -4 / np.expm1(np.minimum(x, 700.0))


def band_weights(lam_nm, base_w, teff):
    """Per-star spectral weights (N, L) over wavelength samples lam_nm (L,): the colourless base
    weights (L,), which sum to 1, times the star's photon SED, renormalised so each row sums to 1.
    A star's total diffracted flux is unchanged; only its distribution along the spectrum is."""
    w = np.asarray(base_w, float)[None, :] * photon_sed(np.asarray(lam_nm, float)[None, :],
                                                       np.asarray(teff, float)[:, None])
    return w / w.sum(axis=1, keepdims=True)


def colour_flux_factor(teff, lam_nm, base_w):
    """Camera-band photons per Gaia-G photon, relative to a solar-colour star. lam_nm and base_w
    describe the camera band (the renderer's wavelength samples and their colourless weights)."""
    teff = np.atleast_1d(np.asarray(teff, dtype=float))
    lg = np.linspace(330.0, 1050.0, 145)

    def ratio(t):
        cam = (np.asarray(base_w)[None, :] * photon_sed(np.asarray(lam_nm)[None, :], t[:, None])).sum(1)
        g = photon_sed(lg[None, :], t[:, None]).sum(1)
        return cam / g

    return ratio(teff) / ratio(np.array([float(teff_from_bprp(BPRP_SOLAR))]))[0]


def effective_wavelength_nm(lam_nm, base_w, teff):
    """Flux-weighted mean wavelength of each star's light across the band."""
    W = band_weights(lam_nm, base_w, teff)
    return (W * np.asarray(lam_nm, float)[None, :]).sum(axis=1)
