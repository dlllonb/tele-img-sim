# sim/physics/vector_grating.py
"""
Physical ("vector grating") rendering of diffraction traces.

Why this exists
---------------
The legacy analytic grating model (`masks._kernel_grating_orders`) builds
ONE shift-invariant kernel -- straight segments at a fixed pixel angle
`mask.angle_deg`, separation f*m*lambda/pitch -- and FFT-convolves the
whole star image with it. Real gratings don't do that: a planar grating
conserves each ray's direction component along the wires (conical
diffraction), d_out . w_hat == d_in . w_hat, so every order of a star lies
on a small circle about w_hat, and on a flat (gnomonic) sensor each trace
is rotated by ~alpha*beta (its own field position) and very slightly
curved. Every estimator validated only on legacy images was therefore
validated on data that cannot contain this effect.

What this module does (only when `mask.trace_model == "vector_grating"`)
----------------------------------------------------------------------
For each star (pinhole camera, boresight +z; a star's camera-frame
direction is exactly unit((x-cx)*ps, (y-cy)*ps, 1) because the simulator's
projection is exact gnomonic):
  1. apply the exact vector grating equation for m = +/-1..order_max at
     the SAME wavelength samples and SAME raised-cosine weights the legacy
     kernel uses (so the only difference from legacy is geometry);
  2. project each diffracted direction back through the same pinhole
     model used for the stars;
  3. bilinear-splat zeroth order (flux * w_m[0]) and every trace sample
     into the pre-PSF image;
then the PSF stage convolves once with the legacy Moffat spot alone.

`grating_params` deliberately MIRRORS `masks._kernel_grating_orders`'s
parameter arithmetic rather than refactoring that frozen legacy function;
tests/test_vector_grating.py rebuilds the legacy kernel from these
parameters and requires an exact match, so any drift fails loudly.

Grid attitude: in-plane angle `mask.angle_deg` (CCW from +x, same meaning
as legacy) plus an optional camera--grid tilt `mask.tilt_deg` about the
in-plane axis at `mask.tilt_axis_deg`. Zero tilt -> grid normal = boresight.

Physics oracle: tests/conical_reference.py (copied from the project-root
context/ folder).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Parameters (mirror of the legacy analytic kernel's arithmetic)
# ---------------------------------------------------------------------------

def grating_params(frame, cfg, sigma_px: float, mask):
    """Order weights, wavelength sample grid and Moffat spot parameters,
    computed exactly as `masks._kernel_grating_orders` computes them
    (segment-smear path). Returns None when that function would return a
    plain Gaussian (no grating)."""
    lines_per_mm = float(getattr(mask, "lines_per_mm", 0.0))
    order_max = max(int(getattr(mask, "order_max", 1)), 0)
    if lines_per_mm <= 0.0 or order_max <= 0:
        return None
    duty_cycle = float(np.clip(float(getattr(mask, "duty_cycle", 0.5)), 0.01, 0.99))

    lam_eff_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
    band_nm = float(getattr(cfg, "band_nm", 0.0))
    n_lambda = max(int(getattr(mask, "n_lambda", 9)), 1)
    use_segment_smear = bool(getattr(mask, "segment_smear", True))
    if band_nm <= 0.0 or not use_segment_smear or n_lambda <= 1:
        raise NotImplementedError(
            "vector_grating supports only the legacy segment-smear sampling "
            "(band_nm > 0, segment_smear=True, n_lambda > 1)")
    lam0_nm = lam_eff_nm - 0.5 * band_nm
    lam1_nm = lam_eff_nm + 0.5 * band_nm

    pitch_mm = 1.0 / lines_per_mm
    f_mm = float(frame.lens.focal_mm)
    pix_mm = float(frame.camera.pixel_um) * 1e-3

    plate_scale_arcsec_per_px = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0
    fnum = float(frame.lens.f_number)
    D_m = (f_mm / max(fnum, 1e-12)) * 1e-3
    lam_m = lam_eff_nm * 1e-9
    fwhm_diff_arcsec = (1.03 * lam_m / max(D_m, 1e-20)) * 206265.0
    sigma_diff_px = (fwhm_diff_arcsec / max(plate_scale_arcsec_per_px, 1e-20)) / 2.355
    sigma_pix_px = 1.0 / np.sqrt(12.0)
    sigma_seeing_px = float(max(sigma_px, 0.0))
    sigma_eff = float(np.sqrt(sigma_seeing_px ** 2 + sigma_diff_px ** 2 + sigma_pix_px ** 2))
    sigma_floor_px = float(getattr(mask, "sigma_floor_px", 0.0))
    if sigma_floor_px > 0.0:
        sigma_eff = max(sigma_eff, sigma_floor_px)

    def _rect_rel_I(m: int, D: float) -> float:
        if m == 0:
            return D * D
        x = np.pi * m * D
        return (np.sin(x) / (np.pi * m)) ** 2

    order_decay = float(getattr(mask, "order_decay", 0.0))
    order_floor = float(getattr(mask, "order_floor", 0.0))
    order_norm_max = max(int(getattr(mask, "order_norm_max", max(50, order_max))), order_max)
    w_all = np.zeros(order_norm_max + 1, dtype=np.float64)
    for m in range(0, order_norm_max + 1):
        w = _rect_rel_I(m, duty_cycle)
        if m > 0 and order_decay > 0.0:
            w *= np.exp(-m / max(order_decay, 1e-12))
        if m > 0 and order_floor > 0.0:
            w += order_floor
        w_all[m] = max(w, 0.0)
    s_all = float(w_all.sum())
    if s_all <= 0.0:
        return None
    w_all /= s_all
    w_m = np.zeros(order_max + 1, dtype=np.float64)
    w_m[0] = float(w_all[0] + w_all[order_max + 1:].sum())
    w_m[1:] = w_all[1:order_max + 1]

    beta = float(np.clip(float(getattr(mask, "moffat_beta", 3.5)), 1.5, 20.0))
    fwhm_eff = 2.355 * sigma_eff
    denom = 2.0 * np.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
    alpha = max(fwhm_eff / max(denom, 1e-12), 1e-6)
    R = int(np.ceil(max(8.0, 8.0 * fwhm_eff)))

    smear_profile = str(getattr(mask, "smear_profile", "raised_cosine")).lower()
    smear_step_px = float(getattr(mask, "smear_step_px", 0.75))
    smear_step_px = min(smear_step_px, max(0.15, 0.5 * fwhm_eff))
    smear_step_px = float(np.clip(smear_step_px, 0.10, 5.0))
    smear_cap = int(np.clip(int(getattr(mask, "smear_cap", 250)), 8, 2000))

    def _longitudinal_weights(u):
        if smear_profile == "flat":
            w = np.ones_like(u, dtype=np.float64)
        else:
            w = np.sin(np.pi * u) ** 2
        s = float(w.sum())
        if s <= 0.0:
            w[:] = 1.0
            s = float(w.sum())
        return w / s

    lam0_mm, lam1_mm = lam0_nm * 1e-6, lam1_nm * 1e-6
    u_by_m, wu_by_m = {}, {}
    for m in range(1, order_max + 1):
        sep0_px = (f_mm * (m * (lam0_mm / pitch_mm))) / pix_mm
        sep1_px = (f_mm * (m * (lam1_mm / pitch_mm))) / pix_mm
        smear_len_px = float(abs(sep1_px - sep0_px))
        if smear_len_px <= 1e-9:
            u = np.array([0.5], dtype=np.float64)
        else:
            n_seg = int(np.clip(int(np.ceil(smear_len_px / smear_step_px)) + 1, 8, smear_cap))
            u = np.linspace(0.0, 1.0, n_seg, dtype=np.float64)
        u_by_m[m] = u
        wu_by_m[m] = _longitudinal_weights(u)

    return dict(order_max=order_max, w_m=w_m, lam0_nm=lam0_nm, lam1_nm=lam1_nm,
                pitch_mm=pitch_mm, f_mm=f_mm, pix_mm=pix_mm, sigma_eff=sigma_eff,
                beta=beta, alpha=alpha, R=R, u_by_m=u_by_m, wu_by_m=wu_by_m)


def add_moffat(img, x0, y0, w, alpha, beta, R):
    """Patch-normalised Moffat deposit -- identical to the legacy kernel's
    `_add_moffat` closure."""
    x0f, y0f = float(x0), float(y0)
    x_c, y_c = int(np.floor(x0f)), int(np.floor(y0f))
    x_min, x_max = max(0, x_c - R), min(img.shape[1], x_c + R + 1)
    y_min, y_max = max(0, y_c - R), min(img.shape[0], y_c + R + 1)
    if x_min >= x_max or y_min >= y_max:
        return
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float64)
    rr2 = (xx - x0f) ** 2 + (yy - y0f) ** 2
    prof = (1.0 + rr2 / (alpha ** 2)) ** (-beta)
    s = float(prof.sum())
    if s > 0.0:
        img[y_min:y_max, x_min:x_max] += (float(w) / s) * prof


def spot_kernel(p) -> np.ndarray:
    """The legacy grating kernel's PSF spot alone (unit sum), size 2R+1."""
    R = int(p["R"])
    k = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float64)
    add_moffat(k, R, R, 1.0, p["alpha"], p["beta"], R)
    return k / k.sum()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rotation_matrix(axis, angle):
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def grid_triad_camera(angle_deg: float, tilt_deg: float = 0.0, tilt_axis_deg: float = 0.0):
    """(g_hat, w_hat, n_hat) in the camera frame (boresight +z, x/y along
    sensor columns/rows). g_hat at `angle_deg` CCW from +x (same meaning as
    the legacy `mask.angle_deg`); right-handed triad g x w = n."""
    a = np.radians(angle_deg)
    g = np.array([np.cos(a), np.sin(a), 0.0])
    w = np.array([-np.sin(a), np.cos(a), 0.0])
    n = np.array([0.0, 0.0, 1.0])
    if tilt_deg:
        t = np.radians(tilt_axis_deg)
        Rm = _rotation_matrix([np.cos(t), np.sin(t), 0.0], np.radians(tilt_deg))
        g, w, n = Rm @ g, Rm @ w, Rm @ n
    return g, w, n


def diffract(d_in, lam_m, pitch_m, m, g_hat, n_hat):
    """Exact planar-grating vector equation, any order/wavelength/incidence.
    d_in: (N, 3) unit directions toward the source; lam_m: (L,) wavelengths.
    Returns (N, L, 3); NaN where the order is evanescent."""
    d_in = np.asarray(d_in, float)
    dn = d_in @ n_hat                                       # (N,)
    dt = d_in - dn[:, None] * n_hat                         # (N, 3)
    shift = (m * np.asarray(lam_m, float) / pitch_m)[:, None] * g_hat  # (L, 3)
    dt_out = dt[:, None, :] + shift[None, :, :]             # (N, L, 3)
    s2 = 1.0 - np.sum(dt_out ** 2, axis=-1)
    out = dt_out + (np.sqrt(np.clip(s2, 0.0, None)) * np.sign(dn)[:, None])[..., None] * n_hat
    out[s2 <= 0] = np.nan
    return out


def camera_dirs(frame, x_px, y_px) -> np.ndarray:
    ny, nx = frame.image.shape
    ps = float(frame.plate_scale_rad_per_px)
    X = (np.asarray(x_px, float) - (nx - 1) / 2.0) * ps
    Y = (np.asarray(y_px, float) - (ny - 1) / 2.0) * ps
    return _unit(np.stack([X, Y, np.ones_like(X)], axis=-1))


def camera_dirs_to_pixels(frame, d):
    ny, nx = frame.image.shape
    ps = float(frame.plate_scale_rad_per_px)
    d = np.asarray(d, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (nx - 1) / 2.0 + (d[..., 0] / d[..., 2]) / ps
        y = (ny - 1) / 2.0 + (d[..., 1] / d[..., 2]) / ps
    return x, y


def trace_samples(frame, p, mask, x_px, y_px, flux_e):
    """All diffracted-order samples (pixel x, y, electrons) of the given stars."""
    g, w, n = grid_triad_camera(float(getattr(mask, "angle_deg", 0.0)),
                                float(getattr(mask, "tilt_deg", 0.0)),
                                float(getattr(mask, "tilt_axis_deg", 0.0)))
    d_star = camera_dirs(frame, x_px, y_px)
    pitch_m = p["pitch_mm"] * 1e-3
    xs, ys, ws = [], [], []
    for m in range(1, p["order_max"] + 1):
        wm = float(p["w_m"][m])
        if wm <= 0.0:
            continue
        lam_m = (p["lam0_nm"] + p["u_by_m"][m] * (p["lam1_nm"] - p["lam0_nm"])) * 1e-9
        wu = p["wu_by_m"][m]
        for sgn in (+1, -1):
            d_out = diffract(d_star, lam_m, pitch_m, sgn * m, g, n)       # (N, L, 3)
            x, y = camera_dirs_to_pixels(frame, d_out)
            wt = (0.5 * wm) * flux_e[:, None] * wu[None, :]
            ok = np.isfinite(x) & np.isfinite(y) & (d_out[..., 2] > 0)
            xs.append(x[ok]); ys.append(y[ok]); ws.append(wt[ok])
    if not xs:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def splat_bilinear(img, x, y, w):
    """Bilinear splat (same scheme as stars.stars_layer), in place."""
    ny, nx = img.shape
    ix = np.floor(x).astype(int); iy = np.floor(y).astype(int)
    fx = x - ix; fy = y - iy
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = ix + dx, iy + dy
        m = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
        np.add.at(img, (yy[m], xx[m]), (w[m] * ww[m]).astype(img.dtype))
    return img


# ---------------------------------------------------------------------------
# Ground truth in the sky frame
# ---------------------------------------------------------------------------

def _radec_to_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def camera_to_icrs_matrix(frame) -> np.ndarray:
    """Orthogonal 3x3 map from camera-frame directions to ICRS unit vectors
    (exact for the simulator's gnomonic projection; det = -1 if the frame
    has mirror parity). Fit by least squares over a pixel grid."""
    ny, nx = frame.image.shape
    xs, ys = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    xs, ys = xs.ravel(), ys.ravel()
    D_cam = camera_dirs(frame, xs, ys)
    ra, dec = frame.pixel_to_radec(xs, ys)
    D_sky = _radec_to_unit(ra, dec)
    M, *_ = np.linalg.lstsq(D_cam, D_sky, rcond=None)
    return M.T


def truth_vectors(frame, mask) -> dict:
    """Ground-truth grid triad + boresight in ICRS (as lists, JSON-friendly)."""
    M = camera_to_icrs_matrix(frame)
    g, w, n = grid_triad_camera(float(getattr(mask, "angle_deg", 0.0)),
                                float(getattr(mask, "tilt_deg", 0.0)),
                                float(getattr(mask, "tilt_axis_deg", 0.0)))
    ortho_err = float(np.abs(M @ M.T - np.eye(3)).max())
    return dict(trace_model="vector_grating",
                g_hat_icrs=(M @ g).tolist(), w_hat_icrs=(M @ w).tolist(),
                n_hat_icrs=(M @ n).tolist(), b_hat_icrs=(M @ np.array([0.0, 0.0, 1.0])).tolist(),
                g_hat_cam=g.tolist(), w_hat_cam=w.tolist(), n_hat_cam=n.tolist(),
                cam_to_icrs=M.tolist(), cam_to_icrs_ortho_err=ortho_err,
                tilt_deg=float(getattr(mask, "tilt_deg", 0.0)),
                tilt_axis_deg=float(getattr(mask, "tilt_axis_deg", 0.0)))
