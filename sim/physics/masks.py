# sim/physics/masks.py
import numpy as np
import astropy.units as u
import poppy


def kernel_for_mask(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    kind = getattr(mask, "kind", "none")

    if kind == "none":
        from .psf import _gaussian_kernel
        return _gaussian_kernel(sigma_px)

    if kind == "grating":
        return _kernel_grating_orders(frame, cfg, sigma_px, mask)

    if kind == "spider":
        return _kernel_spider_cross(frame, cfg, sigma_px, mask)

    if kind == "bitmap":
        from .psf import _gaussian_kernel
        return _gaussian_kernel(sigma_px)

    if kind in ("pupil", "poppy"):
        return _kernel_poppy_newtonian(frame, cfg, sigma_px, mask)

    from .psf import _gaussian_kernel
    return _gaussian_kernel(sigma_px)


def _kernel_poppy_newtonian(frame, cfg, sigma_px: float, mask) -> np.ndarray:

    # ---- telescope geometry ----
    D = float(getattr(mask, "aperture_diam_mm", 203.2)) * u.mm
    D = D.to(u.m) 
    obsc_frac = float(getattr(mask, "obstruction_frac", 0.30))
    secondary_radius = 0.5 * obsc_frac * D

    vane_w = float(getattr(mask, "vane_width_mm", 0.76)) * u.mm
    n_vanes = int(getattr(mask, "n_vanes", 4))
    ang0 = float(getattr(mask, "angle_deg", 0.0))  # degrees

    # ---- wavelength ----
    lam = float(getattr(cfg, "lambda_eff_nm", 550.0)) * u.nm

    # ---- image sampling: arcsec/px in POPPY ----
    # you have plate_scale_rad_per_px already
    ps_arcsec = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0

    # ---- kernel size ----
    npix = int(getattr(mask, "psf_size_px", 513))  # odd is best
    # POPPY wants field of view or pixelscale; we’ll set pixelscale and npix.
    # (Oversampling can be added later.)

    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture(radius=0.5 * D))

    # Secondary obscuration + symmetric supports
    osys.add_pupil(
        poppy.SecondaryObscuration(
            secondary_radius=secondary_radius,
            n_supports=n_vanes,
            support_width=vane_w,
            support_angle_offset=ang0
        )
    )

    osys.add_detector(pixelscale=ps_arcsec, fov_pixels=npix)

    psf_hdul = osys.calc_psf(wavelength=lam)
    k = psf_hdul[0].data.astype(np.float64)

    # normalize energy
    s = float(np.sum(k))
    if s > 0:
        k /= s

    if sigma_px > 0:
        from .psf import _gaussian_kernel
        from .psf import _fft_convolve_same
        g = _gaussian_kernel(sigma_px, radius=(npix - 1) // 2)
        k = _fft_convolve_same(k, g)
        s = float(np.sum(k))
        if s > 0:
            k /= s

    return k.astype(np.float32, copy=False)


def _kernel_spider_cross(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    """
    Spider model (cross): Gaussian core + thin diffraction spikes along vane axes,
    with smooth taper to avoid hard cutoff.
    """
    from .psf import _gaussian_kernel

    n_vanes = int(getattr(mask, "n_vanes", 0))
    vane_w_mm = float(getattr(mask, "vane_width_mm", 0.0))
    if n_vanes <= 0 or vane_w_mm <= 0.0:
        return _gaussian_kernel(sigma_px)

    # Lens geometry
    f_mm = float(frame.lens.focal_mm)
    pix_mm = float(frame.camera.pixel_um) * 1e-3
    lam_mm = float(getattr(cfg, "lambda_eff_nm", 550.0)) * 1e-6

    # --- spike thickness (perpendicular sigma) ---
    perp_scale_mm = f_mm * (lam_mm / max(vane_w_mm, 1e-9))
    spike_sigma_perp_px = max(0.5, perp_scale_mm / max(pix_mm, 1e-12))

    # --- spike extent (kernel half-size) ---
    spike_radius_px = int(getattr(mask, "spike_radius_px", 250))
    spike_radius_px = max(spike_radius_px, int(np.ceil(4.0 * sigma_px)))

    # --- relative weight of spikes vs core ---
    spike_rel_amp = float(getattr(mask, "spike_rel_amp", 0.10))
    spike_rel_amp = max(spike_rel_amp, 0.0)

    # Build grid
    half = spike_radius_px
    size = 2 * half + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = cy = float(half)

    # Core
    k = _gaussian_kernel(sigma_px, radius=half).astype(np.float64, copy=False)

    # --- Vane angles (THIS WAS MISSING) ---
    # n_vanes counts unique *lines* through center, spaced over 180 deg
    base_ang = float(getattr(mask, "angle_deg", 0.0))
    angles = [np.deg2rad(base_ang + k_i * (180.0 / n_vanes)) for k_i in range(n_vanes)]

    # --- Ridge components (v2): across Gaussian * along power-law * smooth taper ---
    spike_falloff = float(getattr(mask, "spike_falloff", 1.6))  # 1.2–2.5
    spike_core_px = float(getattr(mask, "spike_core_px", 8.0))  # inner scale
    taper_frac    = float(getattr(mask, "taper_frac", 0.15))    # 0.05–0.30

    R = float(spike_radius_px)
    taper_start = (1.0 - taper_frac) * R

    ridge = np.zeros_like(k)

    dx = (xx - cx)
    dy = (yy - cy)

    for ang in angles:
        ux, uy = np.cos(ang), np.sin(ang)

        u_par  = dx * ux + dy * uy
        u_perp = -dx * uy + dy * ux

        across = np.exp(-0.5 * (u_perp / spike_sigma_perp_px) ** 2)

        t = np.abs(u_par)
        along = 1.0 / (1.0 + (t / max(spike_core_px, 1e-6)) ** spike_falloff)
        # suppress ridge at the very center so it blends smoothly into the core
        blend_r0 = float(getattr(mask, "blend_r0_px", 2.0))   # 1–4 px
        blend_pow = float(getattr(mask, "blend_pow", 2.0))    # 2–6
        r = np.hypot(dx, dy)
        center_blend = 1.0 - np.exp(-(r / max(blend_r0, 1e-6)) ** blend_pow)

        taper = np.ones_like(along)
        m = t > taper_start
        if np.any(m):
            z = (t[m] - taper_start) / max(R - taper_start, 1e-6)  # 0..1
            taper[m] = 0.5 * (1.0 + np.cos(np.pi * z))             # 1->0

        ridge += across * along * taper * center_blend

    # Normalize ridge so its peak doesn't dominate; then add with weight
    ridge_max = float(np.max(ridge)) if np.isfinite(ridge).any() else 0.0
    if ridge_max > 0.0:
        ridge /= ridge_max

    k = k + spike_rel_amp * ridge

    # Renormalize energy
    s = float(np.sum(k))
    if s > 0.0:
        k /= s

    return k

def _diffraction_order_offsets_px(frame, cfg, mask):
    """
    Compute diffraction order offsets (dx, dy) in pixels for m=1..order_max.

    Uses small-angle grating equation: theta ~ m * lambda / pitch
    pitch is derived from lines_per_mm.
    """
    lines_per_mm = float(getattr(mask, "lines_per_mm", 0.0))
    if lines_per_mm <= 0.0:
        return []

    # pitch in mm
    pitch_mm = 1.0 / lines_per_mm

    # wavelength in mm
    lam_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
    lam_mm = lam_nm * 1e-6  # nm -> mm

    # focal length in mm
    f_mm = float(frame.lens.focal_mm)

    # pixel size in mm
    pix_um = float(frame.camera.pixel_um)
    pix_mm = pix_um * 1e-3

    # diffraction axis angle in sensor coords
    ang = np.deg2rad(float(getattr(mask, "angle_deg", 0.0)))
    ux, uy = float(np.cos(ang)), float(np.sin(ang))

    order_max = int(getattr(mask, "order_max", 1))
    order_max = max(order_max, 0)

    offsets = []
    for m in range(1, order_max + 1):
        theta = m * (lam_mm / pitch_mm)  # radians (small angle)
        sep_mm = f_mm * theta            # mm at sensor
        sep_px = sep_mm / pix_mm         # pixels
        offsets.append((m, sep_px * ux, sep_px * uy))

    return offsets


def _gaussian_2d(xx, yy, x0, y0, sigma_px: float):
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return np.exp(-0.5 * rr2 / (sigma_px ** 2))


def _kernel_grating_orders(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    """
    v1 grating model:
      PSF kernel = central Gaussian + shifted Gaussians at ±m orders.
    """
    from .psf import _gaussian_kernel  # avoid duplication

    offsets = _diffraction_order_offsets_px(frame, cfg, mask)
    if len(offsets) == 0:
        return _gaussian_kernel(sigma_px)

    # relative amplitude per order (each m, each side)
    order_rel_amp = float(getattr(mask, "order_rel_amp", 0.12))
    order_rel_amp = max(order_rel_amp, 0.0)

    # choose kernel support: cover furthest order + margin
    max_sep = max(np.hypot(dx, dy) for (_, dx, dy) in offsets)
    half = int(np.ceil(max_sep + 4.0 * sigma_px))
    half = max(half, int(np.ceil(4.0 * sigma_px)))
    size = 2 * half + 1

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = cy = float(half)

    # central Gaussian (amplitude 1)
    k = _gaussian_2d(xx, yy, cx, cy, sigma_px)

    # add ±m orders
    for (m, dx, dy) in offsets:
        k += order_rel_amp * _gaussian_2d(xx, yy, cx + dx, cy + dy, sigma_px)
        k += order_rel_amp * _gaussian_2d(xx, yy, cx - dx, cy - dy, sigma_px)

    # normalize energy
    s = float(k.sum())
    if s > 0:
        k /= s

    return k

