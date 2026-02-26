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
        model = getattr(mask, "grating_model", "analytic")
        if model == "poppy":
            return _kernel_poppy_grating(frame, cfg, sigma_px, mask)
        return _kernel_grating_orders(frame, cfg, sigma_px, mask)

    if kind == "spider":
        model = getattr(mask, "spider_model", "manual")
        if model == "poppy":
            return _kernel_poppy_spider(frame, cfg, sigma_px, mask)
        return _kernel_spider_cross(frame, cfg, sigma_px, mask)

    if kind == "bitmap":
        from .psf import _gaussian_kernel
        return _gaussian_kernel(sigma_px)

    from .psf import _gaussian_kernel
    return _gaussian_kernel(sigma_px)

    
def _kernel_poppy_grating(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    """
    POPPY amplitude line grating at the entrance pupil.

    Implementation notes:
      - Builds a *band-limited* duty-cycle grating using a truncated Fourier series
        (reduces rotated-grid stair-step artifacts that can create spurious directions).
      - Keeps the user-facing parameters the same: lines_per_mm, duty_cycle, angle_deg.
      - Optional hidden knob: mask.grating_fourier_terms (default 7).
    """
    from .psf import _gaussian_kernel

    # -----------------------------
    # Basic geometry / knobs
    # -----------------------------
    lines_per_mm = float(getattr(mask, "lines_per_mm", 0.0))
    if not np.isfinite(lines_per_mm) or lines_per_mm <= 0.0:
        return _gaussian_kernel(sigma_px)

    duty_cycle = float(getattr(mask, "duty_cycle", 0.5))
    duty_cycle = float(np.clip(duty_cycle, 0.01, 0.99))

    pupil_samples = int(getattr(mask, "pupil_samples", 512))
    pupil_samples = max(32, pupil_samples)

    npix = int(getattr(mask, "psf_size_px", 513))
    npix = max(33, npix)

    # Lens → entrance pupil diameter
    f_mm = float(frame.lens.focal_mm)
    fnum = float(frame.lens.f_number)
    D_mm = f_mm / max(fnum, 1e-6)
    D = (D_mm * u.mm).to(u.m)

    # Grating pitch
    pitch_m = (1.0 / lines_per_mm) * 1e-3  # mm -> m

    # Rotation (dispersion axis in sensor coords)
    ang = np.deg2rad(float(getattr(mask, "angle_deg", 0.0)))

    # Fourier truncation (hidden knob)
    n_terms = int(getattr(mask, "grating_fourier_terms", 5))
    n_terms = int(np.clip(n_terms, 1, 51))

    # -----------------------------
    # Build pupil transmission array
    # -----------------------------
    # Dimensionless pupil grid in [-0.5, 0.5]
    grid = np.linspace(-0.5, 0.5, pupil_samples, dtype=np.float64)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    r = np.hypot(xx, yy)

    # Circular aperture in this normalized coord system
    aperture = (r <= 0.5).astype(np.float64)

    # Physical coordinate across pupil (meters)
    # xx,yy span [-0.5,0.5] so multiply by D [m] to get meters across diameter
    x_phys = xx * D.value
    y_phys = yy * D.value

    # Rotate grating coordinate
    x_rot = x_phys * np.cos(ang) + y_phys * np.sin(ang)

    # -----------------------------
    # Band-limited duty-cycle stripes via Fourier series
    # -----------------------------
    # uu is dimensionless coordinate in *periods*
    uu = x_rot / max(pitch_m, 1e-30)
    two_pi_uu = 2.0 * np.pi * uu

    # IMPORTANT: stripes must be a 2D array (avoid scalar init -> broadcast errors)
    stripes = np.full(x_rot.shape, duty_cycle, dtype=np.float64)

    Dd = duty_cycle
    # Real reconstruction:
    # stripes(u) = D + sum_{n=1..N} [ (2/(pi*n)) sin(pi*n*D) cos(2*pi*n*u - pi*n*D) ]
    for n in range(1, n_terms + 1):
        amp = (2.0 / (np.pi * n)) * np.sin(np.pi * n * Dd)
        if amp == 0.0:
            continue
        stripes = stripes + amp * np.cos(n * two_pi_uu - np.pi * n * Dd)

    # Clamp to [0,1] (truncation can overshoot near edges: Gibbs)
    stripes = np.clip(stripes, 0.0, 1.0)

    transmission = aperture * stripes

    # -----------------------------
    # POPPY ArrayOpticalElement pixelscale
    # -----------------------------
    # POPPY expects pupil-plane pixelscale with units of length/pixel (e.g. m/pix).
    pix_unit = getattr(u, "pixel", getattr(u, "pix", None))
    if pix_unit is None:
        # ultra-paranoid fallback; should not happen in normal astropy
        pix_unit = u.dimensionless_unscaled

    pupil_pixelscale = (D / float(pupil_samples)) / pix_unit  # -> m/pix

    grating_optic = poppy.ArrayOpticalElement(
        transmission=transmission,
        pixelscale=pupil_pixelscale,
    )
    # -----------------------------
    # Detector sampling
    # -----------------------------
    ps_arcsec = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0

    # -----------------------------
    # Bandpass sampling
    # -----------------------------
    lam_eff_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
    band_nm = float(getattr(cfg, "band_nm", 0.0))
    n_lambda = int(getattr(mask, "n_lambda", 9))
    n_lambda = max(1, n_lambda)

    if (not np.isfinite(band_nm)) or band_nm <= 0.0 or n_lambda <= 1:
        lam_list_nm = np.array([lam_eff_nm], dtype=np.float64)
    else:
        lam0 = lam_eff_nm - 0.5 * band_nm
        lam1 = lam_eff_nm + 0.5 * band_nm
        lam_list_nm = np.linspace(lam0, lam1, n_lambda, dtype=np.float64)

    # -----------------------------
    # Build optical system
    # -----------------------------
    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture(radius=0.5 * D))
    osys.add_pupil(grating_optic)
    osys.add_detector(pixelscale=ps_arcsec, fov_pixels=npix, oversample=1)

    # -----------------------------
    # Polychromatic PSF (weighted uniform for now)
    # -----------------------------
    k_accum = None
    w_accum = 0.0

    for lam_nm in lam_list_nm:
        psf_hdul = osys.calc_psf(wavelength=float(lam_nm) * u.nm)
        k = psf_hdul[0].data.astype(np.float64)
        
        # Center-crop if needed
        if k.shape != (npix, npix):
            ky, kx = k.shape
            if ky < npix or kx < npix:
                raise ValueError(f"POPPY PSF smaller than requested: got {k.shape}, want {(npix, npix)}")
            y0 = (ky - npix) // 2
            x0 = (kx - npix) // 2
            k = k[y0:y0 + npix, x0:x0 + npix]

        # Normalize each mono PSF before averaging
        s = float(np.sum(k))
        if s > 0.0:
            k /= s

        if k_accum is None:
            k_accum = np.zeros_like(k, dtype=np.float64)

        k_accum += k
        w_accum += 1.0

    k = k_accum / max(w_accum, 1.0)

    # Normalize again
    s = float(np.sum(k))
    if s > 0.0:
        k /= s

    # Optional Gaussian blur on top
    if sigma_px > 0:
        from .psf import _fft_convolve_same
        g = _gaussian_kernel(float(sigma_px), radius=(npix - 1) // 2)
        k = _fft_convolve_same(k, g)
        s = float(np.sum(k))
        if s > 0.0:
            k /= s

    return k.astype(np.float32, copy=False)


def _kernel_poppy_newtonian(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    # monochromatic version

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


def _kernel_poppy_spider(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    # polychromatic implementation

    # ---- telescope geometry ----
    D = float(getattr(mask, "aperture_diam_mm", 203.2)) * u.mm
    D = D.to(u.m)

    obsc_frac = float(getattr(mask, "obstruction_frac", 0.0))
    secondary_radius = 0.5 * obsc_frac * D  # meters

    vane_w = float(getattr(mask, "vane_width_mm", 0.76)) * u.mm
    n_vanes = int(getattr(mask, "n_vanes", 4))
    ang0 = float(getattr(mask, "angle_deg", 0.0))  # degrees

    # ---- sampling: arcsec/px in POPPY ----
    ps_arcsec = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0

    # ---- kernel size ----
    npix = int(getattr(mask, "psf_size_px", 513))  # odd is best

    # ---- bandpass ----
    lam_eff_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
    band_nm = float(getattr(cfg, "band_nm", 0.0))  # if 0 -> monochromatic
    n_lambda = int(getattr(mask, "n_lambda", 9))   # optional knob on mask; default 9

    if band_nm <= 0 or n_lambda <= 1:
        lam_list_nm = np.array([lam_eff_nm], dtype=float)
    else:
        lam0 = lam_eff_nm - 0.5 * band_nm
        lam1 = lam_eff_nm + 0.5 * band_nm
        # clamp to something sane for "visible-ish" just in case
        lam0 = max(lam0, 350.0)
        lam1 = min(lam1, 750.0)
        lam_list_nm = np.linspace(lam0, lam1, n_lambda, dtype=float)

    # ---- build optical system once ----
    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture(radius=0.5 * D))

    # Central obscuration + symmetric supports (supports work even if secondary_radius=0)
    osys.add_pupil(
        poppy.SecondaryObscuration(
            secondary_radius=secondary_radius,
            n_supports=n_vanes,
            support_width=vane_w,
            support_angle_offset=ang0
        )
    )

    osys.add_detector(pixelscale=ps_arcsec, fov_pixels=npix)

    # ---- polychromatic average ----
    k_accum = None
    w_accum = 0.0

    for lam_nm in lam_list_nm:
        lam = float(lam_nm) * u.nm
        psf_hdul = osys.calc_psf(wavelength=lam)
        k = psf_hdul[0].data.astype(np.float64)

        # If POPPY returned a different size than requested, center-crop to npix
        if k.shape != (npix, npix):
            ky, kx = k.shape
            if ky < npix or kx < npix:
                raise ValueError(f"POPPY PSF smaller than requested: got {k.shape}, want {(npix,npix)}")
            y0 = (ky - npix) // 2
            x0 = (kx - npix) // 2
            k = k[y0:y0+npix, x0:x0+npix]

        # normalize each mono PSF to unit energy before averaging
        s = float(np.sum(k))
        if s > 0:
            k /= s

        if k_accum is None:
            k_accum = np.zeros_like(k, dtype=np.float64)

        k_accum += k
        w_accum += 1.0

    k = k_accum / max(w_accum, 1.0)

    # normalize energy again
    s = float(np.sum(k))
    if s > 0:
        k /= s

    # ---- seeing/focus blur on top (same as before) ----
    if sigma_px > 0:
        from .psf import _gaussian_kernel, _fft_convolve_same
        g = _gaussian_kernel(float(sigma_px), radius=(npix - 1) // 2)
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

# def _kernel_grating_orders(frame, cfg, sigma_px: float, mask) -> np.ndarray:
#     """
#     Analytic grating model with real wavelength sampling smear and conservative
#     energy distribution into orders, with a physically sensible spot profile.

#     Key changes vs "pinprick" version:
#       - Use an effective PSF width set by quadrature:
#             sigma_eff^2 = sigma_seeing^2 + sigma_diff^2 + sigma_pix^2
#       - Deposit energy using a Moffat profile (has wings), so bright stars look
#         larger under stretch (like POPPY), without flux-dependent PSF.

#     Placement:
#       - Uses grating small-angle equation for each wavelength sample:
#             theta = m * lambda / pitch
#             sep_px = f * theta / pix
#       - This should match POPPY if your POPPY detector pixelscale equals your
#         frame plate scale, and pitch units are consistent.

#     Energy conservation:
#       - Compute relative order weights for m=0..order_norm_max, normalize.
#       - Put truncated remainder (m > order_max) back into m=0 so ±1 is NOT
#         artificially boosted when order_max is small.

#     Mask fields used:
#       - lines_per_mm, order_max, duty_cycle, angle_deg
#       - n_lambda (smear sampling), optional order_norm_max
#       - optional: moffat_beta, order_decay, order_floor

#     cfg fields used:
#       - lambda_eff_nm, band_nm
#       - seeing is passed in as sigma_px already (from render.py)

#     """
#     import numpy as np
#     from .psf import _gaussian_kernel

#     # -----------------------------
#     # Basic params
#     # -----------------------------
#     lines_per_mm = float(getattr(mask, "lines_per_mm", 0.0))
#     if lines_per_mm <= 0.0:
#         return _gaussian_kernel(float(sigma_px))

#     order_max = int(getattr(mask, "order_max", 1))
#     order_max = max(order_max, 0)
#     if order_max <= 0:
#         return _gaussian_kernel(float(sigma_px))

#     duty_cycle = float(getattr(mask, "duty_cycle", 0.5))
#     duty_cycle = float(np.clip(duty_cycle, 0.01, 0.99))

#     # -----------------------------
#     # Wavelength sampling (top-hat smear)
#     # -----------------------------
#     lam_eff_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
#     band_nm = float(getattr(cfg, "band_nm", 0.0))
#     n_lambda = int(getattr(mask, "n_lambda", 9))
#     n_lambda = max(n_lambda, 1)

#     if band_nm <= 0.0 or n_lambda <= 1:
#         lam_list_nm = np.array([lam_eff_nm], dtype=float)
#     else:
#         lam0 = lam_eff_nm - 0.5 * band_nm
#         lam1 = lam_eff_nm + 0.5 * band_nm
#         lam_list_nm = np.linspace(lam0, lam1, n_lambda, dtype=float)

#     # -----------------------------
#     # Geometry
#     # -----------------------------
#     pitch_mm = 1.0 / lines_per_mm
#     f_mm = float(frame.lens.focal_mm)
#     pix_mm = float(frame.camera.pixel_um) * 1e-3  # um -> mm

#     ang = np.deg2rad(float(getattr(mask, "angle_deg", 0.0)))
#     ux, uy = float(np.cos(ang)), float(np.sin(ang))

#     # -----------------------------
#     # Effective sigma: seeing + diffraction + pixel integration (quadrature)
#     # -----------------------------
#     # plate scale in arcsec/px
#     plate_scale_arcsec_per_px = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0

#     # Entrance pupil diameter D = f / f#
#     fnum = float(frame.lens.f_number)
#     D_m = (f_mm / max(fnum, 1e-12)) * 1e-3  # mm -> m

#     # Diffraction-limited Airy FWHM approx: ~1.03 * lambda/D (radians)
#     lam_m = lam_eff_nm * 1e-9
#     fwhm_diff_arcsec = (1.03 * lam_m / max(D_m, 1e-20)) * 206265.0
#     sigma_diff_px = (fwhm_diff_arcsec / max(plate_scale_arcsec_per_px, 1e-20)) / 2.355

#     # Pixel integration: square pixel width 1 px -> sigma ~ 1/sqrt(12)
#     sigma_pix_px = 1.0 / np.sqrt(12.0)

#     sigma_seeing_px = float(max(sigma_px, 0.0))
#     sigma_eff = float(np.sqrt(sigma_seeing_px**2 + sigma_diff_px**2 + sigma_pix_px**2))

#     # Optional sanity floor (rarely needed; keep default 0)
#     sigma_floor_px = float(getattr(mask, "sigma_floor_px", 0.0))
#     if sigma_floor_px > 0.0:
#         sigma_eff = max(sigma_eff, sigma_floor_px)

#     # -----------------------------
#     # Order energy model (rect 0/1 amplitude grating)
#     # -----------------------------
#     def _rect_rel_I(m: int, D: float) -> float:
#         # amplitude pulse train of width D in [0,1] => intensity coefficients
#         if m == 0:
#             return D * D
#         x = np.pi * m * D
#         return (np.sin(x) / (np.pi * m)) ** 2

#     order_decay = float(getattr(mask, "order_decay", 0.0))   # optional tweak
#     order_floor = float(getattr(mask, "order_floor", 0.0))   # optional tweak

#     order_norm_max = int(getattr(mask, "order_norm_max", max(50, order_max)))
#     order_norm_max = max(order_norm_max, order_max)

#     w_all = np.zeros(order_norm_max + 1, dtype=np.float64)
#     for m in range(0, order_norm_max + 1):
#         w = _rect_rel_I(m, duty_cycle)
#         if m > 0 and order_decay > 0.0:
#             w *= np.exp(-m / max(order_decay, 1e-12))
#         if m > 0 and order_floor > 0.0:
#             w += order_floor
#         w_all[m] = max(w, 0.0)

#     s_all = float(w_all.sum())
#     if s_all <= 0.0:
#         return _gaussian_kernel(sigma_eff)
#     w_all /= s_all

#     # Put truncated remainder (m>order_max) into m=0 to avoid boosting ±1
#     w_m = np.zeros(order_max + 1, dtype=np.float64)
#     w_m[0] = float(w_all[0] + w_all[order_max + 1:].sum())
#     w_m[1:] = w_all[1:order_max + 1]

#     # -----------------------------
#     # Kernel size: based on max separation at max lambda
#     # -----------------------------
#     lam_max_mm = float(np.max(lam_list_nm)) * 1e-6  # nm -> mm
#     max_sep_px = 0.0
#     for m in range(1, order_max + 1):
#         theta = m * (lam_max_mm / pitch_mm)
#         sep_px = (f_mm * theta) / pix_mm
#         max_sep_px = max(max_sep_px, sep_px)

#     margin = max(6.0, 6.0 * sigma_eff)  # moffat wings need a bit more room than gaussian
#     half = int(np.ceil(max_sep_px + margin))
#     half = max(half, 8)
#     size = 2 * half + 1
#     cx = cy = float(half)

#     k = np.zeros((size, size), dtype=np.float64)

#     # -----------------------------
#     # Deposition: Moffat spot (normalized on patch)
#     # -----------------------------
#     beta = float(getattr(mask, "moffat_beta", 3.5))  # 3–5 is typical; higher => more gaussian-like
#     beta = float(np.clip(beta, 1.5, 20.0))

#     # Convert sigma_eff to an equivalent FWHM, then to Moffat alpha
#     fwhm_eff = 2.355 * sigma_eff
#     # FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
#     denom = 2.0 * np.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
#     alpha = max(fwhm_eff / max(denom, 1e-12), 1e-6)

#     # Choose a patch radius large enough to show wings even when core is small
#     R = int(np.ceil(max(8.0, 8.0 * fwhm_eff)))  # in pixels
#     R = min(R, half)  # don't exceed kernel bounds

#     def _add_moffat(img, x0, y0, w):
#         x0f = float(x0)
#         y0f = float(y0)

#         x_c = int(np.floor(x0f))
#         y_c = int(np.floor(y0f))
#         x_min = max(0, x_c - R)
#         x_max = min(img.shape[1], x_c + R + 1)
#         y_min = max(0, y_c - R)
#         y_max = min(img.shape[0], y_c + R + 1)
#         if x_min >= x_max or y_min >= y_max:
#             return

#         yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float64)
#         rr2 = (xx - x0f) ** 2 + (yy - y0f) ** 2
#         prof = (1.0 + rr2 / (alpha ** 2)) ** (-beta)

#         s = float(prof.sum())
#         if s > 0.0:
#             img[y_min:y_max, x_min:x_max] += (float(w) / s) * prof

#     # -----------------------------
#     # Accumulate kernel
#     # -----------------------------
#     # 0th order at center
#     if w_m[0] > 0.0:
#         _add_moffat(k, cx, cy, float(w_m[0]))

#     # smear weights
#     w_lam = 1.0 / float(len(lam_list_nm))

#     for m in range(1, order_max + 1):
#         wm = float(w_m[m])
#         if wm <= 0.0:
#             continue
#         w_side = 0.5 * wm

#         for lam_nm in lam_list_nm:
#             lam_mm = float(lam_nm) * 1e-6
#             theta = m * (lam_mm / pitch_mm)
#             sep_px = (f_mm * theta) / pix_mm

#             dx = sep_px * ux
#             dy = sep_px * uy
#             w_sample = w_side * w_lam

#             _add_moffat(k, cx + dx, cy + dy, w_sample)
#             _add_moffat(k, cx - dx, cy - dy, w_sample)

#     # normalize
#     s = float(k.sum())
#     if s > 0.0:
#         k /= s

#     return k.astype(np.float32, copy=False)


def _kernel_grating_orders(frame, cfg, sigma_px: float, mask) -> np.ndarray:
    """
    Analytic grating model with real wavelength sampling smear and conservative
    energy distribution into orders, with a physically sensible spot profile.

    Key changes vs "pinprick" version:
      - Use an effective PSF width set by quadrature:
            sigma_eff^2 = sigma_seeing^2 + sigma_diff^2 + sigma_pix^2
      - Deposit energy using a Moffat profile (has wings), so bright stars look
        larger under stretch (like POPPY), without flux-dependent PSF.

    Placement:
      - Uses grating small-angle equation for each wavelength sample:
            theta = m * lambda / pitch
            sep_px = f * theta / pix

    Energy conservation:
      - Compute relative order weights for m=0..order_norm_max, normalize.
      - Put truncated remainder (m > order_max) back into m=0 so ±1 is NOT
        artificially boosted when order_max is small.

    NEW (only change vs prior "worked well" version):
      - Replace n_lambda discrete dot-smear with a *continuous segment smear*
        between band endpoints. We still keep the old n_lambda logic as a
        fallback when band_nm<=0 or smear disabled.
      - Segment deposition uses a raised-cosine (hill-shaped) longitudinal weight:
            w(u) ∝ sin^2(pi u), u∈[0,1]
        which is 0 at endpoints and peaks in the middle (a simple "visible-like"
        taper without needing QE×SED tables yet).

    Mask fields used:
      - lines_per_mm, order_max, duty_cycle, angle_deg
      - n_lambda (legacy), optional order_norm_max
      - optional: moffat_beta, order_decay, order_floor
      - optional (new): smear_step_px (default 0.75), smear_cap (default 250),
                        smear_profile ("raised_cosine" default, "flat" allowed)

    cfg fields used:
      - lambda_eff_nm, band_nm
      - seeing is passed in as sigma_px already (from render.py)

    """
    import numpy as np
    from .psf import _gaussian_kernel

    # -----------------------------
    # Basic params
    # -----------------------------
    lines_per_mm = float(getattr(mask, "lines_per_mm", 0.0))
    if lines_per_mm <= 0.0:
        return _gaussian_kernel(float(sigma_px))

    order_max = int(getattr(mask, "order_max", 1))
    order_max = max(order_max, 0)
    if order_max <= 0:
        return _gaussian_kernel(float(sigma_px))

    duty_cycle = float(getattr(mask, "duty_cycle", 0.5))
    duty_cycle = float(np.clip(duty_cycle, 0.01, 0.99))

    # -----------------------------
    # Wavelength endpoints (and legacy sample list)
    # -----------------------------
    lam_eff_nm = float(getattr(cfg, "lambda_eff_nm", 550.0))
    band_nm = float(getattr(cfg, "band_nm", 0.0))
    n_lambda = int(getattr(mask, "n_lambda", 9))
    n_lambda = max(n_lambda, 1)

    if band_nm <= 0.0:
        lam0_nm = lam1_nm = lam_eff_nm
        lam_list_nm = np.array([lam_eff_nm], dtype=float)
    else:
        lam0_nm = lam_eff_nm - 0.5 * band_nm
        lam1_nm = lam_eff_nm + 0.5 * band_nm
        # legacy discrete list (kept for compatibility / fallback)
        lam_list_nm = np.linspace(lam0_nm, lam1_nm, n_lambda, dtype=float)

    # -----------------------------
    # Geometry
    # -----------------------------
    pitch_mm = 1.0 / lines_per_mm
    f_mm = float(frame.lens.focal_mm)
    pix_mm = float(frame.camera.pixel_um) * 1e-3  # um -> mm

    ang = np.deg2rad(float(getattr(mask, "angle_deg", 0.0)))
    ux, uy = float(np.cos(ang)), float(np.sin(ang))

    # -----------------------------
    # Effective sigma: seeing + diffraction + pixel integration (quadrature)
    # -----------------------------
    # plate scale in arcsec/px
    plate_scale_arcsec_per_px = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0

    # Entrance pupil diameter D = f / f#
    fnum = float(frame.lens.f_number)
    D_m = (f_mm / max(fnum, 1e-12)) * 1e-3  # mm -> m

    # Diffraction-limited Airy FWHM approx: ~1.03 * lambda/D (radians)
    lam_m = lam_eff_nm * 1e-9
    fwhm_diff_arcsec = (1.03 * lam_m / max(D_m, 1e-20)) * 206265.0
    sigma_diff_px = (fwhm_diff_arcsec / max(plate_scale_arcsec_per_px, 1e-20)) / 2.355

    # Pixel integration: square pixel width 1 px -> sigma ~ 1/sqrt(12)
    sigma_pix_px = 1.0 / np.sqrt(12.0)

    sigma_seeing_px = float(max(sigma_px, 0.0))
    sigma_eff = float(np.sqrt(sigma_seeing_px**2 + sigma_diff_px**2 + sigma_pix_px**2))

    # Optional sanity floor (rarely needed; keep default 0)
    sigma_floor_px = float(getattr(mask, "sigma_floor_px", 0.0))
    if sigma_floor_px > 0.0:
        sigma_eff = max(sigma_eff, sigma_floor_px)

    # -----------------------------
    # Order energy model (rect 0/1 amplitude grating)
    # -----------------------------
    def _rect_rel_I(m: int, D: float) -> float:
        # amplitude pulse train of width D in [0,1] => intensity coefficients
        if m == 0:
            return D * D
        x = np.pi * m * D
        return (np.sin(x) / (np.pi * m)) ** 2

    order_decay = float(getattr(mask, "order_decay", 0.0))   # optional tweak
    order_floor = float(getattr(mask, "order_floor", 0.0))   # optional tweak

    order_norm_max = int(getattr(mask, "order_norm_max", max(50, order_max)))
    order_norm_max = max(order_norm_max, order_max)

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
        return _gaussian_kernel(sigma_eff)
    w_all /= s_all

    # Put truncated remainder (m>order_max) into m=0 to avoid boosting ±1
    w_m = np.zeros(order_max + 1, dtype=np.float64)
    w_m[0] = float(w_all[0] + w_all[order_max + 1:].sum())
    w_m[1:] = w_all[1:order_max + 1]

    # -----------------------------
    # Kernel size: based on max separation at max lambda
    # -----------------------------
    lam_max_nm = float(np.max(lam_list_nm))
    lam_max_mm = lam_max_nm * 1e-6  # nm -> mm

    max_sep_px = 0.0
    for m in range(1, order_max + 1):
        theta = m * (lam_max_mm / pitch_mm)
        sep_px = (f_mm * theta) / pix_mm
        max_sep_px = max(max_sep_px, sep_px)

    margin = max(6.0, 6.0 * sigma_eff)  # moffat wings need a bit more room than gaussian
    half = int(np.ceil(max_sep_px + margin))
    half = max(half, 8)
    size = 2 * half + 1
    cx = cy = float(half)

    k = np.zeros((size, size), dtype=np.float64)

    # -----------------------------
    # Deposition: Moffat spot (normalized on patch)
    # -----------------------------
    beta = float(getattr(mask, "moffat_beta", 3.5))  # 3–5 is typical; higher => more gaussian-like
    beta = float(np.clip(beta, 1.5, 20.0))

    # Convert sigma_eff to an equivalent FWHM, then to Moffat alpha
    fwhm_eff = 2.355 * sigma_eff
    # FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
    denom = 2.0 * np.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
    alpha = max(fwhm_eff / max(denom, 1e-12), 1e-6)

    # Choose a patch radius large enough to show wings even when core is small
    R = int(np.ceil(max(8.0, 8.0 * fwhm_eff)))  # in pixels
    R = min(R, half)  # don't exceed kernel bounds

    def _add_moffat(img, x0, y0, w):
        x0f = float(x0)
        y0f = float(y0)

        x_c = int(np.floor(x0f))
        y_c = int(np.floor(y0f))
        x_min = max(0, x_c - R)
        x_max = min(img.shape[1], x_c + R + 1)
        y_min = max(0, y_c - R)
        y_max = min(img.shape[0], y_c + R + 1)
        if x_min >= x_max or y_min >= y_max:
            return

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float64)
        rr2 = (xx - x0f) ** 2 + (yy - y0f) ** 2
        prof = (1.0 + rr2 / (alpha ** 2)) ** (-beta)

        s = float(prof.sum())
        if s > 0.0:
            img[y_min:y_max, x_min:x_max] += (float(w) / s) * prof

    # -----------------------------
    # Accumulate kernel
    # -----------------------------
    # 0th order at center
    if w_m[0] > 0.0:
        _add_moffat(k, cx, cy, float(w_m[0]))

    # -----------------------------
    # NEW smear mechanism: segment fill between band endpoints
    # (only touches smear; preserves previous structure/weights)
    # -----------------------------
    use_segment_smear = bool(getattr(mask, "segment_smear", True))
    smear_profile = str(getattr(mask, "smear_profile", "raised_cosine")).lower()
    smear_step_px = float(getattr(mask, "smear_step_px", 0.75))  # user target spacing
    # Ensure deposits overlap the core: step <= ~0.5*FWHM is a good rule of thumb
    smear_step_px = min(smear_step_px, max(0.15, 0.5 * fwhm_eff))
    smear_step_px = float(np.clip(smear_step_px, 0.10, 5.0))
    smear_cap = int(getattr(mask, "smear_cap", 250))            # max deposits per side per order
    smear_cap = int(np.clip(smear_cap, 8, 2000))

    # helper: hill-shaped weights (0 at ends, peak middle)
    def _longitudinal_weights(u: np.ndarray) -> np.ndarray:
        if smear_profile == "flat":
            w = np.ones_like(u, dtype=np.float64)
        else:
            # raised-cosine "hill": sin^2(pi u), u in [0,1]
            w = np.sin(np.pi * u) ** 2
        s = float(w.sum())
        if s <= 0.0:
            w[:] = 1.0
            s = float(w.sum())
        return w / s

    # If no band, or segment smear disabled, fall back to legacy discrete sampling.
    if (band_nm <= 0.0) or (not use_segment_smear) or (n_lambda <= 1):
        # legacy: equal weights over lam_list_nm
        w_lam = 1.0 / float(len(lam_list_nm))
        for m in range(1, order_max + 1):
            wm = float(w_m[m])
            if wm <= 0.0:
                continue
            w_side = 0.5 * wm
            for lam_nm in lam_list_nm:
                lam_mm = float(lam_nm) * 1e-6
                theta = m * (lam_mm / pitch_mm)
                sep_px = (f_mm * theta) / pix_mm
                dx = sep_px * ux
                dy = sep_px * uy
                w_sample = w_side * w_lam
                _add_moffat(k, cx + dx, cy + dy, w_sample)
                _add_moffat(k, cx - dx, cy - dy, w_sample)
    else:
        # segment: endpoints in sep_px (sep ∝ lambda, so linear interpolation in u is exact here)
        lam0_mm = float(lam0_nm) * 1e-6
        lam1_mm = float(lam1_nm) * 1e-6

        for m in range(1, order_max + 1):
            wm = float(w_m[m])
            if wm <= 0.0:
                continue
            w_side = 0.5 * wm

            theta0 = m * (lam0_mm / pitch_mm)
            theta1 = m * (lam1_mm / pitch_mm)
            sep0_px = (f_mm * theta0) / pix_mm
            sep1_px = (f_mm * theta1) / pix_mm

            smear_len_px = float(abs(sep1_px - sep0_px))
            if smear_len_px <= 1e-9:
                u = np.array([0.5], dtype=np.float64)
            else:
                n_seg = int(np.ceil(smear_len_px / smear_step_px)) + 1
                n_seg = int(np.clip(n_seg, 8, smear_cap))
                u = np.linspace(0.0, 1.0, n_seg, dtype=np.float64)

            w_u = _longitudinal_weights(u)

            # deposit along segment, symmetric ± order
            for uu, wu in zip(u, w_u):
                sep_px = sep0_px + uu * (sep1_px - sep0_px)
                dx = sep_px * ux
                dy = sep_px * uy
                w_sample = w_side * float(wu)
                _add_moffat(k, cx + dx, cy + dy, w_sample)
                _add_moffat(k, cx - dx, cy - dy, w_sample)

    # normalize
    s = float(k.sum())
    if s > 0.0:
        k /= s

    return k.astype(np.float32, copy=False)


# def _kernel_grating_orders(frame, cfg, sigma_px: float, mask) -> np.ndarray:
#     """
#     v1 grating model:
#       PSF kernel = central Gaussian + shifted Gaussians at ±m orders.
#     """
#     from .psf import _gaussian_kernel  # avoid duplication

#     offsets = _diffraction_order_offsets_px(frame, cfg, mask)
#     if len(offsets) == 0:
#         return _gaussian_kernel(sigma_px)

#     # relative amplitude per order (each m, each side)
#     order_rel_amp = float(getattr(mask, "order_rel_amp", 0.12))
#     order_rel_amp = max(order_rel_amp, 0.0)

#     # choose kernel support: cover furthest order + margin
#     max_sep = max(np.hypot(dx, dy) for (_, dx, dy) in offsets)
#     half = int(np.ceil(max_sep + 4.0 * sigma_px))
#     half = max(half, int(np.ceil(4.0 * sigma_px)))
#     size = 2 * half + 1

#     yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
#     cx = cy = float(half)

#     # central Gaussian (amplitude 1)
#     k = _gaussian_2d(xx, yy, cx, cy, sigma_px)

#     # add ±m orders
#     for (m, dx, dy) in offsets:
#         k += order_rel_amp * _gaussian_2d(xx, yy, cx + dx, cy + dy, sigma_px)
#         k += order_rel_amp * _gaussian_2d(xx, yy, cx - dx, cy - dy, sigma_px)

#     # normalize energy
#     s = float(k.sum())
#     if s > 0:
#         k /= s

#     return k
