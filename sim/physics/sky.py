import math

def sky_layer(frame, cfg) -> float:
    """
    Derive sky background electrons per pixel per second from sky surface brightness.

    Uses:
      - cfg.sky_mu_mag_per_arcsec2 (mag/arcsec^2)
      - same zeropoint logic as stars (cfg.zeropoint_e_per_s override else derive from optics)
      - frame.plate_scale_rad_per_px

    Returns e-/px/s.
    """
    mu = float(getattr(cfg, "sky_mu_mag_per_arcsec2", 21.0))

    # Zeropoint: same as stars_layer
    if getattr(cfg, "zeropoint_e_per_s", 0.0) and cfg.zeropoint_e_per_s > 0.0:
        zp_e_s = float(cfg.zeropoint_e_per_s)
    else:
        from sim.physics.stars import derive_zeropoint_e_per_s  # or local import if colocated
        zp_e_s = derive_zeropoint_e_per_s(
            frame.lens,
            qe=float(getattr(frame.camera, "qe", 1.0)),
            lambda_eff_nm=float(getattr(cfg, "lambda_eff_nm", 550.0)),
            band_nm=float(getattr(cfg, "band_nm", 90.0)),
        )

    # pixel area in arcsec^2
    ps_arcsec = frame.plate_scale_rad_per_px * (180.0 / math.pi) * 3600.0
    omega_pix = ps_arcsec * ps_arcsec

    # sky rate: ZP * 10^(-0.4*mu) * omega_pix
    sky_e_px_s = zp_e_s * (10.0 ** (-0.4 * mu)) * omega_pix
    return float(sky_e_px_s)