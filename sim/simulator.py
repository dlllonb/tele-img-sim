# sim/run_case.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np


def run_sim_and_report(
    *,
    # -----------------------
    # I/O
    # -----------------------
    out_dir: str = "out",
    run_name: str = "run",
    stars_csv: str = "sim/physics/starfields/field2.csv",
    overwrite: bool = True,

    # -----------------------
    # Pointing
    # -----------------------
    ra0_deg: float = 30.0,
    dec0_deg: float = 70.0,
    rot_deg: float = 0.0,

    # -----------------------
    # Camera (ASI178/187-ish default)
    # -----------------------
    cam_nx: int = 3096,
    cam_ny: int = 2080,
    cam_pixel_um: float = 2.4,
    cam_read_noise_e: float = 2.0,
    cam_gain_e_per_adu: float = 1.0,
    cam_qe: float = 0.6,

    # -----------------------
    # Lens (wide/fast payload lens default)
    # -----------------------
    lens_focal_mm: float = 120.0,
    lens_f_number: float = 1.8,
    lens_transmission: float = 0.9,

    mask_kind: str = "none",
    
    # -----------------------
    # Mask (POPPY spider)
    # -----------------------
    mask_spider_model: str = "manual",      # "manual" | "poppy"
    mask_angle_deg: float = 12.0,
    mask_n_vanes: int = 2,
    mask_vane_width_mm: float = 0.6,
    mask_obstruction_frac: float = 0.0,
    mask_psf_size_px: int = 300,
    mask_aperture_diam_mm: Optional[float] = None,  # if None, uses f/f#

    # -----------------------
    # Mask (Grating knobs)
    # -----------------------
    mask_grating_model: str = "analytic",   # "analytic" | "poppy"
    mask_lines_per_mm: float = 0.0,
    mask_order_max: int = 1,
    mask_order_rel_amp: float = 0.12,

    mask_duty_cycle: float = 0.5,
    mask_pupil_samples: int = 512,
    mask_n_lambda: int = 9,

    # -----------------------
    # Render config
    # -----------------------
    exposure_s: float = 10.0,
    sky_mu_mag_per_arcsec2: float = 21.0,
    zeropoint_e_per_s: float = 0.0,
    lambda_eff_nm: float = 550.0,
    band_nm: float = 90.0,
    seeing_fwhm_arcsec: float = 1.5,
    jitter_pointing_rms: float = 16.0,

    enable_sky: bool = True,
    enable_stars: bool = True,
    enable_psf: bool = True,
    enable_jitter: bool = True,
    enable_noise: bool = True,

    seed: int = 42,

    # -----------------------
    # Plot controls
    # -----------------------
    show_plots: bool = True,
    save_pngs: bool = True,
    stages_stretch: str = "asinh",
    final_stretch: str = "asinh",
    q_lo: float = 0.0,
    q_hi: float = 99.9,

    # optional ROI montage (requires plot_star_rois in frame.py)
    make_rois: bool = True,
    roi_n: int = 6,
    roi_half_size: int = 90,
    roi_min_sep_px: int = 140,
    roi_stretch: str = "log",
) -> Tuple[Any, Any, Dict[str, str]]:
    """
    Run one simulation case (payload-ish defaults), display + save outputs, return results.

    Returns
    -------
    frame2, res2, paths
      paths is a dict with keys like: "fits", "final_png", "stages_png", "rois_png"
    """

    # Local imports so this module is importable even when optional deps aren't installed
    from .camera import Camera
    from .lens import Lens
    from .mask import Mask
    from .frame import make_blank_frame, display_frame
    from .render import RenderConfig, render, plot_render_stages
    from .frame import save_frame_fits  # adjust if your save function lives elsewhere

    out_path = Path(out_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Build camera/lens/frame
    # -----------------------
    cam = Camera(
        nx=int(cam_nx),
        ny=int(cam_ny),
        pixel_um=float(cam_pixel_um),
        read_noise_e=float(cam_read_noise_e),
        gain_e_per_adu=float(cam_gain_e_per_adu),
        qe=float(cam_qe),
    )

    lens = Lens(
        focal_mm=float(lens_focal_mm),
        f_number=float(lens_f_number),
        transmission=float(lens_transmission),
    )

    frame = make_blank_frame(cam, lens, ra0_deg=float(ra0_deg), dec0_deg=float(dec0_deg), rot_deg=float(rot_deg))

    # -----------------------
    # Mask
    # -----------------------
    if mask_aperture_diam_mm is None:
        # entrance pupil diameter ~ f / f#
        mask_ap_diam = float(lens.focal_mm) / max(float(lens.f_number), 1e-12)
    else:
        mask_ap_diam = float(mask_aperture_diam_mm)

    mask = Mask(
        kind=mask_kind,
    
        # common
        angle_deg=float(mask_angle_deg),
        psf_size_px=int(mask_psf_size_px),
    
        # aperture-ish (used by poppy paths; safe for others)
        aperture_diam_mm=float(mask_ap_diam),
        obstruction_frac=float(mask_obstruction_frac),
    
        # spider params
        vane_width_mm=float(mask_vane_width_mm),
        n_vanes=int(mask_n_vanes),
    
        # grating params
        lines_per_mm=float(mask_lines_per_mm),
        grating_model=str(mask_grating_model),
        order_max=int(mask_order_max),
        order_rel_amp=float(mask_order_rel_amp),
    
        duty_cycle=float(mask_duty_cycle),
        pupil_samples=int(mask_pupil_samples),
        n_lambda=int(mask_n_lambda),
    )

    # probably shouldn't do this here but its easy for now
    plate_scale = 206265.0 * (cam.pixel_um / 1000.0) / lens.focal_mm
    sigma_arcsec = seeing_fwhm_arcsec / 2.355
    
    # -----------------------
    # RenderConfig
    # -----------------------
    cfg = RenderConfig(
        exposure_s=float(exposure_s),
        sky_mu_mag_per_arcsec2=float(sky_mu_mag_per_arcsec2),
        zeropoint_e_per_s=float(zeropoint_e_per_s),
        lambda_eff_nm=float(lambda_eff_nm),
        band_nm=float(band_nm),
        psf_sigma_px=float(sigma_arcsec / plate_scale),
        mask=mask,

        jitter_pointing_rms=float(jitter_pointing_rms),

        enable_sky=bool(enable_sky),
        enable_stars=bool(enable_stars),
        enable_psf=bool(enable_psf),
        enable_jitter=bool(enable_jitter),
        enable_noise=bool(enable_noise),

        seed=int(seed),
    )

    # -----------------------
    # Run
    # -----------------------
    frame2, res2 = render(frame, cfg, stars=stars_csv)

    paths: Dict[str, str] = {}

    # -----------------------
    # Plots (display and/or save)
    # -----------------------
    if show_plots or save_pngs:
        import matplotlib.pyplot as plt

        # stages
        fig_s, axs_s = plot_render_stages(frame2, res2, cmap="gray", stretch=stages_stretch, shared_scale=True)
        if fig_s is None:
            fig_s = plt.gcf()

        if save_pngs:
            p = out_path / "stages.png"
            fig_s.savefig(p, dpi=160)
            paths["stages_png"] = str(p)

        if show_plots:
            plt.show()
        else:
            plt.close(fig_s)

        # final view
        fig_f, ax_f, im_f = display_frame(frame2, cmap="gray", stretch=final_stretch, q_lo=q_lo, q_hi=q_hi)
        if save_pngs:
            p = out_path / "final.png"
            fig_f.savefig(p, dpi=160)
            paths["final_png"] = str(p)
        if show_plots:
            plt.show()
        else:
            plt.close(fig_f)

        # ROI montage (optional)
        if make_rois:
            try:
                from .frame import plot_star_rois  # if you added it
            except Exception:
                plot_star_rois = None

            if plot_star_rois is not None:
                fig_r = plot_star_rois(
                    frame2,
                    image=res2.final_e,
                    n=int(roi_n),
                    half_size=int(roi_half_size),
                    min_sep_px=int(roi_min_sep_px),
                    stretch=roi_stretch,
                )
                if fig_r is None:
                    fig_r = plt.gcf()
                if save_pngs:
                    p = out_path / "rois.png"
                    fig_r.savefig(p, dpi=160)
                    paths["rois_png"] = str(p)
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig_r)

    # -----------------------
    # Save FITS (raw linear electrons)
    # -----------------------
    fits_path = out_path / f"{run_name}.fits"
    out_fits = save_frame_fits(
        str(fits_path),
        frame2,
        cfg=cfg,
        res=res2,
        stage="final_e",
        stars_csv=stars_csv,
        overwrite=overwrite,
    )
    paths["fits"] = str(out_fits)

    # -----------------------
    # Save a tiny text sidecar of params (easy to diff)
    # -----------------------
    if save_pngs:
        sidecar = out_path / "params.txt"
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("# --- Camera ---\n")
            f.write(repr(cam) + "\n\n")
            f.write("# --- Lens ---\n")
            f.write(repr(lens) + "\n\n")
            f.write("# --- Mask ---\n")
            f.write(repr(mask) + "\n\n")
            f.write("# --- RenderConfig ---\n")
            f.write(repr(cfg) + "\n\n")
        paths["params_txt"] = str(sidecar)

    return frame2, res2, paths