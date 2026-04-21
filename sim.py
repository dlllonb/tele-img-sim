#!/usr/bin/env python3
"""sim.py — command-line entry point for the tele-img-sim simulation pipeline.

Usage
-----
    python sim.py                          # uses sim_config.toml in the current directory
    python sim.py --config path/to/cfg.toml
    python sim.py --run-name my_run        # overrides [io].run_name
    python sim.py --out-dir results        # overrides [io].out_dir
    python sim.py --no-show                # suppress interactive plots
    python sim.py --no-save                # skip PNG/FITS output

Config file format: TOML (see sim_config.toml for annotated reference).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# TOML loader (stdlib tomllib in Python 3.11+, else tomli)
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # type: ignore  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            sys.exit(
                "TOML support requires Python 3.11+ or the 'tomli' package.\n"
                "Install with:  pip install tomli"
            )
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Build keyword arguments for run_sim_and_report from a config dict
# ---------------------------------------------------------------------------

def _config_to_kwargs(cfg: dict, cli_overrides: dict) -> dict:
    io      = cfg.get("io", {})
    point   = cfg.get("pointing", {})
    cam     = cfg.get("camera", {})
    lens    = cfg.get("lens", {})
    mask    = cfg.get("mask", {})
    render  = cfg.get("render", {})
    toggles = render.get("toggles", {})
    out     = cfg.get("output", {})

    kwargs = {
        # I/O
        "out_dir":   io.get("out_dir",   "out"),
        "run_name":  io.get("run_name",  "sim_run"),
        "stars_csv": io.get("stars_csv", "sim/physics/starfields/field3.csv"),
        "overwrite": io.get("overwrite", True),

        # Pointing
        "ra0_deg":  point.get("ra0_deg",  30.0),
        "dec0_deg": point.get("dec0_deg", 70.0),
        "rot_deg":  point.get("rot_deg",  0.0),

        # Camera
        "cam_nx":            cam.get("nx",             3096),
        "cam_ny":            cam.get("ny",             2080),
        "cam_pixel_um":      cam.get("pixel_um",       2.4),
        "cam_read_noise_e":  cam.get("read_noise_e",   2.0),
        "cam_gain_e_per_adu":cam.get("gain_e_per_adu", 1.0),
        "cam_qe":            cam.get("qe",             0.6),

        # Lens
        "lens_focal_mm":      lens.get("focal_mm",     120.0),
        "lens_f_number":      lens.get("f_number",     1.8),
        "lens_transmission":  lens.get("transmission", 0.9),

        # Mask — common
        "mask_kind":            mask.get("kind",            "none"),
        "mask_angle_deg":       mask.get("angle_deg",       12.0),
        "mask_psf_size_px":     mask.get("psf_size_px",     300),
        "mask_aperture_diam_mm":mask.get("aperture_diam_mm", None),

        # Mask — grating
        "mask_grating_model": mask.get("grating_model", "analytic"),
        "mask_lines_per_mm":  mask.get("lines_per_mm",  0.0),
        "mask_duty_cycle":    mask.get("duty_cycle",    0.5),
        "mask_pupil_samples": mask.get("pupil_samples", 512),
        "mask_order_max":     mask.get("order_max",     1),
        "mask_order_rel_amp": mask.get("order_rel_amp", 0.12),
        "mask_n_lambda":      mask.get("n_lambda",      9),

        # Mask — spider
        "mask_spider_model":    mask.get("spider_model",    "manual"),
        "mask_n_vanes":         mask.get("n_vanes",         2),
        "mask_vane_width_mm":   mask.get("vane_width_mm",   0.6),
        "mask_obstruction_frac":mask.get("obstruction_frac",0.0),

        # Render
        "exposure_s":             render.get("exposure_s",             10.0),
        "sky_mu_mag_per_arcsec2": render.get("sky_mu_mag_per_arcsec2", 21.0),
        "zeropoint_e_per_s":      render.get("zeropoint_e_per_s",      0.0),
        "lambda_eff_nm":          render.get("lambda_eff_nm",          550.0),
        "band_nm":                render.get("band_nm",                90.0),
        "seeing_fwhm_arcsec":     render.get("seeing_fwhm_arcsec",     1.5),
        "jitter_pointing_rms":    render.get("jitter_pointing_rms",    16.0),
        "seed":                   render.get("seed",                   42),

        # Toggles
        "enable_sky":    toggles.get("enable_sky",    True),
        "enable_stars":  toggles.get("enable_stars",  True),
        "enable_psf":    toggles.get("enable_psf",    True),
        "enable_jitter": toggles.get("enable_jitter", True),
        "enable_noise":  toggles.get("enable_noise",  True),

        # Output / display
        "show_plots":     out.get("show_plots",     True),
        "save_pngs":      out.get("save_pngs",      True),
        "stages_stretch": out.get("stages_stretch", "asinh"),
        "final_stretch":  out.get("final_stretch",  "asinh"),
        "q_lo":           out.get("q_lo",           0.0),
        "q_hi":           out.get("q_hi",           99.9),
        "make_rois":      out.get("make_rois",       True),
        "roi_n":          out.get("roi_n",           6),
        "roi_half_size":  out.get("roi_half_size",   90),
        "roi_min_sep_px": out.get("roi_min_sep_px",  140),
        "roi_stretch":    out.get("roi_stretch",     "log"),
    }

    # CLI overrides win over config file
    kwargs.update({k: v for k, v in cli_overrides.items() if v is not None})
    return kwargs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the tele-img-sim simulation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   default="sim_config.toml",
                   help="Path to TOML config file.")
    p.add_argument("--run-name", dest="run_name", default=None,
                   help="Override [io].run_name.")
    p.add_argument("--out-dir",  dest="out_dir",  default=None,
                   help="Override [io].out_dir.")
    p.add_argument("--stars-csv",dest="stars_csv",default=None,
                   help="Override [io].stars_csv.")
    p.add_argument("--no-show",  dest="show_plots", action="store_false",
                   default=None, help="Suppress interactive plots.")
    p.add_argument("--no-save",  dest="save_pngs",  action="store_false",
                   default=None, help="Skip PNG/FITS output.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    cfg = _load_toml(cfg_path)

    cli_overrides = {
        "run_name":   args.run_name,
        "out_dir":    args.out_dir,
        "stars_csv":  args.stars_csv,
        "show_plots": args.show_plots,
        "save_pngs":  args.save_pngs,
    }

    kwargs = _config_to_kwargs(cfg, cli_overrides)

    # Bootstrap project root so sim/ is importable regardless of cwd
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from sim.simulator import run_sim_and_report

    print(f"Running simulation: run_name='{kwargs['run_name']}' → {kwargs['out_dir']}/")
    frame, res, paths = run_sim_and_report(**kwargs)

    print("\nOutput files:")
    for key, path in paths.items():
        print(f"  {key:20s} {path}")

    return frame, res, paths


if __name__ == "__main__":
    main()
