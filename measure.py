#!/usr/bin/env python3
"""measure.py — command-line entry point for the measurement pipeline.

Usage
-----
    python measure.py                               # uses measure_config.toml
    python measure.py --config path/to/cfg.toml
    python measure.py --input data/my_image.fit     # overrides [io].input_fits
    python measure.py --run-name my_run             # overrides [io].run_name
    python measure.py --no-show                     # suppress interactive plots
    python measure.py --no-save                     # skip output files

Config file format: TOML (see measure_config.toml for annotated reference).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress non-fatal deprecation warnings from third-party libraries so CLI
# output stays clean.  These are library-author notices, not user errors.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from astropy.utils.exceptions import AstropyDeprecationWarning
    warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
except ImportError:
    pass
try:
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
except ImportError:
    pass


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
# Map [metadata_overrides] and [metadata_defaults] to the keyword names
# expected by MeasurementMetadata.from_header
# ---------------------------------------------------------------------------

# TOML key → header keyword used by from_header (first key in each _pick tuple)
_META_KEY_MAP = {
    "ra_deg":                   "RA0",
    "dec_deg":                  "DEC0",
    "rot_deg":                  "ROT",
    "plate_scale_arcsec_per_px":"PLTSCL",
    "exposure_s":               "EXPTIME",
    "focal_length_mm":          "FOCALLEN",
    "f_number":                 "FNUMBER",
    "pixel_size_um":            "PIXSIZE",
    "read_noise_e":             "RDNOISE",
    "gain_e_per_adu":           "GAIN",
    "qe":                       "QE",
    "lambda_nm":                "LAMBDAN",
    "bandwidth_nm":             "BANDWID",
    "jitter_rms_arcsec":        "JITRMS",
    "mask_kind":                "MASKKIND",
    "mask_angle_deg":           "MASKANG",
    "grating_lines_per_mm":     "GRATLIN",
}


def _build_meta_dict(section: dict) -> Dict[str, Any]:
    """Convert a TOML [metadata_overrides] or [metadata_defaults] section
    into the header-keyword dict expected by MeasurementMetadata.from_header."""
    out: Dict[str, Any] = {}
    for toml_key, header_key in _META_KEY_MAP.items():
        if toml_key in section:
            out[header_key] = section[toml_key]
    return out


# ---------------------------------------------------------------------------
# Build pipeline kwargs from config dict
# ---------------------------------------------------------------------------

def _config_to_kwargs(cfg: dict, cli_overrides: dict) -> dict:
    io      = cfg.get("io", {})
    display = cfg.get("display", {})
    ov_sec  = cfg.get("metadata_overrides", {})
    df_sec  = cfg.get("metadata_defaults",  {})

    # blank run_name → derive from input filename stem (e.g. "fuji6_asi178_100_15s")
    run_name = io.get("run_name", "") or None

    kwargs = {
        "filepath":          io.get("input_fits",         ""),
        "output_dir":        io.get("output_dir",         "out"),
        "run_name":          run_name,
        "save_outputs":      io.get("save_outputs",       True),
        "keep_intermediates":io.get("keep_intermediates", True),
        "verbose":           io.get("verbose",            True),
        "show":              display.get("show",          False),  # default off for CLI
        "overrides":         _build_meta_dict(ov_sec),
        "defaults":          _build_meta_dict(df_sec),
    }

    # CLI wins over config
    kwargs.update({k: v for k, v in cli_overrides.items() if v is not None})

    # Derive run_name from input filepath stem if still unset
    if not kwargs.get("run_name"):
        kwargs["run_name"] = Path(kwargs["filepath"]).stem

    return kwargs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the measurement pipeline on a FITS image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   default="measure_config.toml",
                   help="Path to TOML config file.")
    p.add_argument("--input",    dest="filepath",  default=None,
                   help="Override [io].input_fits — path to input FITS file.")
    p.add_argument("--run-name", dest="run_name",  default=None,
                   help="Override [io].run_name.")
    p.add_argument("--out-dir",  dest="out_dir",   default=None,
                   help="Override [io].output_dir.")
    p.add_argument("--no-show",  dest="show", action="store_false",
                   default=None, help="Suppress interactive plots.")
    p.add_argument("--no-save",  dest="save_outputs", action="store_false",
                   default=None, help="Skip output files.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    cfg = _load_toml(cfg_path)

    cli_overrides = {
        "filepath":     args.filepath,
        "run_name":     args.run_name,
        "output_dir":   args.out_dir,
        "show":         args.show,
        "save_outputs": args.save_outputs,
    }

    kwargs = _config_to_kwargs(cfg, cli_overrides)

    if not kwargs["filepath"]:
        sys.exit("No input FITS file specified. Set [io].input_fits in the config or use --input.")

    # Bootstrap project root
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from measure.pipeline import run_measurement_pipeline

    print(f"Running measurement pipeline on: {kwargs['filepath']}")
    result = run_measurement_pipeline(**kwargs)

    print(f"\nSuccess: {result.success}")
    print(f"Platesolve: {result.platesolve_result.success}")
    print(f"Stripe angle: {result.spike_result.success}  "
          f"({result.spike_result.image_angle_deg}° ± {result.spike_result.sigma_angle_deg}°)")
    if result.metrics.sky_angle_deg is not None:
        print(f"Sky angle (East of North): {result.metrics.sky_angle_deg:.3f}° "
              f"± {result.metrics.final_sigma_deg:.3f}°")

    if result.output_paths:
        print("\nOutput files:")
        for key, path in result.output_paths.items():
            print(f"  {key:25s} {path}")

    return result


if __name__ == "__main__":
    main()
