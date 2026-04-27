"""
Test 3 — Plate-solve a real FITS image (m81.fits).

Loads `tests/m81.fits`, prepares the star branch, submits detected sources
to nova.astrometry.net, and saves a diagnostic figure showing:
  • The background-subtracted star branch image in greyscale.
  • Green circles drawn at every source sent to the plate solver.
  • A WCS sky-coordinate grid (RA/Dec) overlaid if the solve succeeds.
  • Annotation with the solved field centre, rotation, and solve status.

Prerequisites:
  • Place `m81.fits` in this directory (tests/).
  • Place your nova.astrometry.net API key in `astrometry_api.txt` at the
    project root (tele-img-sim/astrometry_api.txt).  The key is a single
    line of plain text, e.g. "abcdefghij123456".

Outputs saved to this directory:
    test_platesolve_m81.png
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS

from measure.preprocess import prepare_star_branch_input
from measure.platesolve import run_platesolve, _detect_sources
from measure.types import MeasurementMetadata, BranchImageResult

HERE = Path(__file__).resolve().parent
FITS_PATH = HERE / "m81.fits"


def _load_fits(path: Path):
    with fits.open(path) as hdul:
        data   = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D FITS image, got ndim={data.ndim}")
    return data, header


def run_test():
    if not FITS_PATH.exists():
        raise FileNotFoundError(
            f"Input FITS not found: {FITS_PATH}\n"
            "Place m81.fits in the tests/ directory and re-run."
        )

    # -------------------------------------------------------------------
    # 1. Load image + metadata.
    # -------------------------------------------------------------------
    print(f"Loading {FITS_PATH.name} …")
    image, header = _load_fits(FITS_PATH)
    print(f"  Image shape: {image.shape}  dtype: {image.dtype}")

    # Build metadata from header.  Add overrides here if the header is
    # missing critical fields (e.g. plate scale, RA/Dec hints for M81).
    meta = MeasurementMetadata.from_header(
        header,
        overrides={},   # add e.g. "PLTSCL": 1.23 if header lacks plate scale
        defaults={},
    )
    print(f"  Metadata — RA: {meta.ra_deg}  Dec: {meta.dec_deg}  "
          f"plate_scale: {meta.plate_scale_arcsec_per_px}")

    # -------------------------------------------------------------------
    # 2. Star-branch preprocessing.
    # -------------------------------------------------------------------
    star_branch = prepare_star_branch_input(image, meta)
    print(f"  Star branch: {star_branch.messages}")

    # -------------------------------------------------------------------
    # 3. Detect sources (independently, so we can draw them before the
    #    solve and show exactly what was submitted).
    # -------------------------------------------------------------------
    sources = _detect_sources(star_branch.image)
    if sources is None:
        raise RuntimeError("photutils is required for source detection.")

    xs_all = sources["x"]
    ys_all = sources["y"]
    fs_all = sources["flux"]
    print(f"  Detected {len(xs_all)} raw sources.")

    # -------------------------------------------------------------------
    # 4. Plate solve.
    # -------------------------------------------------------------------
    print("Submitting to nova.astrometry.net (this may take 1–3 minutes) …")
    plate_res = run_platesolve(star_branch, meta)

    for msg in plate_res.messages:
        print(f"  {msg}")

    # If the solve was never attempted (e.g. missing API key), debug["sources"]
    # is absent.  Falling through to the figure step would draw circles for
    # every raw detection (potentially thousands) and hang on savefig.
    if plate_res.debug.get("sources") is None:
        print("  Plate solve was not attempted — skipping figure generation.")
        return

    # The platesolve stores the deduped, top-N sources it actually submitted.
    submitted_xs = plate_res.debug.get("sources", {}).get("x", xs_all)
    submitted_ys = plate_res.debug.get("sources", {}).get("y", ys_all)
    print(f"  Submitted {len(submitted_xs)} sources to solver.")
    print(f"  Solve success: {plate_res.success}")

    # -------------------------------------------------------------------
    # 5. Build figure.
    # -------------------------------------------------------------------
    wcs = None
    if plate_res.success and plate_res.wcs_info is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(plate_res.wcs_info)

    fig = plt.figure(figsize=(12, 10))
    subplot_kw = {"projection": wcs} if wcs is not None else {}
    ax = fig.add_subplot(111, **subplot_kw)

    # Display: asinh stretch
    finite = star_branch.image[np.isfinite(star_branch.image)]
    vmin, vmax = np.percentile(finite, [0.5, 99.5])
    disp = np.arcsinh(np.clip(star_branch.image, vmin, vmax) / max(abs(vmax), 1e-30))
    ax.imshow(
        disp,
        origin="lower",
        cmap="gray",
        vmin=float(np.arcsinh(vmin / max(abs(vmax), 1e-30))),
        vmax=float(np.arcsinh(1.0)),
        interpolation="nearest",
    )

    # WCS coordinate grid
    if wcs is not None:
        ax.coords.grid(True, color="cyan", alpha=0.45, linestyle="--", linewidth=0.8)
        ax.coords["ra"].set_axislabel("RA")
        ax.coords["dec"].set_axislabel("Dec")
        ax.coords["ra"].set_ticklabel(size=8)
        ax.coords["dec"].set_ticklabel(size=8)
    else:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    # Green circles at submitted source positions.
    # When using WCSAxes the default transform is world coordinates, so we
    # must explicitly request the pixel transform for data in pixel coords.
    pix_transform = ax.get_transform("pixel") if wcs is not None else ax.transData
    for x, y in zip(submitted_xs, submitted_ys):
        circ = Circle(
            (x, y), radius=14,
            edgecolor="lime", facecolor="none",
            linewidth=1.3, alpha=0.85,
            transform=pix_transform,
        )
        ax.add_patch(circ)

    # Annotation box
    if plate_res.success:
        solve_text = (
            f"SOLVED\n"
            f"RA  = {plate_res.ra_deg:.5f}°\n"
            f"Dec = {plate_res.dec_deg:.5f}°\n"
            f"rot = {plate_res.rot_deg:.3f}°"
        )
        box_color = "lime"
    else:
        solve_text = "PLATE SOLVE FAILED\n(see console for details)"
        box_color = "red"

    ax.text(
        0.02, 0.97, solve_text,
        transform=ax.transAxes,
        color="white", fontsize=10, fontweight="bold", va="top",
        bbox=dict(facecolor="black", alpha=0.6, pad=5, edgecolor=box_color),
    )

    n_src = len(submitted_xs)
    ax.set_title(
        f"Test 3 — Plate solve: {FITS_PATH.name}  |  "
        f"{n_src} sources submitted (green circles)  |  "
        f"{'WCS grid shown in cyan' if wcs is not None else 'no WCS — solve failed'}",
        fontsize=10,
    )

    out_path = HERE / "test_platesolve_m81.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path.name}")


if __name__ == "__main__":
    run_test()
