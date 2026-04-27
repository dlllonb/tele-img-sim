# tele-img-sim

A Python framework for simulating and analysing astrophotographs taken by a drone-mounted star camera equipped with a diffraction mask. The goal is to optimise a camera + lens + mask system for drone-based polarisation calibration — specifically, to maximise plate-solve success rate and diffraction stripe angle estimation precision under realistic drone pointing jitter.

The science goal behind this project (also in the project proposal) is to develop a framework for testing and informing design ideas for a drone mounted polarization calibration system. Modern CMB expermiments have tight requirements for the calibration of absolute polarization measurements, and one method to do this calibration is by providing an external artificial source in the sky. Such a source could provide a known beam of polarized light directed back towards the observatory which can be used for calibration. This method, however, requires that the artificial source have a well known polarization in the sky frame itself. One potential method to "calibrate the calibration source" and achieve a well known polarization in the sky frame is to use a star camera pointing through a diffraction grating that is aligned to the polarizing grid on the artificial source. The polarizing grid orientation defines the outgoing polarization of the source, the diffraction grating is closely aligned to the polarizing grid, and so the diffraction pattern seen by the camera defines the outgoing polarization vector, and this vector can be immediately converted to the sky frame via the astrometric solution from the star camera. In this way, a star camera image contains both pieces of information (i.e. the polarizing grid angle, and the astrometric solution) that are needed to full define the outgoing polarrization vector. The question then is what hardware (i.e. camera, lens, diffraction grating, drone) will produce an image that will work well for this process. So, this project develops pipelines that 1) simulate how the images will look based on the hardware, and 2) attempt to extract the astrometric solution and the spectra angle then reconstruct the outgoing polarization. 

---

## How it works

The project has two independent pipelines.

### Simulation (`sim/`)

Renders a synthetic FITS image of the night sky as it would appear through a given camera, lens, and diffraction mask, under realistic drone jitter.

The render pipeline runs in stages:

1. **Sky background** — uniform sky glow from a configurable surface brightness
2. **Stars** — a star catalogue (CSV) is projected onto the sensor; flux scales with exposure time and photometric zeropoint
3. **PSF / diffraction** — convolved with a Gaussian seeing kernel, or a physically-modelled diffraction kernel when a grating or spider mask is active
4. **Jitter** — an additional Gaussian blur from drone pointing instability (arcsec RMS)
5. **Noise** — Poisson shot noise + Gaussian read noise

Output is a FITS file with full metadata in the header, plus diagnostic PNGs.

### Measurement (`measure/`)

Analyses a real or simulated FITS image to recover the diffraction stripe orientation in sky coordinates.

The pipeline runs in stages:

1. **Preprocessing** — two image branches are produced: a *star branch* (background-subtracted, optimised for point sources) and a *stripe branch* (smoothed to enhance diffraction streaks)
2. **Stripe angle** — connected-component analysis on the stripe branch estimates the diffraction angle in the image frame, with uncertainty
3. **Plate solving** — detected stars from the star branch are submitted to [nova.astrometry.net](https://nova.astrometry.net) to recover RA, Dec, and the astrometric rotation of the detector
4. **Metrics** — the image-frame stripe angle and the astrometric rotation are combined to produce the final diffraction orientation **East of North** in sky coordinates

Output includes FITS branch images, a JSON/text summary, and diagnostic PNGs.

---

## Requirements

- Python 3.11 or later
- An internet connection for plate solving (nova.astrometry.net API)
- A free API key from [nova.astrometry.net](https://nova.astrometry.net) — register, then copy your key into a file called `astrometry_api.txt` in the project root (one line, no quotes)

Install all dependencies into a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `poppy` is only required when using `mask_kind = "grating"` or `"spider"` with `grating_model = "poppy"` / `spider_model = "poppy"`. The faster `"analytic"` / `"manual"` backends work without it.

---

## Running the simulation

### From the terminal

```bash
python sim.py
```

This reads `sim_config.toml` and writes outputs to `out/<run_name>/`.

**Common overrides via flags:**

```bash
# Use a different config file
python sim.py --config my_experiment.toml

# Change the run name and output directory
python sim.py --run-name my_run --out-dir results

# Use a different star catalogue
python sim.py --stars-csv sim/physics/starfields/field1.csv

# Show interactive plots (suppressed by default in CLI mode)
python sim.py --show
```

**Key parameters in `sim_config.toml`:**

```toml
[pointing]
ra0_deg  = 10.127   # pointing centre
dec0_deg = 56.537
rot_deg  = 0.0      # camera roll

[camera]
nx           = 3096
ny           = 2080
pixel_um     = 2.4
read_noise_e = 1.6
qe           = 0.6

[lens]
focal_mm     = 25
f_number     = 1.2

[mask]
kind         = "grating"   # "none" | "grating" | "spider"
angle_deg    = 23
lines_per_mm = 100.0

[render]
exposure_s          = 5.0
jitter_pointing_rms = 20.0   # arcsec RMS drone jitter
seeing_fwhm_arcsec  = 2.0
```

### From a notebook

```python
from sim.simulator import run_sim_and_report

frame, res, paths = run_sim_and_report(
    run_name="my_run",
    stars_csv="sim/physics/starfields/field3.csv",
    lens_focal_mm=25,
    lens_f_number=1.2,
    mask_kind="grating",
    mask_angle_deg=23,
    mask_lines_per_mm=100.0,
    exposure_s=5.0,
    jitter_pointing_rms=20.0,
    show_plots=True,
)
```

---

## Running the measurement pipeline

> **Before you start:** create `astrometry_api.txt` in the project root containing your nova.astrometry.net API key. Without it the plate-solve step is skipped and no sky-frame angle can be computed.

### From the terminal

```bash
python measure.py
```

This reads `measure_config.toml` and processes the FITS file specified under `[io].input_fits`.

**Common overrides via flags:**

```bash
# Point at a specific FITS file
python measure.py --input data/my_image.fit

# Custom run name (output goes to out/<run_name>/)
python measure.py --run-name my_measurement

# Show interactive plots after the run
python measure.py --show

# Dry run — process but don't write any output files
python measure.py --no-save
```

**Key parameters in `measure_config.toml`:**

```toml
[io]
input_fits   = "data/fuji6_asi178_100_15s.fit"
output_dir   = "out"
run_name     = ""      # blank = use input filename as run name

[display]
show = false           # set true to open plots interactively

# Override FITS header values if needed:
[metadata_overrides]
# focal_length_mm    = 100.0
# mask_angle_deg     = 23.0
# grating_lines_per_mm = 100.0
```

### From a notebook

```python
from measure.pipeline import run_measurement_pipeline

result = run_measurement_pipeline(
    "data/my_image.fit",
    output_dir="out",
    run_name="my_measurement",
    show=True,
)

print(result.metrics.sky_angle_deg)   # diffraction angle, East of North (degrees)
print(result.spike_result.image_angle_deg)  # raw image-frame angle
print(result.platesolve_result.rot_deg)     # astrometric rotation
```

---

## Output files

Both pipelines write their outputs to `out/<run_name>/`.

| File | Pipeline | Description |
|---|---|---|
| `<run_name>.fits` | Simulation | Simulated image with full metadata header |
| `final.png` | Simulation | Final rendered image |
| `stages.png` | Simulation | Six-panel pipeline stage overview |
| `rois.png` | Simulation | Star ROI cutout montage |
| `params.txt` | Simulation | Human-readable run parameters |
| `star_branch.fits` | Measurement | Background-subtracted star image |
| `stripe_branch.fits` | Measurement | Smoothed stripe image |
| `stripe_angle.png` | Measurement | Stripe branch with fitted angle overlay |
| `final_result.png` | Measurement | RA/Dec grid, sources, north arrow, and stripe orientation |
| `summary.json` | Measurement | Machine-readable results (angles, uncertainties, flags) |
| `summary.txt` | Measurement | Human-readable pipeline log |

---

## Project structure

```
tele-img-sim/
├── sim/                     # Simulation package
│   ├── simulator.py         # run_sim_and_report() entry point
│   ├── render.py            # Staged render pipeline + RenderConfig
│   ├── camera.py            # Camera sensor model
│   ├── lens.py              # Lens model
│   ├── mask.py              # Diffraction mask (grating / spider / none)
│   ├── frame.py             # Image frame, WCS, coordinate transforms
│   └── physics/             # Sky, stars, PSF, jitter, noise models
│       └── starfields/      # Star catalogue CSV files
├── measure/                 # Measurement package
│   ├── pipeline.py          # run_measurement_pipeline() entry point
│   ├── preprocess.py        # Star and stripe branch preprocessing
│   ├── spikes.py            # Ensemble diffraction angle estimator
│   ├── platesolve.py        # nova.astrometry.net plate solver
│   ├── metrics.py           # Sky-frame angle combination
│   ├── io.py                # FITS I/O and output utilities
│   └── types.py             # Shared dataclasses
├── notebooks/
│   ├── dev/                 # Development and exploration notebooks
│   └── test/                # Experiment notebooks
├── sim.py                   # CLI entry point for simulation
├── measure.py               # CLI entry point for measurement
├── sim_config.toml          # Simulation configuration (all parameters)
├── measure_config.toml      # Measurement configuration (all parameters)
├── requirements.txt         # Python dependencies
└── astrometry_api.txt       # API key (not tracked by git — create manually)
```
