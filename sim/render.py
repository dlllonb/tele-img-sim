# sim/render.py
from dataclasses import dataclass
import numpy as np

from .frame import Frame
from .mask import Mask
from .physics.stars import stars_layer
from .physics.psf import apply_psf
from .physics.jitter import apply_jitter
from .physics.noise import apply_noise
from .physics.sky import sky_layer


@dataclass(frozen=True)
class RenderConfig:
    # --- primary knob ---
    exposure_s: float = 0.1

    # --- photometry / background ---
    sky_e_per_px_s: float = 3.0 # estimate of bortle ~5 sky 
    sky_mu_mag_per_arcsec2: float = 21.0
    zeropoint_e_per_s: float = 0.0  # mag=0 -> e/s, placeholder for later
    lambda_eff_nm: float = 550.0
    band_nm: float = 90.0

    # --- optics ---o
    mask: Mask = Mask()
    psf_sigma_px: float = 1.0

    # --- jitter ---
    jitter_pointing_rms: float = 0.0

    # --- toggles ---
    enable_sky: bool = True
    enable_stars: bool = True
    enable_psf: bool = True
    enable_jitter: bool = True
    enable_noise: bool = True

    # --- reproducibility ---
    seed: int = 0


@dataclass
class RenderResult:
    # Stored as float electron images. Any can be None if stage skipped.
    sky_e: np.ndarray | None = None
    stars_e_pre_psf: np.ndarray | None = None
    stars_e_post_psf: np.ndarray | None = None
    mean_e: np.ndarray | None = None
    after_jitter_e: np.ndarray | None = None
    final_e: np.ndarray | None = None


def _stop_here(stop_after: str | None, stage: str) -> bool:
    return (stop_after is not None) and (stop_after == stage)


def render(frame: Frame,
           cfg: RenderConfig,
           stars=None,
           stop_after: str | None = None,
           return_intermediates: bool = True):
    """
    Build frame.image via a staged rendering pipeline.

    Parameters
    ----------
    frame : Frame
        Frame object; frame.image will be overwritten with the final image.
    cfg : RenderConfig
        Rendering settings.
    stars : optional
        Placeholder StarField/cat slice for later. Can be None for now.
    stop_after : str or None
        If set, stops after a named stage and writes that stage output into frame.image.
        Valid stage names:
          "sky", "stars_pre_psf", "psf", "mean", "jitter", "noise"
    return_intermediates : bool
        If True, returns (frame, RenderResult). Else returns frame.

    Returns
    -------
    frame or (frame, RenderResult)
    """
    rng = np.random.default_rng(cfg.seed)
    res = RenderResult()

    ny, nx = frame.image.shape

    # ---- 1) sky layer (electrons) ----
    if cfg.enable_sky:
        if getattr(cfg, "sky_e_per_px_s", 0.0) and cfg.sky_e_per_px_s > 0.0:
            sky_rate = float(cfg.sky_e_per_px_s)
        else:
            sky_rate = sky_layer(frame, cfg)
    
        sky_e = np.full((ny, nx), sky_rate * cfg.exposure_s, dtype=np.float32)
    else:
        sky_e = np.zeros((ny, nx), dtype=np.float32)

    res.sky_e = sky_e
    if _stop_here(stop_after, "sky"):
        frame.image = sky_e
        return (frame, res) if return_intermediates else frame

    # ---- 2) stars layer (electrons, pre-PSF) ----
    if cfg.enable_stars:
        stars_e = stars_layer(frame, stars, cfg, rng=rng)
    else:
        stars_e = np.zeros((ny, nx), dtype=np.float32)

    res.stars_e_pre_psf = stars_e
    if _stop_here(stop_after, "stars_pre_psf"):
        frame.image = stars_e
        return (frame, res) if return_intermediates else frame

    # ---- 3) PSF ----
    if cfg.enable_psf:
        stars_psf_e = apply_psf(stars_e, frame, cfg)
    else:
        stars_psf_e = stars_e

    res.stars_e_post_psf = stars_psf_e
    if _stop_here(stop_after, "psf"):
        frame.image = stars_psf_e
        return (frame, res) if return_intermediates else frame

    # ---- 4) combine mean signal ----
    mean_e = sky_e + stars_psf_e
    res.mean_e = mean_e
    if _stop_here(stop_after, "mean"):
        frame.image = mean_e
        return (frame, res) if return_intermediates else frame


    # ---- 6) jitter blur ----
    if cfg.enable_jitter:
        after_jitter = apply_jitter(mean_e, frame, cfg)
    else:
        after_jitter = mean_e

    res.after_jitter_e = after_jitter
    if _stop_here(stop_after, "jitter"):
        frame.image = after_jitter
        return (frame, res) if return_intermediates else frame

    # ---- 7) noise ----
    if cfg.enable_noise:
        final_e = apply_noise(after_jitter, frame, cfg, rng=rng)
    else:
        final_e = after_jitter

    res.final_e = final_e
    frame.image = final_e

    if _stop_here(stop_after, "noise"):
        return (frame, res) if return_intermediates else frame

    return (frame, res) if return_intermediates else frame


def plot_render_stages(frame, res, *,
                       figsize=(12, 6),
                       cmap="gray",
                       shared_scale=True,
                       show_colorbar=False,
                       stretch="asinh",
                       ref_stage="final",
                       q_lo=5.0,
                       q_hi=99.9,
                       eps=1e-12):
    """
    Visualize intermediate render stages as a 2x4 subplot grid.

    stretch:
      - "linear": raw values, shared vmin/vmax via percentiles on ref_stage (or per-panel if shared_scale=False)
      - "percentile": same as linear but just a reminder that scaling is percentile-based
      - "asinh": astronomy-style stretch; best for huge dynamic range
      - "log": log10(1 + x/scale); also good but harsher than asinh

    ref_stage: which panel defines the shared scaling when shared_scale=True.
      One of: "empty","sky_e","stars_pre_psf","stars_post_psf","mean_e","after_mask","after_jitter","final"
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ny, nx = frame.image.shape
    empty = np.zeros((ny, nx), dtype=np.float32)

    panels = [
        #("empty", empty),
        ("sky_e", res.sky_e),
        ("stars_pre_psf", res.stars_e_pre_psf),
        ("stars_post_psf", res.stars_e_post_psf),
        ("mean_e", res.mean_e),
       # ("after_mask", res.after_mask_e),
        ("after_jitter", res.after_jitter_e),
        ("final", res.final_e),
    ]

    # Replace None with NaN arrays for plotting
    arrays = []
    for _, arr in panels:
        if arr is None:
            arrays.append(np.full((ny, nx), np.nan, dtype=np.float32))
        else:
            arrays.append(arr.astype(np.float32, copy=False))

    # --- helper: robust percentiles on finite values ---
    # --- choose reference scaling (shared dynamic range) ---
    name_to_idx = {name: i for i, (name, _) in enumerate(panels)}
    ref_idx = name_to_idx.get(ref_stage, len(panels) - 1)
    
    def _pstats(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(a, q_lo))
        hi = float(np.percentile(a, q_hi))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = lo + 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    
    if shared_scale:
        ref_lo, ref_hi = _pstats(arrays[ref_idx])
        shared_span = max(ref_hi - ref_lo, eps)   # <- shared "contrast scale"
    else:
        shared_span = None

    # --- display transform ---
    def _transform(a):
        a = np.asarray(a, dtype=np.float32)

       # IMPORTANT: baseline is per-panel so sky pedestal doesn't zero out star-only panels
        lo_i, hi_i = _pstats(a)
        x = a - lo_i
        x = np.where(np.isfinite(x), x, np.nan)
        x = np.clip(x, 0.0, None)
        
        # scale: shared if requested, otherwise per-panel
        span = shared_span if (shared_span is not None) else max(hi_i - lo_i, eps)
        
        if stretch in ("linear", "percentile"):
            disp = x
            vmin, vmax = 0.0, span
        
        elif stretch == "asinh":
            disp = np.arcsinh(x / span)
            vmin, vmax = 0.0, float(np.arcsinh(1.0))
        
        elif stretch == "log":
            disp = np.log10(1.0 + x / span)
            vmin, vmax = 0.0, float(np.log10(2.0))
        
        else:
            raise ValueError(f"Unknown stretch='{stretch}'")
        
        return disp, vmin, vmax


    fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axs = axs.ravel()

    ims = []
    for k, (name, _) in enumerate(panels):
        ax = axs[k]

        if shared_scale:
            disp, vmin, vmax = _transform(arrays[k])
        else:
            disp, vmin, vmax = _transform(arrays[k], None, None)

        im = ax.imshow(disp, origin="lower", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_aspect("auto", adjustable="box")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    if show_colorbar:
        fig.colorbar(ims[-1], ax=axs.tolist(), fraction=0.02, pad=0.02)

    # return fig, axs
    