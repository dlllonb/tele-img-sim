# sim/render.py
from dataclasses import dataclass
import numpy as np

from .frame import Frame
from .physics.stars import stars_layer
from .physics.psf import apply_psf
from .physics.mask import apply_mask
from .physics.jitter import apply_jitter
from .physics.noise import apply_noise


@dataclass(frozen=True)
class RenderConfig:
    # --- primary knob ---
    exposure_s: float = 0.1

    # --- photometry / background ---
    sky_e_per_px_s: float = 3 # estimate of bortle ~5 sky 
    zeropoint_e_per_s: float = 0.0  # mag=0 -> e/s, placeholder for later

    # --- optics ---o
    psf_sigma_px: float = 1.0

    # --- noise ---
    read_noise_e: float = 0.0

    # --- jitter ---
    jitter_sigma_px: float = 0.0

    # --- toggles ---
    enable_sky: bool = True
    enable_stars: bool = True
    enable_psf: bool = True
    enable_mask: bool = True
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
    after_mask_e: np.ndarray | None = None
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
          "sky", "stars_pre_psf", "psf", "mean", "mask", "jitter", "noise"
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
        # stub sky: uniform background electrons over exposure
        # (kept here because it's trivial; if it grows, move to sim/physics/background.py)
        sky_e = np.full((ny, nx), cfg.sky_e_per_px_s * cfg.exposure_s, dtype=np.float32)
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

    # ---- 5) mask diffraction ----
    if cfg.enable_mask:
        after_mask = apply_mask(mean_e, frame, cfg)
    else:
        after_mask = mean_e

    res.after_mask_e = after_mask
    if _stop_here(stop_after, "mask"):
        frame.image = after_mask
        return (frame, res) if return_intermediates else frame

    # ---- 6) jitter blur ----
    if cfg.enable_jitter:
        after_jitter = apply_jitter(after_mask, frame, cfg)
    else:
        after_jitter = after_mask

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
                       cmap=None,
                       vmin=None,
                       vmax=None,
                       shared_scale=True,
                       show_colorbar=False):
    """
    Visualize intermediate render stages as a 2x4 subplot grid.

    Stages shown (in order):
      1) empty (zeros)
      2) sky_e
      3) stars_e_pre_psf
      4) stars_e_post_psf
      5) mean_e
      6) after_mask_e
      7) after_jitter_e
      8) final_e

    Parameters
    ----------
    frame : Frame
        Used for shape and optional metadata.
    res : RenderResult
        Intermediate images stored by render().
    shared_scale : bool
        If True, uses a common vmin/vmax across all panels (unless vmin/vmax provided).
    vmin, vmax : float or None
        Manual scaling overrides.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ny, nx = frame.image.shape
    empty = np.zeros((ny, nx), dtype=np.float32)

    panels = [
        ("empty", empty),
        ("sky_e", res.sky_e),
        ("stars_pre_psf", res.stars_e_pre_psf),
        ("stars_post_psf", res.stars_e_post_psf),
        ("mean_e", res.mean_e),
        ("after_mask", res.after_mask_e),
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

    if shared_scale and (vmin is None or vmax is None):
        finite_vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.any(np.isfinite(a))])
        if finite_vals.size > 0:
            if vmin is None:
                vmin = float(np.min(finite_vals))
            if vmax is None:
                vmax = float(np.max(finite_vals))

    fig, axs = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    axs = axs.ravel()

    ims = []
    for k, (name, _) in enumerate(panels):
        ax = axs[k]
        im = ax.imshow(arrays[k], origin="lower", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_aspect("auto", adjustable="box")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    if show_colorbar:
        # one shared colorbar using the last image handle
        fig.colorbar(ims[-1], ax=axs.tolist(), fraction=0.02, pad=0.02)

   # return fig, axs

