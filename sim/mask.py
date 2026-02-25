# sim/mask.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

MaskKind = Literal["none", "spider", "grating", "bitmap"]


@dataclass(frozen=True)
class Mask:
    """
    Physical diffraction mask placed near the lens entrance pupil.

    v0 intent:
      - "none": no mask (default; should reproduce current results)
      - "spider": Newtonian-style vanes (X, +, etc.)
      - "grating": 1D line grating (lines/mm)
      - "bitmap": arbitrary pupil transmission bitmap loaded from disk

    Note: actual diffraction physics is NOT implemented yet; this class is just plumbing.
    """

    kind: MaskKind = "none"

    # Common orientation (degrees CCW from +x sensor axis)
    angle_deg: float = 0.0

    # --- Spider vane params (for kind="spider") ---
    spider_model: Literal["manual", "poppy"] = "poppy"
    n_vanes: int = 0                 # e.g. 2 for an "X" (two bars crossing), 4 for "+", etc.
    vane_width_mm: float = 0.0       # physical width of vane in mm (at pupil)
    transmission: float = 0.0        # 0 => opaque vanes; >0 => partially transmissive
    spike_radius_px: int = 250      # kernel half-size for spider spikes
    spike_rel_amp: float = 0.10     # relative spike strength vs core
    spike_falloff: float = 1.6
    spike_core_px: float =10.0
    taper_frac: float = 0.2
    aperture_diam_mm: float = 300
    obstruction_frac: float = 0.30
    psf_size_px: float = 513

    # --- Grating params (for kind="grating") ---
    grating_model: Literal["analytic", "poppy"] = "poppy"
    lines_per_mm: float = 0.0        # e.g. 100 lines/mm => pitch 0.01 mm
    # analytic model knobs (legacy)
    order_max: int = 1
    order_rel_amp: float = 0.12
    # poppy amplitude grating knobs
    duty_cycle: float = 0.5          # open fraction (0..1)
    pupil_samples: int = 512         # sampling of pupil array
    n_lambda: int = 9                # polychromatic samples

    # --- Bitmap params (for kind="bitmap") ---
    bitmap_path: Optional[str] = None  # path to a pupil transmission bitmap (future)

    def is_enabled(self) -> bool:
        return self.kind != "none"