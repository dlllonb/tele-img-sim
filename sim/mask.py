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
    n_vanes: int = 0                 # e.g. 2 for an "X" (two bars crossing), 4 for "+", etc.
    vane_width_mm: float = 0.0       # physical width of vane in mm (at pupil)
    transmission: float = 0.0        # 0 => opaque vanes; >0 => partially transmissive

    # --- Grating params (for kind="grating") ---
    lines_per_mm: float = 0.0        # e.g. 100 lines/mm => pitch 0.01 mm

    # --- Bitmap params (for kind="bitmap") ---
    bitmap_path: Optional[str] = None  # path to a pupil transmission bitmap (future)

    def is_enabled(self) -> bool:
        return self.kind != "none"