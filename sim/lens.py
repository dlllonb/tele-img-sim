from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Lens:
    focal_mm: float
    f_number: float
    transmission: float = 1.0