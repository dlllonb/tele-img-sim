from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Camera:
    nx: int              # pixels (width)
    ny: int              # pixels (height)
    pixel_um: float      # micron

    read_noise_e: float = 0.0
    gain_e_per_adu: float = 1.0
    qe: float = 1.0

    @property
    def sensor_x_mm(self) -> float:
        return self.nx * self.pixel_um * 1e-3

    @property
    def sensor_y_mm(self) -> float:
        return self.ny * self.pixel_um * 1e-3