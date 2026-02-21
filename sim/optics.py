from dataclasses import dataclass
import math
from .camera import Camera
from .lens import Lens

def plate_scale_arcsec_per_px(camera: Camera, lens: Lens) -> float:
    # rad/px = pixel_size / focal_length
    rad_per_px = (camera.pixel_um * 1e-6) / (lens.focal_mm * 1e-3)
    return rad_per_px * (180.0 / math.pi) * 3600.0

def fov_deg(camera: Camera, lens: Lens) -> tuple[float, float]:
    # small-angle FOV ≈ sensor_size / focal_length (radians)
    fov_x_rad = camera.sensor_x_mm / lens.focal_mm
    fov_y_rad = camera.sensor_y_mm / lens.focal_mm
    return (fov_x_rad * 180.0 / math.pi, fov_y_rad * 180.0 / math.pi)

def print_fov(camera: Camera, lens: Lens) -> None:
    fx, fy = fov_deg(camera, lens)
    ps = plate_scale_arcsec_per_px(camera, lens)
    print(f"Sensor: {camera.sensor_x_mm:.3f} mm x {camera.sensor_y_mm:.3f} mm")
    print(f"FOV:    {fx:.3f} deg x {fy:.3f} deg")
    print(f"Scale:  {ps:.3f} arcsec / px")