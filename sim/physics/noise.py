# sim/physics/noise.py
import numpy as np

def apply_noise(image_e, frame, cfg, rng=None):
    """
    Apply shot + read noise.

    image_e is a float electron expectation image.
    Returns a float image representing a realized noisy frame.
    """
    # Stub: no noise yet
    return image_e