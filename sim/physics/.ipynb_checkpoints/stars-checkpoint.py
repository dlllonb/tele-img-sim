# sim/physics/stars.py
import numpy as np

def stars_layer(frame, stars, cfg, rng=None):
    """
    Produce a stars-only image layer in electrons (float), typically pre-PSF.

    Parameters
    ----------
    frame : Frame
    stars : object or None
        Placeholder for a StarField / catalog slice.
    cfg : RenderConfig
    rng : np.random.Generator or None

    Returns
    -------
    stars_e : np.ndarray
        Same shape as frame.image, float electrons.
    """
    # Stub: no stars yet
    return np.zeros_like(frame.image, dtype=np.float32)