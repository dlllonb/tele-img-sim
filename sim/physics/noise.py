# sim/physics/noise.py
import numpy as np

def apply_noise(image_e, frame, cfg, rng=None):
    """
    Apply shot (Poisson) + read (Gaussian) noise.

    Parameters
    ----------
    image_e : ndarray
        Expected electrons per pixel (float).
    frame : Frame
        Frame metadata (unused for now).
    cfg : RenderConfig
        Must contain read_noise_e.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    noisy_e : ndarray (float32)
        Realized noisy image in electrons.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure nonnegative before Poisson
    lam = np.clip(image_e, 0.0, None)

    # Shot noise (Poisson)
    shot = rng.poisson(lam).astype(np.float32)

    # Read noise (Gaussian)
    read_sigma = float(getattr(cfg, "read_noise_e", 0.0))
    if read_sigma > 0.0:
        read = rng.normal(loc=0.0,
                          scale=read_sigma,
                          size=shot.shape).astype(np.float32)
        noisy = shot + read
    else:
        noisy = shot

    return noisy