# sim/physics/psf.py
import numpy as np
from .masks import kernel_for_mask 

def _gaussian_kernel(sigma_px: float, radius: int | None = None) -> np.ndarray:
    sigma_px = float(sigma_px)
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        return np.array([[1.0]], dtype=np.float64)

    if radius is None:
        radius = int(np.ceil(4.0 * sigma_px))
    radius = max(1, int(radius))

    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    k = np.exp(-(x*x + y*y) / (2.0 * sigma_px * sigma_px))
    k /= np.sum(k)
    return k.astype(np.float64)

def _fft_convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    ny, nx = image.shape
    ky, kx = kernel.shape

    py = ny + ky - 1
    px = nx + kx - 1

    F = np.fft.rfft2(image, s=(py, px))
    K = np.fft.rfft2(kernel, s=(py, px))
    conv = np.fft.irfft2(F * K, s=(py, px))

    y0 = (ky - 1) // 2
    x0 = (kx - 1) // 2
    return conv[y0:y0+ny, x0:x0+nx]

def apply_psf(image_e, frame, cfg):
    """
    Apply optical PSF to an electron image (float).

    v1: Gaussian PSF plus optional diffraction mask (mask -> kernel dispatch).
    """
    sigma_px = float(getattr(cfg, "psf_sigma_px", 0.0))
    if sigma_px <= 0.0:
        return image_e

    mask = getattr(cfg, "mask", None)

    if (mask is None) or (getattr(mask, "kind", "none") == "none"):
        kernel = _gaussian_kernel(sigma_px)
    else:
        kernel = kernel_for_mask(frame, cfg, sigma_px, mask)

    return _fft_convolve_same(image_e, kernel).astype(np.float32, copy=False)