# sim/physics/jitter.py
import numpy as np

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

def apply_jitter(image_e, frame, cfg):
    """
    Apply pointing jitter as an additional Gaussian blur.

    cfg.jitter_pointing_rms is interpreted as RMS pointing jitter in arcseconds.
    Converted to pixels using frame.plate_scale_rad_per_px.
    """
    # RMS pointing jitter [arcsec]
    rms_arcsec = float(getattr(cfg, "jitter_pointing_rms", 0.0))
    if rms_arcsec <= 0.0:
        return image_e

    # plate scale [arcsec/px]
    ps_arcsec_per_px = float(frame.plate_scale_rad_per_px) * (180.0 / np.pi) * 3600.0
    if not np.isfinite(ps_arcsec_per_px) or ps_arcsec_per_px <= 0:
        return image_e

    # Convert RMS arcsec -> sigma in pixels (Gaussian blur sigma)
    sigma_px = rms_arcsec / ps_arcsec_per_px
    if sigma_px <= 0.0:
        return image_e

    #print(f"jitter: {rms_arcsec:.3f}\" RMS => {sigma_px:.3f} px (plate={ps_arcsec_per_px:.3f}\"/px)")
    kernel = _gaussian_kernel(sigma_px)
    return _fft_convolve_same(image_e, kernel).astype(np.float32, copy=False)