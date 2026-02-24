import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from .camera import Camera
from .lens import Lens

@dataclass
class Frame:
    camera: Camera
    lens: Lens
    image: np.ndarray
    x_rad: np.ndarray
    y_rad: np.ndarray
    ra0_deg: float
    dec0_deg: float
    rot_deg: float
    plate_scale_rad_per_px: float
    fov_x_deg: float
    fov_y_deg: float

    def ra_dec_grids(self):
        """
        Returns RA and Dec grids (degrees) using small-angle tangent approximation.
        """
        ra0 = math.radians(self.ra0_deg)
        dec0 = math.radians(self.dec0_deg)

        dec = dec0 + self.y_rad
        ra  = ra0 + self.x_rad / math.cos(dec0)

        return np.degrees(ra), np.degrees(dec)

    
    def pixel_to_radec(self, x_px, y_px):
        """
        Small-angle pixel -> (RA, Dec) mapping.

        x_px, y_px can be scalars or numpy arrays (pixel coordinates).
        Returns (ra_deg, dec_deg) with same shape.
        """
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)

        ny, nx = self.image.shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

        # pixel -> tangent plane (radians)
        x = (x_px - cx) * self.plate_scale_rad_per_px
        y = (y_px - cy) * self.plate_scale_rad_per_px

        # undo rotation applied in make_blank_frame (because x_rad/y_rad were rotated)
        # make_blank_frame does: [x;y] = R(+theta) [x0;y0]
        # so here we do: [x0;y0] = R(-theta) [x;y]
        if self.rot_deg != 0.0:
            th = math.radians(self.rot_deg)
            c, s = math.cos(th), math.sin(th)
            x0 =  c * x + s * y
            y0 = -s * x + c * y
            x, y = x0, y0

        ra0 = math.radians(self.ra0_deg)
        dec0 = math.radians(self.dec0_deg)

        dec = dec0 + y
        ra  = ra0 + x / max(1e-12, math.cos(dec0))

        return np.degrees(ra), np.degrees(dec)

    
    def radec_to_pixel(self, ra_deg, dec_deg):
        """
        Small-angle (RA, Dec) -> pixel mapping.

        ra_deg, dec_deg can be scalars or numpy arrays.
        Returns (x_px, y_px) with same shape.
        """
        ra_deg = np.asarray(ra_deg, dtype=float)
        dec_deg = np.asarray(dec_deg, dtype=float)
    
        # wrap RA difference into [-180, 180] degrees to handle 0/360 boundary
        dra_deg = (ra_deg - self.ra0_deg + 180.0) % 360.0 - 180.0
    
        dec0 = math.radians(self.dec0_deg)
    
        x = np.radians(dra_deg) * math.cos(dec0)
        y = np.radians(dec_deg - self.dec0_deg)

        # apply rotation (to match make_blank_frame)
        if self.rot_deg != 0.0:
            th = math.radians(self.rot_deg)
            c, s = math.cos(th), math.sin(th)
            x_rot = c * x - s * y
            y_rot = s * x + c * y
            x, y = x_rot, y_rot

        ny, nx = self.image.shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

        x_px = cx + x / self.plate_scale_rad_per_px
        y_px = cy + y / self.plate_scale_rad_per_px

        return x_px, y_px

    


def make_blank_frame(camera: Camera,
                     lens: Lens,
                     ra0_deg: float = 0.0,
                     dec0_deg: float = 0.0,
                     rot_deg: float = 0.0,
                     dtype=np.float32) -> Frame:

    nx, ny = camera.nx, camera.ny
    img = np.zeros((ny, nx), dtype=dtype)

    ps_rad_per_px = (camera.pixel_um * 1e-6) / (lens.focal_mm * 1e-3)

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    j, i = np.indices((ny, nx))

    x = (i - cx) * ps_rad_per_px
    y = (j - cy) * ps_rad_per_px

    if rot_deg != 0.0:
        th = math.radians(rot_deg)
        c, s = math.cos(th), math.sin(th)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        x, y = x_rot, y_rot

    fov_x_deg = (camera.sensor_x_mm / lens.focal_mm) * (180.0 / math.pi)
    fov_y_deg = (camera.sensor_y_mm / lens.focal_mm) * (180.0 / math.pi)

    return Frame(
        camera=camera,
        lens=lens,
        image=img,
        x_rad=x,
        y_rad=y,
        ra0_deg=ra0_deg,
        dec0_deg=dec0_deg,
        rot_deg=rot_deg,
        plate_scale_rad_per_px=ps_rad_per_px,
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg
    )


def _display_transform(img,
                       *,
                       stretch="asinh",
                       vmin=None,
                       vmax=None,
                       q_lo=5.0,
                       q_hi=99.9,
                       eps=1e-12):
    """
    Apply the SAME display transform used by display_frame() to an image array.
    Returns (disp, dvmin, dvmax, vmin_used, vmax_used).
    """
    img = np.asarray(img)

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.percentile(finite, q_lo))
        hi = float(np.percentile(finite, q_hi))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = lo + 1.0
        if hi <= lo: hi = lo + 1.0

    if vmin is None:
        vmin = lo
    if vmax is None:
        vmax = hi

    x = img.astype(np.float32, copy=False)
    x = x - float(vmin)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.clip(x, 0.0, None)

    span = max(float(vmax) - float(vmin), eps)

    if stretch == "linear":
        disp = x
        dvmin, dvmax = 0.0, span
    elif stretch == "asinh":
        disp = np.arcsinh(x / span)
        dvmin, dvmax = 0.0, float(np.arcsinh(1.0))
    elif stretch == "log":
        disp = np.log10(1.0 + x / span)
        dvmin, dvmax = 0.0, float(np.log10(2.0))
    else:
        raise ValueError(f"Unknown stretch='{stretch}'")

    return disp.astype(np.float32, copy=False), dvmin, dvmax, float(vmin), float(vmax)


def display_frame(frame,
                  ax=None,
                  show_ra_dec=True,
                  title=None,
                  vmin=None,
                  vmax=None,
                  cmap="gray",
                  annotate=False,
                  overlays=None,
                  interpolation="nearest",
                  aspect="equal",
                  stretch="asinh",      # NEW: "linear"|"asinh"|"log"
                  q_lo=5.0,             # NEW: low percentile for pedestal
                  q_hi=99.9,            # NEW: high percentile for scale
                  eps=1e-12):           # NEW: stability
    """
    Display a Frame with pixel axes + (optionally) RA/Dec axes.

    stretch:
      - "linear": raw image with percentile-based vmin/vmax
      - "asinh": good default for astro images
      - "log": stronger compression than asinh
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7*1.5, 5*1.5))
    else:
        fig = ax.figure

    img = np.asarray(frame.image)
    ny, nx = img.shape

    # --- robust scaling on finite values ---
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.percentile(finite, q_lo))
        hi = float(np.percentile(finite, q_hi))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = lo + 1.0
        if hi <= lo: hi = lo + 1.0

    # If user didn’t pass vmin/vmax, use percentiles.
    if vmin is None:
        vmin = lo
    if vmax is None:
        vmax = hi

    # --- display transform (does NOT modify frame.image) ---
    x = img.astype(np.float32, copy=False)

    # pedestal-subtract so faint stuff is visible
    x = x - float(vmin)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.clip(x, 0.0, None)

    span = max(float(vmax) - float(vmin), eps)

    if stretch == "linear":
        disp = x
        dvmin, dvmax = 0.0, span

    elif stretch == "asinh":
        disp = np.arcsinh(x / span)
        dvmin, dvmax = 0.0, float(np.arcsinh(1.0))

    elif stretch == "log":
        disp = np.log10(1.0 + x / span)
        dvmin, dvmax = 0.0, float(np.log10(2.0))

    else:
        raise ValueError(f"Unknown stretch='{stretch}'")

    im = ax.imshow(
        disp,
        origin="lower",
        vmin=dvmin,
        vmax=dvmax,
        cmap=cmap,
        interpolation=interpolation,
    )

    ax.set_aspect(aspect, adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if title is None:
        ps_arcsec = frame.plate_scale_rad_per_px * (180.0 / math.pi) * 3600.0
        title = (f"{nx}×{ny}  |  FOV {frame.fov_x_deg:.3f}°×{frame.fov_y_deg:.3f}°  |  "
                 f"{ps_arcsec:.3f}\"/px  |  "
                 f"RA0={frame.ra0_deg:.6f}°  Dec0={frame.dec0_deg:.6f}°  Rot={frame.rot_deg:.3f}°")
    ax.set_title(title)

    # --- overlays hook ---
    if overlays is not None:
        for ov in overlays:
            if ov is None:
                continue
            if callable(ov):
                ov(ax, frame)
            elif hasattr(ov, "draw"):
                ov.draw(ax, frame)

    if annotate:
        pass

    # --- RA/Dec axes using secondary axes ---
    if show_ra_dec:
        x_min = float(frame.x_rad[0, 0])
        x_max = float(frame.x_rad[0, -1])
        y_min = float(frame.y_rad[0, 0])
        y_max = float(frame.y_rad[-1, 0])

        dec0 = math.radians(frame.dec0_deg)
        cosd = max(1e-12, math.cos(dec0))

        ra_min_deg  = frame.ra0_deg  + math.degrees(x_min / cosd)
        ra_max_deg  = frame.ra0_deg  + math.degrees(x_max / cosd)
        dec_min_deg = frame.dec0_deg + math.degrees(y_min)
        dec_max_deg = frame.dec0_deg + math.degrees(y_max)

        def px_to_ra(x_px):
            return ra_min_deg + (np.asarray(x_px) / (nx - 1)) * (ra_max_deg - ra_min_deg)

        def ra_to_px(ra_deg):
            return (np.asarray(ra_deg) - ra_min_deg) * (nx - 1) / (ra_max_deg - ra_min_deg)

        def px_to_dec(y_px):
            return dec_min_deg + (np.asarray(y_px) / (ny - 1)) * (dec_max_deg - dec_min_deg)

        def dec_to_px(dec_deg):
            return (np.asarray(dec_deg) - dec_min_deg) * (ny - 1) / (dec_max_deg - dec_min_deg)

        sax = ax.secondary_xaxis("top", functions=(px_to_ra, ra_to_px))
        sax.set_xlabel("RA (deg)")

        say = ax.secondary_yaxis("right", functions=(px_to_dec, dec_to_px))
        say.set_ylabel("Dec (deg)")
    
    return fig, ax, im


def find_star_peaks(image,
                    n=6,
                    q=99.95,
                    min_sep_px=40,
                    border_px=20):
    """
    Very simple bright-peak finder (no scipy):
      - pick candidate pixels above percentile q
      - sort by brightness
      - greedily accept peaks separated by min_sep_px
    Returns list of (x, y, value).
    """
    img = np.asarray(image)
    ny, nx = img.shape

    # Ignore borders (prevents ROI from clipping)
    y0, y1 = border_px, ny - border_px
    x0, x1 = border_px, nx - border_px
    if y1 <= y0 or x1 <= x0:
        return []

    sub = img[y0:y1, x0:x1]
    thresh = np.percentile(sub[np.isfinite(sub)], q)

    ys, xs = np.where(sub >= thresh)
    if xs.size == 0:
        return []

    vals = sub[ys, xs]
    order = np.argsort(vals)[::-1]

    peaks = []
    for idx in order:
        x = int(xs[idx] + x0)
        y = int(ys[idx] + y0)
        v = float(img[y, x])

        ok = True
        for (px, py, _) in peaks:
            if (x - px) ** 2 + (y - py) ** 2 < (min_sep_px ** 2):
                ok = False
                break
        if ok:
            peaks.append((x, y, v))
            if len(peaks) >= n:
                break

    return peaks

def plot_star_rois(frame,
                   image=None,
                   peaks=None,
                   n=6,
                   half_size=40,
                   q=99.95,
                   min_sep_px=60,
                   cmap="gray",
                   show_centroid=True,
                   show_title=True,
                   interpolation="nearest",
                   # --- NEW: match display_frame scaling exactly ---
                   stretch="asinh",
                   q_lo=5.0,
                   q_hi=99.9,
                   vmin=None,
                   vmax=None,
                   # --- Option: pick which image sets the scaling ---
                   shared_scale_image=None,
                   eps=1e-12):
    """
    Plot a 2x3 grid of ROIs around bright star peaks.

    Parameters
    ----------
    frame : Frame
    image : ndarray or None
        Image to display/crop. Defaults to frame.image.
    peaks : list[(x,y,val)] or None
        If None, auto-finds peaks in `image` via find_star_peaks().
    n : int
        Number of ROIs to show (max 6 is best).
    half_size : int
        ROI half-width in pixels (ROI is (2*half+1)^2).
    q : float
        Percentile threshold for peak candidate selection.
    min_sep_px : float
        Minimum separation between chosen peaks.
    stretch : "linear"|"asinh"|"log"
        Display stretch.
    shared_scale_image : ndarray or None
        If provided, all ROIs use a single pedestal/span derived from this image's percentiles.
        This makes the ROI grid visually consistent with the full-frame view.
    shared_q_lo, shared_q_hi : float
        Percentiles used for shared pedestal/span.
    """

    # --- helper: peak finding (expects you have find_star_peaks in frame.py; fallback included) ---
    def _fallback_find_star_peaks(img, n=6, q=99.95, min_sep_px=40, border_px=20):
        img = np.asarray(img)
        ny, nx = img.shape
        y0, y1 = border_px, ny - border_px
        x0, x1 = border_px, nx - border_px
        if y1 <= y0 or x1 <= x0:
            return []
        sub = img[y0:y1, x0:x1]
        fin = sub[np.isfinite(sub)]
        if fin.size == 0:
            return []
        thresh = np.percentile(fin, q)
        ys, xs = np.where(sub >= thresh)
        if xs.size == 0:
            return []
        vals = sub[ys, xs]
        order = np.argsort(vals)[::-1]
        peaks = []
        for idx in order:
            x = int(xs[idx] + x0)
            y = int(ys[idx] + y0)
            v = float(img[y, x])
            ok = True
            for (px, py, _) in peaks:
                if (x - px) ** 2 + (y - py) ** 2 < (min_sep_px ** 2):
                    ok = False
                    break
            if ok:
                peaks.append((x, y, v))
                if len(peaks) >= n:
                    break
        return peaks

    img = frame.image if image is None else np.asarray(image)

    # Use shared_scale_image to derive scaling if provided, otherwise use img.
    ref = img if shared_scale_image is None else np.asarray(shared_scale_image)

    disp_full, dvmin, dvmax, vmin_used, vmax_used = _display_transform(
        ref,
        stretch=stretch,
        vmin=vmin,
        vmax=vmax,
        q_lo=q_lo,
        q_hi=q_hi,
        eps=eps,
    )

    # IMPORTANT: if ref != img, we still need to transform img with the SAME vmin/vmax
    if shared_scale_image is not None:
        disp_img, _, _, _, _ = _display_transform(
            img,
            stretch=stretch,
            vmin=vmin_used,
            vmax=vmax_used,
            q_lo=q_lo,
            q_hi=q_hi,
            eps=eps,
        )
    else:
        disp_img = disp_full
    ny, nx = img.shape

    # Try to use your real find_star_peaks if it exists in the global namespace (frame.py import),
    # otherwise use fallback.
    _finder = globals().get("find_star_peaks", None)
    if _finder is None:
        _finder = _fallback_find_star_peaks

    if peaks is None:
        peaks = _finder(img, n=n, q=q, min_sep_px=min_sep_px, border_px=half_size + 2)

    m = min(len(peaks), n)
    if m == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("No peaks found")
        ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    # --- shared pedestal/span ---
    shared_lo = shared_hi = None
    if shared_scale_image is not None:
        ref = np.asarray(shared_scale_image)
        ref = ref[np.isfinite(ref)]
        if ref.size > 0:
            shared_lo = float(np.percentile(ref, shared_q_lo))
            shared_hi = float(np.percentile(ref, shared_q_hi))
            if not np.isfinite(shared_lo):
                shared_lo = 0.0
            if (not np.isfinite(shared_hi)) or (shared_hi <= shared_lo):
                shared_hi = shared_lo + 1.0

    # --- plot grid ---
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(12, 7), constrained_layout=True)
    axs = axs.ravel()

    for i in range(rows * cols):
        ax = axs[i]
        ax.set_xticks([]); ax.set_yticks([])

        if i >= m:
            ax.axis("off")
            continue

        x0, y0, v = peaks[i]
        x1, x2 = max(0, x0 - half_size), min(nx, x0 + half_size + 1)
        y1, y2 = max(0, y0 - half_size), min(ny, y0 + half_size + 1)

        roi_disp = disp_img[y1:y2, x1:x2]
        ax.imshow(roi_disp, origin="lower", cmap=cmap, interpolation=interpolation, vmin=dvmin, vmax=dvmax)

        if show_centroid:
            ax.plot([x0 - x1], [y0 - y1], marker="+", markersize=10)

        if show_title:
            ax.set_title(f"#{i+1}  (x={x0}, y={y0})  peak={v:.1f}")

    return fig


def save_frame_fits(path,
                    frame,
                    *,
                    image=None,
                    cfg=None,
                    res=None,
                    stars_csv=None,
                    stage="final_e",
                    overwrite=True):
    """
    Save a frame image (electrons) to a FITS file with simulation metadata in header.

    Parameters
    ----------
    path : str or Path
        Output FITS path.
    frame : Frame
        Frame object with camera/lens/pointing metadata.
    image : ndarray or None
        If provided, this array is saved. Otherwise:
          - if res and stage in res -> uses getattr(res, stage)
          - else uses frame.image
    cfg : RenderConfig or None
        If provided, includes render parameters (exposure, sky model, psf, toggles, seed, etc.)
    res : RenderResult or None
        If provided and image is None, can choose which stage to save via `stage`.
    stars_csv : str or None
        If provided, stored in header for provenance.
    stage : str
        Name of RenderResult field to save if image is None. Common: "final_e", "mean_e", "stars_e_post_psf".
    overwrite : bool
        Overwrite existing file.
    """
    from pathlib import Path
    import numpy as np

    # Astropy is the standard FITS writer; if it's not installed you'll get a clear error.
    from astropy.io import fits

    path = Path(path)

    # ---- pick image ----
    if image is None:
        if res is not None and hasattr(res, stage):
            image = getattr(res, stage)
        else:
            image = frame.image

    data = np.asarray(image, dtype=np.float32)

    hdr = fits.Header()

    # ---- basic provenance ----
    hdr["ORIGIN"] = "tele-img-sim"
    hdr["BUNIT"]  = "electron"
    hdr["STAGE"]  = (str(stage), "Pipeline stage saved")

    if stars_csv is not None:
        hdr["STARCAT"] = (str(stars_csv), "Star catalog source")

    # ---- frame/pointing ----
    hdr["RA0DEG"]  = (float(frame.ra0_deg), "Pointing RA0 [deg]")
    hdr["DEC0DEG"] = (float(frame.dec0_deg), "Pointing Dec0 [deg]")
    hdr["ROTDEG"]  = (float(frame.rot_deg), "Rotation [deg]")

    # plate scale + FOV
    import math
    ps_arcsec = float(frame.plate_scale_rad_per_px) * (180.0 / math.pi) * 3600.0
    hdr["PSARC"] = (ps_arcsec, "Plate scale [arcsec/px]")
    hdr["FOVX"]  = (float(frame.fov_x_deg), "FOV x [deg]")
    hdr["FOVY"]  = (float(frame.fov_y_deg), "FOV y [deg]")

    # ---- camera ----
    cam = frame.camera
    hdr["CAMNX"] = (int(cam.nx), "Camera width [px]")
    hdr["CAMNY"] = (int(cam.ny), "Camera height [px]")
    hdr["PIXLUM"] = (float(cam.pixel_um), "Pixel size [um]")
    hdr["RDNOISE"] = (float(cam.read_noise_e), "Read noise [e-]")
    hdr["GAIN"] = (float(cam.gain_e_per_adu), "Gain [e-/ADU]")
    hdr["QE"]   = (float(cam.qe), "Quantum efficiency (assumed)")

    # ---- lens / telescope ----
    lens = frame.lens
    hdr["FOCMM"] = (float(lens.focal_mm), "Focal length [mm]")
    hdr["FNUM"]  = (float(lens.f_number), "f-number")
    hdr["TRANSM"] = (float(lens.transmission), "Throughput (assumed)")

    # ---- cfg / simulation settings ----
    if cfg is not None:
        # core
        if hasattr(cfg, "exposure_s"): hdr["EXPTIME"] = (float(cfg.exposure_s), "Exposure [s]")
        if hasattr(cfg, "seed"):       hdr["SEED"]    = (int(cfg.seed), "RNG seed")

        # sky / photometry
        if hasattr(cfg, "sky_e_per_px_s"):         hdr["SKYEPXS"] = (float(cfg.sky_e_per_px_s), "Sky rate [e-/px/s] (0=>use mu)")
        if hasattr(cfg, "sky_mu_mag_per_arcsec2"): hdr["SKYMU"]   = (float(cfg.sky_mu_mag_per_arcsec2), "Sky brightness [mag/arcsec^2]")
        if hasattr(cfg, "zeropoint_e_per_s"):      hdr["ZP_EPS"]  = (float(cfg.zeropoint_e_per_s), "ZP [e-/s] for mag=0 (0=>derived/placeholder)")
        if hasattr(cfg, "lambda_eff_nm"):          hdr["LAMNM"]   = (float(cfg.lambda_eff_nm), "Effective wavelength [nm]")
        if hasattr(cfg, "band_nm"):                hdr["BANDNM"]  = (float(cfg.band_nm), "Bandpass width [nm]")

        # psf / jitter toggles
        if hasattr(cfg, "psf_sigma_px"):           hdr["PSFSIG"]  = (float(cfg.psf_sigma_px), "Gaussian PSF sigma [px]")
        if hasattr(cfg, "jitter_pointing_rms"):    hdr["JITRMS"]  = (float(cfg.jitter_pointing_rms), "Pointing jitter RMS [arcsec]")

        # stage toggles
        for k, key in [
            ("enable_sky",   "ENSKY"),
            ("enable_stars", "ENSTR"),
            ("enable_psf",   "ENPSF"),
            ("enable_jitter","ENJIT"),
            ("enable_noise", "ENNOIS"),
        ]:
            if hasattr(cfg, k):
                hdr[key] = (bool(getattr(cfg, k)), f"Toggle {k}")

        # ---- mask ----
        m = getattr(cfg, "mask", None)
        if m is not None:
            hdr["MASKKIND"] = (str(getattr(m, "kind", "none")), "Mask kind")
            if hasattr(m, "angle_deg"): hdr["MASKANG"] = (float(m.angle_deg), "Mask angle [deg]")

            # spider
            if hasattr(m, "n_vanes"):        hdr["SPVN"]   = (int(m.n_vanes), "Spider: number of vane lines")
            if hasattr(m, "vane_width_mm"):  hdr["SPWMM"]  = (float(m.vane_width_mm), "Spider: vane width [mm]")
            if hasattr(m, "spike_radius_px"):hdr["SPRAD"]  = (int(m.spike_radius_px), "Spider: kernel half-size [px]")
            if hasattr(m, "spike_rel_amp"):  hdr["SPAMP"]  = (float(m.spike_rel_amp), "Spider: relative spike amplitude")
            if hasattr(m, "spike_falloff"):  hdr["SPFALL"] = (float(m.spike_falloff), "Spider: along-spike falloff exponent")
            if hasattr(m, "spike_core_px"):  hdr["SPCORE"] = (float(m.spike_core_px), "Spider: along-spike core scale [px]")
            if hasattr(m, "taper_frac"):     hdr["SPTAPR"] = (float(m.taper_frac), "Spider: edge taper fraction")

            # grating
            if hasattr(m, "lines_per_mm"):   hdr["GRLPMM"] = (float(m.lines_per_mm), "Grating: lines per mm")
            if hasattr(m, "order_max"):      hdr["GROMAX"] = (int(m.order_max), "Grating: max order")
            if hasattr(m, "order_rel_amp"):  hdr["GROAMP"] = (float(m.order_rel_amp), "Grating: rel amp per order")

            # bitmap
            if hasattr(m, "bitmap_path") and getattr(m, "bitmap_path") is not None:
                hdr["MBITMAP"] = (str(m.bitmap_path), "Bitmap: pupil transmission path")

        # ---- long-form JSON dump (optional but useful) ----
        # FITS cards are short; store a compact dump in HISTORY lines.
        try:
            import json
            def _as_dict(obj):
                d = {}
                for k, v in getattr(obj, "__dict__", {}).items():
                    # make sure it's JSONable
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        d[k] = v
                    else:
                        d[k] = str(v)
                return d

            blob = {"cfg": _as_dict(cfg)}
            if getattr(cfg, "mask", None) is not None:
                blob["mask"] = _as_dict(cfg.mask)

            s = json.dumps(blob, separators=(",", ":"), sort_keys=True)
            # split into <=70-char chunks so HISTORY lines stay readable
            for i in range(0, len(s), 70):
                hdr.add_history(s[i:i+70])
        except Exception:
            pass

    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=overwrite)

    return str(path)