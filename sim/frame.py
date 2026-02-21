import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from .camera import Camera
from .lens import Lens

@dataclass
class Frame:
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

        ra0 = math.radians(self.ra0_deg)
        dec0 = math.radians(self.dec0_deg)

        ra  = np.radians(ra_deg)
        dec = np.radians(dec_deg)

        # small-angle tangent plane (radians)
        x = (ra - ra0) * math.cos(dec0)
        y = (dec - dec0)

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


def display_frame(frame,
                  ax=None,
                  show_ra_dec=True,
                  title=None,
                  vmin=None,
                  vmax=None,
                  cmap=None,
                  annotate=False,
                  overlays=None):
    """
    Display a Frame with pixel axes + (optionally) RA/Dec axes.

    Parameters
    ----------
    frame : Frame
        Frame object with image + mapping metadata.
    ax : matplotlib.axes.Axes or None
        If None, creates a new figure+axes.
    show_ra_dec : bool
        If True, adds RA (top) and Dec (right) axes.
    title : str or None
        If None, auto-generates a title.
    vmin, vmax : float or None
        Passed through to imshow scaling.
    cmap : str or None
        Passed through to imshow. (Leave None for default.)
    annotate : bool
        Placeholder toggle for future annotations (stars, solve results, etc.)
    overlays : list or None
        Placeholder for future overlay objects (mask lines, star markers, etc.)

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    img = frame.image
    ny, nx = img.shape

    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest", aspect="auto",)
    ax.set_aspect("auto", adjustable="box")

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if title is None:
        ps_arcsec = frame.plate_scale_rad_per_px * (180.0 / math.pi) * 3600.0
        title = (f"{nx}×{ny}  |  FOV {frame.fov_x_deg:.3f}°×{frame.fov_y_deg:.3f}°  |  "
                 f"{ps_arcsec:.3f}\"/px  |  "
                 f"RA0={frame.ra0_deg:.6f}°  Dec0={frame.dec0_deg:.6f}°  Rot={frame.rot_deg:.3f}°")
    ax.set_title(title)

    # --- Hooks for future features ---
    if overlays is not None:
        # overlays could be callables or objects with a .draw(ax, frame) method.
        # For now, keep it permissive.
        for ov in overlays:
            if ov is None:
                continue
            if callable(ov):
                ov(ax, frame)
            elif hasattr(ov, "draw"):
                ov.draw(ax, frame)
            else:
                # unknown overlay type; ignore for now
                pass

    if annotate:
        # placeholder for star IDs, solve success text, sigma(angle), etc.
        pass

    # --- RA/Dec axes (small-angle tangent approximation) ---
    if show_ra_dec:
        # Use the precomputed x_rad/y_rad to infer axis extents.
        # We assume x_rad and y_rad are 2D grids with monotonic edges.
        x_min = float(frame.x_rad[0, 0])
        x_max = float(frame.x_rad[0, -1])
        y_min = float(frame.y_rad[0, 0])
        y_max = float(frame.y_rad[-1, 0])

        dec0 = math.radians(frame.dec0_deg)

        # Convert edge coordinates to degrees.
        # RA uses x / cos(dec0). Dec uses y directly.
        ra_min_deg = frame.ra0_deg + math.degrees(x_min / max(1e-12, math.cos(dec0)))
        ra_max_deg = frame.ra0_deg + math.degrees(x_max / max(1e-12, math.cos(dec0)))
        dec_min_deg = frame.dec0_deg + math.degrees(y_min)
        dec_max_deg = frame.dec0_deg + math.degrees(y_max)

        # Top axis: RA (deg)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        xt = np.asarray(ax.get_xticks())
        xt = xt[(xt >= 0) & (xt <= nx - 1)]
        if xt.size > 0:
            ra_ticks = ra_min_deg + (xt / (nx - 1)) * (ra_max_deg - ra_min_deg)
            ax_top.set_xticks(xt)
            ax_top.set_xticklabels([f"{v:.4f}" for v in ra_ticks])
        ax_top.set_xlabel("RA (deg)")

        # Right axis: Dec (deg)
        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        yt = np.asarray(ax.get_yticks())
        yt = yt[(yt >= 0) & (yt <= ny - 1)]
        if yt.size > 0:
            dec_ticks = dec_min_deg + (yt / (ny - 1)) * (dec_max_deg - dec_min_deg)
            ax_right.set_yticks(yt)
            ax_right.set_yticklabels([f"{v:.4f}" for v in dec_ticks])
        ax_right.set_ylabel("Dec (deg)")

    #return fig, ax




