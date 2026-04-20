# measure/metrics.py
# compute performance metrics such as solve rate and angle uncertainties

from __future__ import annotations

from .types import (
    MeasurementMetadata,
    PlateSolveResult,
    SpikeMeasurementResult,
    MeasurementMetrics,
)


def compute_metrics(
    plate_res: PlateSolveResult,
    spike_res: SpikeMeasurementResult,
    meta: MeasurementMetadata,
) -> MeasurementMetrics:
    """Combine branch results into final science-facing metrics.

    The primary task of this stage is to translate an image-frame
    diffraction angle into a sky-frame orientation using the astrometric
    rotation recovered by the platesolver.  Even though both inputs are
    currently stubs, we set up the fields to enable that calculation.
    """
    metrics = MeasurementMetrics()
    metrics.solve_success = bool(plate_res.success)
    metrics.stripe_success = bool(spike_res.success)

    # copy raw values
    metrics.image_angle_deg = spike_res.image_angle_deg
    metrics.astrometric_rot_deg = plate_res.rot_deg

    if metrics.image_angle_deg is not None and metrics.astrometric_rot_deg is not None:
        # rot_deg = position angle of detector +Y axis East of North.
        # image_angle_deg = CCW from horizontal (+X) in image frame (origin='lower').
        # +X sky PA = rot_deg - 90°, so image angle α maps to:
        #   sky_angle = rot_deg - 90° + α  (East of North, normalised to [0, 360))
        raw = metrics.astrometric_rot_deg - 90.0 + metrics.image_angle_deg
        metrics.sky_angle_deg = float(raw % 360.0)
        metrics.final_sigma_deg = spike_res.sigma_angle_deg
        metrics.messages.append(
            f"sky_angle_deg={metrics.sky_angle_deg:.3f}° "
            f"(rot={metrics.astrometric_rot_deg:.3f}° + image={metrics.image_angle_deg:.3f}°)"
        )
    else:
        metrics.messages.append("could not compute sky_angle_deg; missing inputs")

    if not metrics.solve_success:
        metrics.quality_flags.append("platesolve_failed")
    if not metrics.stripe_success:
        metrics.quality_flags.append("stripe_failed")

    return metrics