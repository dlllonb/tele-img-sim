# measure/metrics.py
# compute performance metrics such as solve rate and angle uncertainties

from __future__ import annotations

from .types import (
    MeasurementInput,
    StarDetectionResult,
    PlateSolveResult,
    SpikeMeasurementResult,
    MeasurementMetrics,
)


def compute_measurement_metrics(
    inp: MeasurementInput,
    star_res: StarDetectionResult,
    plate_res: PlateSolveResult,
    spike_res: SpikeMeasurementResult,
) -> MeasurementMetrics:
    """Stub metrics computation.

    Real version will combine uncertainty estimates from each stage and
    produce final polarization angle uncertainty, success rates, etc.
    """
    metrics = MeasurementMetrics()
    metrics.solve_success = bool(plate_res.success)
    metrics.spike_success = bool(spike_res.success)
    # placeholder: propagate spike angle if available
    metrics.final_angle_deg = spike_res.angle_deg
    metrics.final_sigma_deg = spike_res.sigma_angle_deg
    if not metrics.solve_success:
        metrics.quality_flags.append("platesolve_failed")
    if not metrics.spike_success:
        metrics.quality_flags.append("spike_failed")
    return metrics