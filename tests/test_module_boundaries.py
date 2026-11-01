from __future__ import annotations

import attitude_convention_inference.core as core
import attitude_convention_inference.decoding as decoding
import attitude_convention_inference.integration as integration
import attitude_convention_inference.math_utils as math_utils
import attitude_convention_inference.pruning as pruning
import attitude_convention_inference.timing as timing
import attitude_convention_inference.trajectory as trajectory


def test_core_compatibility_facade_reexports_focused_implementations() -> None:
    compatibility_exports = {
        "FloatArray": math_utils.FloatArray,
        "IncrementWindowComposer": integration.IncrementWindowComposer,
        "IntegratedIntervals": integration.IntegratedIntervals,
        "IntervalDurationResolution": timing.IntervalDurationResolution,
        "ObservedIncrementCache": integration.ObservedIncrementCache,
        "QuaternionNormStatistics": trajectory.QuaternionNormStatistics,
        "SamplePeriodEstimate": timing.SamplePeriodEstimate,
        "TrajectoryData": trajectory.TrajectoryData,
        "WindowComposition": integration.WindowComposition,
        "compose_increment_windows": integration.compose_increment_windows,
        "decode_attitudes": decoding.decode_attitudes,
        "decode_rates": decoding.decode_rates,
        "infer_constant_sample_period": timing.infer_constant_sample_period,
        "integrate_interval_increments": integration.integrate_interval_increments,
        "observed_interval_increments": integration.observed_interval_increments,
        "observed_window_increments": integration.observed_window_increments,
        "quaternion_norm_statistics": trajectory.quaternion_norm_statistics,
        "refine_constant_sample_period": timing.refine_constant_sample_period,
        "resolve_interval_durations": timing.resolve_interval_durations,
        "should_prune_by_exact_lower_bound": pruning.should_prune_by_exact_lower_bound,
    }

    assert set(core.__all__) == set(compatibility_exports)
    for name, implementation in compatibility_exports.items():
        assert getattr(core, name) is implementation
    ####
####
