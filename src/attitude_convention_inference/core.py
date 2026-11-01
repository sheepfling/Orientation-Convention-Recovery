"""Compatibility facade for trajectory, kinematics, integration, and timing APIs.

New implementation code should import from the focused modules directly. This
module re-exports the historical names so existing callers continue to work.
"""

from __future__ import annotations

from .decoding import decode_attitudes, decode_rates
from .integration import (
    IncrementWindowComposer,
    IntegratedIntervals,
    ObservedIncrementCache,
    WindowComposition,
    compose_increment_windows,
    integrate_interval_increments,
    observed_interval_increments,
    observed_window_increments,
)
from .math_utils import FloatArray
from .pruning import should_prune_by_exact_lower_bound
from .timing import (
    IntervalDurationResolution,
    SamplePeriodEstimate,
    infer_constant_sample_period,
    refine_constant_sample_period,
    resolve_interval_durations,
)
from .trajectory import QuaternionNormStatistics, TrajectoryData, quaternion_norm_statistics

__all__ = [
    "FloatArray",
    "IncrementWindowComposer",
    "IntegratedIntervals",
    "IntervalDurationResolution",
    "ObservedIncrementCache",
    "QuaternionNormStatistics",
    "SamplePeriodEstimate",
    "TrajectoryData",
    "WindowComposition",
    "compose_increment_windows",
    "decode_attitudes",
    "decode_rates",
    "infer_constant_sample_period",
    "integrate_interval_increments",
    "observed_interval_increments",
    "observed_window_increments",
    "quaternion_norm_statistics",
    "refine_constant_sample_period",
    "resolve_interval_durations",
    "should_prune_by_exact_lower_bound",
]
