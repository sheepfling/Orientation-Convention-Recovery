from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from scripts.validation_noise import (
    sample_time_domain_dropout_mask,
    sample_time_domain_uniform_values,
)


def test_seeded_uniform_values_match_at_common_nested_cadence_times() -> None:
    at_40_hz = sample_time_domain_uniform_values(default_rng(8101), 161, 0.025, -0.002, 0.002)
    at_80_hz = sample_time_domain_uniform_values(default_rng(8101), 321, 0.0125, -0.002, 0.002)

    np.testing.assert_array_equal(at_40_hz, at_80_hz[::2])
    assert np.all(at_80_hz >= -0.002)
    assert np.all(at_80_hz <= 0.002)
####


def test_seeded_dropout_flags_match_at_common_nested_cadence_times() -> None:
    at_40_hz = sample_time_domain_dropout_mask(default_rng(8102), 161, 0.025, 0.1)
    at_80_hz = sample_time_domain_dropout_mask(default_rng(8102), 321, 0.0125, 0.1)

    np.testing.assert_array_equal(at_40_hz, at_80_hz[::2])
    assert np.any(~at_40_hz)
    assert np.any(at_40_hz)
####


def test_unsupported_sample_periods_retain_seeded_per_sample_draws() -> None:
    first_values = sample_time_domain_uniform_values(default_rng(8103), 9, 0.00725, -0.002, 0.002)
    repeated_values = sample_time_domain_uniform_values(
        default_rng(8103), 9, 0.00725, -0.002, 0.002
    )
    first_mask = sample_time_domain_dropout_mask(default_rng(8104), 9, 0.00725, 0.1)
    repeated_mask = sample_time_domain_dropout_mask(default_rng(8104), 9, 0.00725, 0.1)

    np.testing.assert_array_equal(first_values, repeated_values)
    np.testing.assert_array_equal(first_mask, repeated_mask)
####
