"""First-order colored-noise reference metrics for synthetic validation only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Protocol, cast

from numpy import arange, concatenate, float64, square, zeros
from numpy import min as np_min
from numpy import sum as np_sum
from numpy.typing import NDArray
from scipy.linalg import (
    solve_triangular as scipy_solve_triangular,  # pyright: ignore[reportUnknownVariableType]
)

FloatArray = NDArray[float64]


class _TriangularSolver(Protocol):
    def __call__(
        self,
        factor: FloatArray,
        right_hand_side: FloatArray,
        *,
        lower: bool,
    ) -> FloatArray: ...
####


_solve_triangular = cast(_TriangularSolver, scipy_solve_triangular)

__all__ = [
    "IntervalResidualNoiseModel",
    "PairedLogLikelihoodRatios",
    "covariance_adjusted_log_likelihood_ratio",
    "covariance_weighted_sum_squares",
    "paired_log_likelihood_ratios_by_case",
    "solve_lower_triangular",
]


@dataclass(frozen=True, slots=True)
class IntervalResidualNoiseModel:
    """Small-angle interval-error covariance from noisy attitudes and trapezoid rates.

    The model assumes uniform, synchronized samples, independent isotropic
    orientation/rate measurement channels, and small rotations between samples.
    Orientation noise is differenced; rate noise is trapezoid-integrated.
    """

    orientation_sigma_rad: float
    orientation_ar1_coefficient: float
    rate_sigma_rad_s: float
    rate_ar1_coefficient: float
    sample_period_s: float

    def __post_init__(self) -> None:
        for name, sigma in (
            ("orientation_sigma_rad", self.orientation_sigma_rad),
            ("rate_sigma_rad_s", self.rate_sigma_rad_s),
        ):
            if not isfinite(sigma) or sigma < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            ####
        ####
        for name, coefficient in (
            ("orientation_ar1_coefficient", self.orientation_ar1_coefficient),
            ("rate_ar1_coefficient", self.rate_ar1_coefficient),
        ):
            if not isfinite(coefficient) or abs(coefficient) >= 1.0:
                raise ValueError(f"{name} must be strictly between -1 and 1.")
            ####
        ####
        if not isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive and finite.")
        ####
        if self.orientation_sigma_rad == 0.0 and self.rate_sigma_rad_s == 0.0:
            raise ValueError("At least one measurement-noise sigma must be positive.")
        ####
    ####

    def lag_covariance_rad2(self, lag: int) -> float:
        """Return one temporal covariance entry for a 3-axis interval residual."""

        if lag < 0:
            raise ValueError("lag must be non-negative.")
        ####
        orientation_variance = self.orientation_sigma_rad**2
        rate_variance = self.rate_sigma_rad_s**2
        half_period = self.sample_period_s / 2.0
        if lag == 0:
            orientation_covariance = (
                2.0 * orientation_variance * (1.0 - self.orientation_ar1_coefficient)
            )
            rate_covariance = (
                2.0 * half_period**2 * rate_variance * (1.0 + self.rate_ar1_coefficient)
            )
        else:
            orientation_covariance = (
                -orientation_variance
                * (1.0 - self.orientation_ar1_coefficient) ** 2
                * self.orientation_ar1_coefficient ** (lag - 1)
            )
            rate_covariance = (
                half_period**2
                * rate_variance
                * (1.0 + self.rate_ar1_coefficient) ** 2
                * self.rate_ar1_coefficient ** (lag - 1)
            )
        ####
        return orientation_covariance + rate_covariance
    ####

    def covariance_matrix(self, interval_count: int) -> FloatArray:
        """Build the shared temporal covariance matrix for interval residual axes."""

        if interval_count < 1:
            raise ValueError("interval_count must be positive.")
        ####
        covariance = zeros((interval_count, interval_count), dtype=float64)
        for lag in range(interval_count):
            indices = arange(interval_count - lag)
            lag_value = self.lag_covariance_rad2(lag)
            covariance[indices, indices + lag] = lag_value
            covariance[indices + lag, indices] = lag_value
        ####
        return covariance
    ####
####


@dataclass(frozen=True, slots=True)
class PairedLogLikelihoodRatios:
    """Per-case evidence scores; positive values favor the proposed winner."""

    marginal_iid_by_case: FloatArray
    temporal_covariance_by_case: FloatArray
####


def solve_lower_triangular(factor: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    """Solve a lower-triangular system without refactorizing its Cholesky factor."""

    return _solve_triangular(factor, right_hand_side, lower=True)
####


def covariance_weighted_sum_squares(
    interval_errors: FloatArray,
    covariance_cholesky: FloatArray,
) -> float:
    """Evaluate ``trace(E.T @ covariance^-1 @ E)`` using a lower Cholesky factor."""

    if interval_errors.ndim != 2 or interval_errors.shape[1] != 3:
        raise ValueError("interval_errors must have shape (N, 3).")
    ####
    if interval_errors.shape[0] == 0:
        raise ValueError("interval_errors must contain at least one interval.")
    ####
    expected_shape = (interval_errors.shape[0], interval_errors.shape[0])
    if covariance_cholesky.shape != expected_shape:
        raise ValueError("covariance_cholesky must be square and match the interval count.")
    ####
    whitened = solve_lower_triangular(covariance_cholesky, interval_errors)
    return float(np_sum(square(whitened)))
####


def covariance_adjusted_log_likelihood_ratio(
    winner_errors: FloatArray,
    competitor_errors: Sequence[FloatArray],
    covariance_cholesky: FloatArray,
) -> float:
    """Compare a winner with its best competitor under one shared temporal covariance."""

    if not competitor_errors:
        raise ValueError("At least one competitor is required.")
    ####
    expected_shape = winner_errors.shape
    if winner_errors.ndim != 2 or winner_errors.shape[1] != 3 or winner_errors.shape[0] == 0:
        raise ValueError("winner_errors must have shape (N, 3) with N > 0.")
    ####
    if any(errors.shape != expected_shape for errors in competitor_errors):
        raise ValueError("Competitor residuals must match winner_errors shape.")
    ####
    expected_factor_shape = (winner_errors.shape[0], winner_errors.shape[0])
    if covariance_cholesky.shape != expected_factor_shape:
        raise ValueError("covariance_cholesky must match the residual interval count.")
    ####
    all_errors = concatenate((winner_errors, *competitor_errors), axis=1)
    whitened = solve_lower_triangular(covariance_cholesky, all_errors)
    quadratics = np_sum(
        square(whitened.reshape(winner_errors.shape[0], len(competitor_errors) + 1, 3)),
        axis=(0, 2),
    )
    return 0.5 * (float(np_min(quadratics[1:])) - float(quadratics[0]))
####


def paired_log_likelihood_ratios_by_case(
    winner_errors: FloatArray,
    competitor_errors: FloatArray,
    *,
    marginal_variance_rad2: float,
    covariance_cholesky: FloatArray,
    alternative_marginal_variance_rad2: float | None = None,
    alternative_covariance_cholesky: FloatArray | None = None,
) -> PairedLogLikelihoodRatios:
    """Compare paired residuals under truth and alternative Gaussian noise models.

    Each model is isotropic across the three axes. Its i.i.d. marginal variance and
    temporal covariance may differ. The temporal covariance is represented by its
    lower Cholesky factor. Positive scores favor ``winner_errors``; omitted
    alternative parameters reuse the truth model.
    """

    if (
        winner_errors.ndim != 3
        or winner_errors.shape[0] == 0
        or winner_errors.shape[1] == 0
        or winner_errors.shape[2] != 3
    ):
        raise ValueError("winner_errors must have shape (case_count, N, 3), with case_count,N > 0.")
    ####
    if competitor_errors.shape != winner_errors.shape:
        raise ValueError("competitor_errors must have the same shape as winner_errors.")
    ####
    if not isfinite(marginal_variance_rad2) or marginal_variance_rad2 <= 0.0:
        raise ValueError("marginal_variance_rad2 must be positive and finite.")
    ####
    alternative_variance = (
        marginal_variance_rad2
        if alternative_marginal_variance_rad2 is None
        else alternative_marginal_variance_rad2
    )
    if not isfinite(alternative_variance) or alternative_variance <= 0.0:
        raise ValueError("alternative_marginal_variance_rad2 must be positive and finite.")
    ####
    case_count, interval_count, _ = winner_errors.shape
    expected_factor_shape = (interval_count, interval_count)
    if covariance_cholesky.shape != expected_factor_shape:
        raise ValueError("covariance_cholesky must match the interval count.")
    ####
    alternative_factor = (
        covariance_cholesky
        if alternative_covariance_cholesky is None
        else alternative_covariance_cholesky
    )
    if alternative_factor.shape != expected_factor_shape:
        raise ValueError("alternative_covariance_cholesky must match the interval count.")
    ####
    truth_factor_diagonal = covariance_cholesky.diagonal()
    alternative_factor_diagonal = alternative_factor.diagonal()
    if any(
        not isfinite(float(value)) or value <= 0.0
        for value in (*truth_factor_diagonal, *alternative_factor_diagonal)
    ):
        raise ValueError("Covariance Cholesky factors must have positive finite diagonals.")
    ####
    truth_sum_squares = np_sum(square(winner_errors), axis=(1, 2))
    alternative_sum_squares = np_sum(square(competitor_errors), axis=(1, 2))
    axis_count = winner_errors.shape[2]
    iid_log_determinant_difference = (
        axis_count * interval_count * log(alternative_variance / marginal_variance_rad2)
    )
    marginal_iid_scores = 0.5 * (
        iid_log_determinant_difference
        + alternative_sum_squares / alternative_variance
        - truth_sum_squares / marginal_variance_rad2
    )
    truth_right_hand_side = winner_errors.transpose(1, 0, 2).reshape(
        interval_count,
        case_count * axis_count,
    )
    alternative_right_hand_side = competitor_errors.transpose(1, 0, 2).reshape(
        interval_count,
        case_count * axis_count,
    )
    truth_whitened = solve_lower_triangular(
        covariance_cholesky,
        truth_right_hand_side,
    ).reshape(
        interval_count,
        case_count,
        axis_count,
    )
    alternative_whitened = solve_lower_triangular(
        alternative_factor,
        alternative_right_hand_side,
    ).reshape(
        interval_count,
        case_count,
        axis_count,
    )
    truth_quadratics = np_sum(square(truth_whitened), axis=(0, 2))
    alternative_quadratics = np_sum(square(alternative_whitened), axis=(0, 2))
    truth_time_log_determinant = 2.0 * sum(log(float(value)) for value in truth_factor_diagonal)
    alternative_time_log_determinant = 2.0 * sum(
        log(float(value)) for value in alternative_factor_diagonal
    )
    colored_log_determinant_difference = axis_count * (
        alternative_time_log_determinant - truth_time_log_determinant
    )
    temporal_scores = 0.5 * (
        colored_log_determinant_difference + alternative_quadratics - truth_quadratics
    )
    return PairedLogLikelihoodRatios(
        marginal_iid_by_case=marginal_iid_scores,
        temporal_covariance_by_case=temporal_scores,
    )
####
