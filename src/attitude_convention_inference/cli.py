"""Command-line interface for trajectory inspection and inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from .api import infer_conventions, resolve_config
from .inference import best_interval_diagnostics
from .io import (
    CsvLayout,
    detect_csv_layout,
    load_config,
    load_csv_trajectory,
    write_interval_diagnostics,
    write_report,
)
from .models import (
    AngleUnit,
    AttitudeDirection,
    AttitudeRepresentation,
    ConventionConstraints,
    EulerMode,
    InferenceDecision,
    PreferenceProfile,
    QuaternionOrder,
    RateFrame,
    RateUnit,
)
from .profiles import available_preference_profiles, load_preference_profile
from .version import __version__

__all__ = ["main"]


class _CsvLayoutPayload(TypedDict):
    """JSON fields printed by ``--inspect --format json``."""

    representation: str
    orientation_columns: list[str]
    rate_columns: list[str]
    time_column: str | None
    available_columns: list[str]
####


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attitude-conventions",
        description=(
            "Rank Euler/quaternion and angular-rate convention hypotheses by integrated "
            "trajectory consistency. Common CSV column names are detected automatically; "
            "explicit columns always override detection."
        ),
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        help="Input CSV trajectory; omit it when listing or inspecting profiles.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--representation",
        choices=[value.value for value in AttitudeRepresentation],
        help="Quaternion or Euler input. Inferred from conventional columns when omitted.",
    )
    parser.add_argument(
        "--orientation-columns",
        nargs="+",
        metavar="COLUMN",
        help="Three Euler or four quaternion columns. Conventional names are auto-detected.",
    )
    parser.add_argument(
        "--rate-columns",
        nargs=3,
        metavar=("RATE0", "RATE1", "RATE2"),
        help="Three angular-rate columns. Conventional names are auto-detected.",
    )
    parser.add_argument(
        "--time-column",
        help="Timestamp column. Common names such as time, time_s, and t are auto-detected.",
    )
    parser.add_argument(
        "--ignore-time-column",
        action="store_true",
        help="Do not auto-detect or use timestamps; use or infer a constant sample period.",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        help="Known constant sample period in seconds when timestamps are unavailable.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional JSON InferenceConfig containing hard constraints, soft priors, "
            "and a persistent preference profile."
        ),
    )
    parser.add_argument(
        "--profile",
        metavar="PROFILE",
        help="Packaged preference profile to apply after loading --config.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List preference profiles bundled with this installation and exit.",
    )
    parser.add_argument(
        "--inspect-profile",
        metavar="PROFILE",
        help="Print one packaged preference profile as JSON and exit.",
    )
    parser.add_argument(
        "--attitude-direction",
        nargs="+",
        choices=[value.value for value in AttitudeDirection],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--quaternion-order",
        nargs="+",
        choices=[value.value for value in QuaternionOrder],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--euler-order",
        nargs="+",
        choices=(
            "xyz",
            "xzy",
            "yxz",
            "yzx",
            "zxy",
            "zyx",
            "xyx",
            "xzx",
            "yxy",
            "yzy",
            "zxz",
            "zyz",
        ),
        help="One known Tait-Bryan or proper-Euler sequence or an allowed subset.",
    )
    parser.add_argument(
        "--euler-mode",
        nargs="+",
        choices=[value.value for value in EulerMode],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--angle-unit",
        nargs="+",
        choices=[value.value for value in AngleUnit],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--rate-frame",
        nargs="+",
        choices=[value.value for value in RateFrame],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--rate-order",
        nargs="+",
        choices=("pqr", "prq", "qpr", "qrp", "rpq", "rqp"),
        help="One known p/q/r order or an allowed subset.",
    )
    parser.add_argument(
        "--rate-unit",
        nargs="+",
        choices=[value.value for value in RateUnit],
        help="One known value or an allowed subset.",
    )
    parser.add_argument(
        "--rate-sign",
        nargs="+",
        type=int,
        choices=(-1, 1),
        help="One known sign or both allowed signs.",
    )
    parser.add_argument(
        "--format",
        "--output-format",
        dest="output_format",
        choices=("text", "json", "decision-json"),
        default="text",
        help="Text summary, complete JSON report, or evidence-gated decision JSON.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Resolve and print the input layout without running inference.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--details-csv", type=Path)
    parser.add_argument("--top", type=int, default=10)
    return parser
####


def _cli_constraints(arguments: argparse.Namespace) -> ConventionConstraints:
    return ConventionConstraints(
        attitude_direction=arguments.attitude_direction,
        quaternion_order=arguments.quaternion_order,
        euler_order=arguments.euler_order,
        euler_mode=arguments.euler_mode,
        angle_unit=arguments.angle_unit,
        rate_frame=arguments.rate_frame,
        rate_order=arguments.rate_order,
        rate_unit=arguments.rate_unit,
        rate_sign=arguments.rate_sign,
    )
####


def _layout_payload(layout: CsvLayout) -> _CsvLayoutPayload:
    return {
        "representation": layout.representation.value,
        "orientation_columns": list(layout.orientation_columns),
        "rate_columns": list(layout.rate_columns),
        "time_column": layout.time_column,
        "available_columns": list(layout.available_columns),
    }
####


def _print_layout(layout: CsvLayout) -> None:
    print("Input layout")
    print(f"  representation: {layout.representation.value}")
    print(f"  orientation:    {', '.join(layout.orientation_columns)}")
    print(f"  rates:          {', '.join(layout.rate_columns)}")
    print(f"  time:           {layout.time_column or '<none>'}")
####


def _print_profile_catalog(profiles: Sequence[PreferenceProfile]) -> None:
    print("Packaged preference profiles")
    for profile in profiles:
        status = "configured" if profile.configured() else "neutral"
        print(f"- {profile.profile_id} [{status}] source={profile.source or '<unspecified>'}")
        if profile.context:
            print(f"  context={json.dumps(dict(profile.context), sort_keys=True)}")
        ####
    ####
####


def _print_report(report_json: str, top: int, decision: InferenceDecision) -> None:
    payload = json.loads(report_json)
    best = payload["best"]
    diagnostics = payload["diagnostics"]
    space = payload["hypothesis_space"]

    print(
        f"result_label={payload.get('result_label', 'trajectory-quotient')}  "
        f"operational_reason={payload.get('operational_reason', 'trajectory')}  "
        "representative_reason="
        f"{payload.get('representative_selection_reason', 'canonical-policy')}"
    )
    print(
        f"operational_class={payload.get('operational_class_id', '')}  "
        f"operational_hypothesis={payload.get('operational_hypothesis_id', '')}"
    )
    print(
        f"decision={decision.status.value}  reason={decision.reason.value}  "
        f"recommended_class={decision.recommended_class_id or '<none>'}"
    )
    print(
        "evidence_contenders="
        + json.dumps(payload.get("evidence_contender_class_ids", []))
        + f"  domain_prior_abstained={payload.get('domain_prior_abstained', False)}"
        + f"  preference_profile={payload.get('preference_profile_id', '') or '<none>'}"
    )

    print(
        "Hypothesis space "
        f"{space['base_hypothesis_count']} -> {space['constrained_hypothesis_count']} "
        f"({100.0 * space['reduction_fraction']:.1f}% reduction)"
    )
    if space["fixed_dimensions"]:
        print("Fixed dimensions")
        print(json.dumps(space["fixed_dimensions"], indent=2))
    ####
    if space["relative_prior_weights"]:
        print("Relative prior weights")
        print(json.dumps(space["relative_prior_weights"], indent=2))
    ####

    print("Best hypothesis")
    print(json.dumps(best["hypothesis"], indent=2))
    print(
        f"raw_risk={best['raw_geometric_risk']:.6g} rad^2  "
        f"untrimmed_risk={best['untrimmed_geometric_risk']:.6g} rad^2  "
        f"fit={best['fit_score']:.2f}/100  "
        f"interval_rmse={best['interval_rmse_rad']:.6g} rad  "
        f"posterior_support={best['relative_support']:.6f}  "
        f"data_support={best['data_relative_support']:.6f}"
    )
    if best["inferred_sample_period_s"] is not None:
        print(f"inferred sample period={best['inferred_sample_period_s']:.9g} s")
    ####
    heuristic = best.get("heuristic_features", {})
    if heuristic.get("wrap_unit_status") not in (None, "not-applicable"):
        print(
            f"wrap_unit_status={heuristic['wrap_unit_status']}  "
            f"wrap_gain={heuristic['wrap_gain_raw']:.6g}  "
            f"wrap_count={heuristic['inferred_wrap_count']}"
        )
    ####
    if len(best["exact_equivalent_hypothesis_ids"]) > 1:
        print("Exact equivalence class")
        for identifier in best["exact_equivalent_hypothesis_ids"]:
            print(f"- {identifier}")
        ####
    ####
    print(
        "heuristic confidence score="
        f"{diagnostics['confidence_label']} ({diagnostics['confidence']:.3f})"
    )
    print(
        f"noise model={diagnostics['noise_model']}  "
        f"sigma={diagnostics['estimated_noise_sigma_rad']:.6g} rad  "
        f"lag1={diagnostics['noise_lag1_autocorrelation']:.4f}"
    )
    print(
        f"evidence sufficient={diagnostics['evidence_sufficient']}  "
        f"intervals={diagnostics['evidence_intervals_used']} "
        f"({100.0 * diagnostics['evidence_fraction']:.1f}%)  "
        f"log-likelihood-ratio={diagnostics['evidence_log_likelihood_ratio']:.6g}"
    )
    print(
        f"raw excitation rank={diagnostics.get('raw_excitation_rank', 0)}  "
        f"weakest-axis-ratio={diagnostics.get('raw_excitation_condition_ratio', 0.0):.6g}"
    )
    if diagnostics["unresolved_dimensions"]:
        print("Unresolved from trajectory evidence")
        print(json.dumps(diagnostics["unresolved_dimensions"], indent=2))
    ####
    if diagnostics["posterior_unresolved_dimensions"]:
        print("Unresolved after soft priors")
        print(json.dumps(diagnostics["posterior_unresolved_dimensions"], indent=2))
    ####
    if diagnostics["warnings"]:
        print("Warnings")
        for warning in diagnostics["warnings"]:
            print(f"- {warning}")
        ####
    ####

    print("\nRanking")
    for candidate in payload["rankings"][:top]:
        hypothesis = candidate["hypothesis"]
        attitude = (
            f"quat/{hypothesis['quaternion_order']}"
            if hypothesis["quaternion_order"] is not None
            else (
                f"euler/{hypothesis['euler_mode']}/{hypothesis['euler_order']}/"
                f"{hypothesis['angle_unit']}"
            )
        )
        print(
            f"{candidate['rank']:>3}  data-rank={candidate['data_rank']:>3}  "
            f"risk={candidate['raw_geometric_risk']:.6g}  "
            f"score={candidate['fit_score']:>7.2f}  "
            f"support={candidate['relative_support']:.5f}  "
            f"prior={candidate['prior_relative_weight']:.5f}  "
            f"rmse={candidate['interval_rmse_rad']:.6g}  "
            f"{attitude}/{hypothesis['attitude_direction']}  "
            f"rate={hypothesis['rate_frame']}/{hypothesis['rate_order']}/"
            f"{hypothesis['rate_unit']}/sign={hypothesis['rate_sign']:+d}"
        )
    ####
####


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.output_format == "decision-json" and (
            arguments.list_profiles or arguments.inspect_profile is not None or arguments.inspect
        ):
            raise ValueError("--format decision-json is available only for trajectory inference.")
        ####
        if arguments.list_profiles and arguments.inspect_profile is not None:
            raise ValueError("--list-profiles and --inspect-profile are mutually exclusive.")
        ####
        if arguments.list_profiles or arguments.inspect_profile is not None:
            if arguments.csv is not None:
                raise ValueError("Profile inspection commands do not accept a CSV path.")
            ####
            if arguments.list_profiles:
                profiles = tuple(
                    load_preference_profile(profile_id)
                    for profile_id in available_preference_profiles()
                )
                if arguments.output_format == "json":
                    print(
                        json.dumps(
                            [profile.model_dump(mode="json") for profile in profiles],
                            indent=2,
                        )
                    )
                else:
                    _print_profile_catalog(profiles)
                ####
                return 0
            ####
            inspect_profile = arguments.inspect_profile
            if not isinstance(inspect_profile, str):
                raise ValueError("--inspect-profile requires a profile ID.")
            ####
            profile = load_preference_profile(inspect_profile)
            if arguments.output_format == "json":
                print(profile.model_dump_json(indent=2))
            else:
                _print_profile_catalog((profile,))
                print(json.dumps(profile.model_dump(mode="json"), indent=2))
            ####
            return 0
        ####
        if arguments.csv is None:
            raise ValueError("a CSV path is required unless a profile inspection command is used")
        ####
        if arguments.ignore_time_column and arguments.time_column is not None:
            raise ValueError("--time-column and --ignore-time-column are mutually exclusive.")
        ####
        if arguments.top < 1:
            raise ValueError("--top must be at least 1.")
        ####

        representation = (
            None
            if arguments.representation is None
            else AttitudeRepresentation(arguments.representation)
        )
        layout = detect_csv_layout(
            arguments.csv,
            representation=representation,
            orientation_columns=arguments.orientation_columns,
            rate_columns=arguments.rate_columns,
            time_column=None if arguments.ignore_time_column else arguments.time_column,
        )
        if arguments.ignore_time_column:
            layout = CsvLayout(
                representation=layout.representation,
                orientation_columns=layout.orientation_columns,
                rate_columns=layout.rate_columns,
                time_column=None,
                available_columns=layout.available_columns,
            )
        ####

        if arguments.inspect:
            if arguments.output_format == "json":
                print(json.dumps(_layout_payload(layout), indent=2))
            else:
                _print_layout(layout)
            ####
            return 0
        ####

        config = load_config(arguments.config) if arguments.config else None
        config = resolve_config(
            config,
            constraints=_cli_constraints(arguments),
            preference_profile=(
                None if arguments.profile is None else load_preference_profile(arguments.profile)
            ),
            sample_period_s=arguments.sample_period_s,
        )
        data = load_csv_trajectory(
            arguments.csv,
            layout.representation,
            layout.orientation_columns,
            layout.rate_columns,
            layout.time_column,
        )
        report = infer_conventions(data, config)
        report_json = report.model_dump_json(indent=2)

        if arguments.output_format == "decision-json":
            print(report.decision.model_dump_json(indent=2))
        elif arguments.output_format == "json":
            print(report_json)
        else:
            _print_layout(layout)
            _print_report(report_json, arguments.top, report.decision)
        ####

        if arguments.report_json:
            write_report(arguments.report_json, report)
        ####
        if arguments.details_csv:
            rows = best_interval_diagnostics(data, report, config)
            write_interval_diagnostics(arguments.details_csv, rows)
        ####
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    ####
####


if __name__ == "__main__":
    raise SystemExit(main())
####
