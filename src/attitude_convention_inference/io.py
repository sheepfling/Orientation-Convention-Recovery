"""CSV layout detection and report/configuration serialization."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy import asarray, float64

from .core import TrajectoryData
from .models import AttitudeRepresentation, InferenceConfig, InferenceReport

__all__ = [
    "CsvLayout",
    "detect_csv_layout",
    "load_config",
    "load_csv_trajectory",
    "read_csv_columns",
    "write_interval_diagnostics",
    "write_report",
]

_QUATERNION_COLUMN_GROUPS: tuple[tuple[str, str, str, str], ...] = (
    ("qw", "qx", "qy", "qz"),
    ("qx", "qy", "qz", "qw"),
    ("q0", "q1", "q2", "q3"),
    ("quat_w", "quat_x", "quat_y", "quat_z"),
    ("quat_x", "quat_y", "quat_z", "quat_w"),
    ("orientation_w", "orientation_x", "orientation_y", "orientation_z"),
    ("orientation_x", "orientation_y", "orientation_z", "orientation_w"),
)
_EULER_COLUMN_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("roll", "pitch", "yaw"),
    ("phi", "theta", "psi"),
    ("euler_x", "euler_y", "euler_z"),
    ("angle_x", "angle_y", "angle_z"),
)
_RATE_COLUMN_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("p", "q", "r"),
    ("wx", "wy", "wz"),
    ("w_x", "w_y", "w_z"),
    ("omega_x", "omega_y", "omega_z"),
    ("angular_rate_x", "angular_rate_y", "angular_rate_z"),
    ("body_rate_x", "body_rate_y", "body_rate_z"),
)
_TIME_COLUMNS: tuple[str, ...] = (
    "time_s",
    "time",
    "sim_time_s",
    "sim_time",
    "elapsed_s",
    "timestamp",
    "t",
)


@dataclass(frozen=True, slots=True)
class CsvLayout:
    representation: AttitudeRepresentation
    orientation_columns: tuple[str, ...]
    rate_columns: tuple[str, str, str]
    time_column: str | None
    available_columns: tuple[str, ...]
####




def _normalize_column_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
####




def read_csv_columns(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("CSV input is empty.") from error
        ####
    ####

    columns = tuple(header)
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("CSV input has an invalid or empty header column.")
    ####
    if len(set(columns)) != len(columns):
        raise ValueError("CSV input contains duplicate header names.")
    ####
    return columns
####




def _normalized_lookup(columns: Sequence[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column_name(column)
        if normalized in lookup:
            raise ValueError(
                "CSV columns become ambiguous after case/spacing normalization: "
                f"{lookup[normalized]!r} and {column!r}."
            )
        ####
        lookup[normalized] = column
    ####
    return lookup
####




def _resolve_explicit_columns(
    requested: Sequence[str],
    lookup: dict[str, str],
    *,
    label: str,
) -> tuple[str, ...]:
    resolved: list[str] = []
    for requested_name in requested:
        normalized = _normalize_column_name(requested_name)
        actual = lookup.get(normalized)
        if actual is None:
            raise ValueError(f"CSV input has no {label} column named {requested_name!r}.")
        ####
        resolved.append(actual)
    ####
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Explicit {label} columns may not contain duplicates.")
    ####
    return tuple(resolved)
####




def _matching_groups(
    groups: Sequence[Sequence[str]],
    lookup: dict[str, str],
) -> tuple[tuple[str, ...], ...]:
    matches: list[tuple[str, ...]] = []
    for group in groups:
        if all(name in lookup for name in group):
            matches.append(tuple(lookup[name] for name in group))
        ####
    ####
    return tuple(matches)
####




def _select_single_group(
    matches: tuple[tuple[str, ...], ...],
    *,
    label: str,
    available_columns: tuple[str, ...],
) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        available = ", ".join(available_columns)
        raise ValueError(
            f"Could not detect {label} columns. Supply them explicitly. "
            f"Available columns: {available}"
        )
    ####

    grouped_by_members: dict[frozenset[str], list[tuple[str, ...]]] = {}
    for candidate in unique:
        grouped_by_members.setdefault(frozenset(candidate), []).append(candidate)
    ####

    collapsed: list[tuple[str, ...]] = []
    for member_set, candidates in grouped_by_members.items():
        if len(candidates) == 1:
            collapsed.append(candidates[0])
            continue
        ####
        file_order = tuple(column for column in available_columns if column in member_set)
        exact_order = [candidate for candidate in candidates if candidate == file_order]
        if len(exact_order) == 1:
            collapsed.append(exact_order[0])
        else:
            collapsed.extend(candidates)
        ####
    ####

    if len(collapsed) > 1:
        rendered = "; ".join(", ".join(group) for group in collapsed)
        raise ValueError(
            f"Multiple plausible {label} column groups were found ({rendered}). "
            "Supply the intended columns explicitly."
        )
    ####
    return collapsed[0]
####




def detect_csv_layout(
    path: str | Path,
    *,
    representation: AttitudeRepresentation | None = None,
    orientation_columns: Sequence[str] | None = None,
    rate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> CsvLayout:
    """Resolve explicit or conventional CSV column names.

    Explicit values always win. Automatic detection is intentionally conservative: if
    more than one plausible orientation or rate group exists, the caller must choose.
    """

    available_columns = read_csv_columns(path)
    lookup = _normalized_lookup(available_columns)

    explicit_orientation: tuple[str, ...] | None = None
    if orientation_columns is not None:
        explicit_orientation = _resolve_explicit_columns(
            orientation_columns,
            lookup,
            label="orientation",
        )
        if len(explicit_orientation) not in (3, 4):
            raise ValueError(
                "Orientation columns must contain three Euler or four quaternion values."
            )
        ####
        inferred_representation = (
            AttitudeRepresentation.QUATERNION
            if len(explicit_orientation) == 4
            else AttitudeRepresentation.EULER
        )
        if representation is not None and representation is not inferred_representation:
            raise ValueError(
                f"{representation.value} input is incompatible with "
                f"{len(explicit_orientation)} orientation columns."
            )
        ####
        representation = inferred_representation
    ####

    if representation is None:
        quaternion_matches = _matching_groups(_QUATERNION_COLUMN_GROUPS, lookup)
        euler_matches = _matching_groups(_EULER_COLUMN_GROUPS, lookup)
        has_quaternion = bool(quaternion_matches)
        has_euler = bool(euler_matches)
        if has_quaternion == has_euler:
            available = ", ".join(available_columns)
            raise ValueError(
                "Could not uniquely infer quaternion versus Euler representation. "
                "Supply --representation or explicit orientation columns. "
                f"Available columns: {available}"
            )
        ####
        representation = (
            AttitudeRepresentation.QUATERNION if has_quaternion else AttitudeRepresentation.EULER
        )
    ####

    if explicit_orientation is None:
        groups: Sequence[Sequence[str]] = (
            _QUATERNION_COLUMN_GROUPS
            if representation is AttitudeRepresentation.QUATERNION
            else _EULER_COLUMN_GROUPS
        )
        explicit_orientation = _select_single_group(
            _matching_groups(groups, lookup),
            label=f"{representation.value} orientation",
            available_columns=available_columns,
        )
    ####

    expected_width = 4 if representation is AttitudeRepresentation.QUATERNION else 3
    if len(explicit_orientation) != expected_width:
        raise ValueError(f"{representation.value} requires {expected_width} orientation columns.")
    ####

    if rate_columns is None:
        resolved_rates = _select_single_group(
            _matching_groups(_RATE_COLUMN_GROUPS, lookup),
            label="angular-rate",
            available_columns=available_columns,
        )
    else:
        resolved_rates = _resolve_explicit_columns(rate_columns, lookup, label="angular-rate")
    ####
    if len(resolved_rates) != 3:
        raise ValueError("Exactly three angular-rate columns are required.")
    ####

    resolved_time: str | None
    if time_column is not None:
        resolved_time = _resolve_explicit_columns((time_column,), lookup, label="time")[0]
    else:
        resolved_time = next((lookup[name] for name in _TIME_COLUMNS if name in lookup), None)
    ####

    return CsvLayout(
        representation=representation,
        orientation_columns=explicit_orientation,
        rate_columns=(resolved_rates[0], resolved_rates[1], resolved_rates[2]),
        time_column=resolved_time,
        available_columns=available_columns,
    )
####




def load_csv_trajectory(
    path: str | Path,
    representation: AttitudeRepresentation,
    orientation_columns: tuple[str, ...],
    rate_columns: tuple[str, str, str],
    time_column: str | None = None,
) -> TrajectoryData:
    expected_orientation_columns = 4 if representation is AttitudeRepresentation.QUATERNION else 3
    if len(orientation_columns) != expected_orientation_columns:
        raise ValueError(
            f"{representation} requires {expected_orientation_columns} orientation columns."
        )
    ####

    orientation_rows: list[list[float]] = []
    rate_rows: list[list[float]] = []
    time_rows: list[float] = []
    source = Path(path)

    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV input has no header row.")
        ####
        required = set(orientation_columns) | set(rate_columns)
        if time_column is not None:
            required.add(time_column)
        ####
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV input is missing columns: {', '.join(missing)}")
        ####

        for line_number, row in enumerate(reader, start=2):
            try:
                orientation_rows.append([float(row[column]) for column in orientation_columns])
                rate_rows.append([float(row[column]) for column in rate_columns])
                if time_column is not None:
                    time_rows.append(float(row[time_column]))
                ####
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}.") from error
            ####
        ####
    ####

    return TrajectoryData(
        representation=representation,
        orientation=asarray(orientation_rows, dtype=float64),
        rates=asarray(rate_rows, dtype=float64),
        time_s=None if time_column is None else asarray(time_rows, dtype=float64),
    )
####




def load_config(path: str | Path) -> InferenceConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload: Any = json.load(stream)
    ####
    return InferenceConfig.model_validate(payload)
####




def write_report(path: str | Path, report: InferenceReport) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
####




def write_interval_diagnostics(
    path: str | Path,
    rows: tuple[dict[str, float | int | bool], ...],
) -> None:
    if not rows:
        raise ValueError("No interval diagnostics are available to write.")
    ####
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    ####
####


