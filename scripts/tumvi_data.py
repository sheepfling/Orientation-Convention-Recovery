"""Shared, unit-checked readers for extracted TUM-VI CSV streams."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from numpy import all as np_all
from numpy import any as np_any
from numpy import asarray, float64, int64
from numpy import isfinite as np_isfinite
from numpy.typing import NDArray

__all__ = [
    "NANOSECONDS_PER_SECOND",
    "TimedVectorSeries",
    "read_timed_vectors",
    "sha256_file",
]

FloatArray = NDArray[float64]
IntegerArray = NDArray[int64]
NANOSECONDS_PER_SECOND = 1_000_000_000
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class TimedVectorSeries:
    """One finite, strictly timestamped vector stream parsed from TUM-VI CSV."""

    timestamps_ns: IntegerArray
    values: FloatArray
####


def _normalize_header(value: str) -> str:
    without_units = re.sub(r"\[[^\]]*\]", "", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", without_units.lstrip("#")).strip("_").lower()
####


def _header_unit(value: str) -> str | None:
    matches = re.findall(r"\[([^\]]*)\]", value)
    if len(matches) != 1:
        return None
    ####
    return matches[0].strip()
####


def _normalize_unit(value: str) -> str:
    return (
        value.casefold()
        .replace("⁻", "-")
        .replace("¹", "1")
        .replace("^", "")
        .replace("{", "")
        .replace("}", "")
        .replace(" ", "")
    )
####


def _axis_column_indices(
    headers: tuple[str, ...],
    *,
    axes: tuple[str, ...],
    prefix: str,
    path: Path,
) -> tuple[int, ...]:
    indices: list[int] = []
    for axis in axes:
        matches = tuple(
            index for index, header in enumerate(headers) if header == f"{prefix}_{axis}"
        )
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected one TUM-VI {prefix!r} {axis!r} column, found {len(matches)}."
            )
        ####
        indices.append(matches[0])
    ####
    return tuple(indices)
####


def read_timed_vectors(
    path: Path,
    *,
    axes: tuple[str, ...],
    prefix: str,
    expected_unit: Literal["rad/s", "unitless"],
) -> TimedVectorSeries:
    """Read one EuRoC-style timestamped TUM-VI vector stream with declared units."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            raw_headers = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"{path}: CSV is empty.") from error
        ####
        if not raw_headers:
            raise ValueError(f"{path}: CSV has no header columns.")
        ####
        headers = tuple(_normalize_header(value) for value in raw_headers)
        timestamp_indices = tuple(
            index for index, header in enumerate(headers) if header == "timestamp"
        )
        if len(timestamp_indices) != 1:
            raise ValueError(f"{path}: expected one TUM-VI timestamp [ns] column.")
        ####
        timestamp_unit = _header_unit(raw_headers[timestamp_indices[0]])
        if timestamp_unit is None or _normalize_unit(timestamp_unit) != "ns":
            raise ValueError(f"{path}: the TUM-VI timestamp column must declare [ns] units.")
        ####
        value_indices = _axis_column_indices(
            headers,
            axes=axes,
            prefix=prefix,
            path=path,
        )
        expected_units = {"rads-1", "rad/s"} if expected_unit == "rad/s" else {""}
        expected_description = (
            "radians per second" if expected_unit == "rad/s" else "unitless quaternion"
        )
        for index in value_indices:
            declared_unit = _header_unit(raw_headers[index])
            if declared_unit is None or _normalize_unit(declared_unit) not in expected_units:
                raise ValueError(
                    f"{path}: TUM-VI column {raw_headers[index]!r} must declare "
                    f"{expected_description} units."
                )
            ####
        ####
        timestamp_index = timestamp_indices[0]
        timestamps: list[int] = []
        vectors: list[tuple[float, ...]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            ####
            if len(row) != len(raw_headers):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(raw_headers)} columns, found {len(row)}."
                )
            ####
            try:
                timestamp = int(row[timestamp_index].strip())
                vector = tuple(float(row[index]) for index in value_indices)
            except ValueError as error:
                message = f"{path}:{line_number}: invalid timestamp or vector value."
                raise ValueError(message) from error
            ####
            if not _INT64_MIN <= timestamp <= _INT64_MAX:
                raise ValueError(
                    f"{path}:{line_number}: timestamp must fit signed 64-bit nanoseconds."
                )
            ####
            timestamps.append(timestamp)
            vectors.append(vector)
        ####
    ####

    timestamp_array = asarray(timestamps, dtype=int64)
    vector_array = asarray(vectors, dtype=float64)
    if timestamp_array.size < 2 or vector_array.shape != (timestamp_array.size, len(axes)):
        raise ValueError(f"{path}: expected at least two complete measurement rows.")
    ####
    if np_any(timestamp_array[1:] <= timestamp_array[:-1]):
        raise ValueError(f"{path}: timestamps must be strictly increasing.")
    ####
    timestamp_span_ns = int(timestamp_array[-1]) - int(timestamp_array[0])
    if timestamp_span_ns > _INT64_MAX:
        raise ValueError(f"{path}: timestamp span must fit signed 64-bit nanosecond offsets.")
    ####
    if not np_all(np_isfinite(vector_array)):
        raise ValueError(f"{path}: vectors must contain only finite values.")
    ####
    return TimedVectorSeries(timestamps_ns=timestamp_array, values=vector_array)
####


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 fingerprint for a source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        ####
    ####
    return digest.hexdigest()
####
