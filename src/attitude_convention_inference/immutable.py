"""Copied read-only mappings that retain Pydantic's object-shaped JSON contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, TypeAlias, TypeVar, cast

from pydantic import PlainSerializer

__all__ = ["ReadOnlyMapping", "freeze_mapping", "freeze_nested_mapping"]

Key = TypeVar("Key")
InnerKey = TypeVar("InnerKey")
Value = TypeVar("Value")


def _plain_nested_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        return {key: _plain_nested_mapping(item) for key, item in items.items()}
    ####
    return value
####


def _serialize_mapping(value: Mapping[Key, Value]) -> dict[Key, Value]:
    """Return plain mappings at every depth for Pydantic's serializers."""

    return {key: cast(Value, _plain_nested_mapping(item)) for key, item in value.items()}
####


ReadOnlyMapping: TypeAlias = Annotated[
    Mapping[Key, Value], PlainSerializer(_serialize_mapping, return_type=dict[Key, Value])
]


def freeze_mapping(value: Mapping[Key, Value]) -> ReadOnlyMapping[Key, Value]:
    """Copy a mapping before exposing its read-only view to a model consumer."""

    return MappingProxyType(dict(value))
####


def freeze_nested_mapping(
    value: Mapping[Key, Mapping[InnerKey, Value]],
) -> ReadOnlyMapping[Key, ReadOnlyMapping[InnerKey, Value]]:
    """Copy and freeze both levels of a two-level mapping."""

    return freeze_mapping({key: freeze_mapping(items) for key, items in value.items()})
####
