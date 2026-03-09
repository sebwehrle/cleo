"""Shared turbine-selection validation helpers."""

from __future__ import annotations

from collections.abc import Sequence as ABCSequence


def _validate_sequence_not_scalar(
    value: object,
    param_name: str,
    item_type_desc: str,
    *,
    empty_hint: str | None = None,
    sequence_types: tuple[type[object], ...] = (list, tuple),
    sequence_label: str = "list/tuple",
) -> list[object]:
    """Validate that value is a non-empty sequence, not a scalar.

    :param value: Value to validate.
    :param param_name: Parameter name for error messages.
    :param item_type_desc: Description of expected item types for error messages.
    :param empty_hint: Optional suffix for the empty-sequence error message.
    :param sequence_types: Accepted runtime sequence container types.
    :param sequence_label: Human-readable label for accepted container types.
    :returns: Value converted to list.
    :raises ValueError: If value is a scalar, wrong type, or empty.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{param_name} must be a non-empty {sequence_label} of {item_type_desc}; "
            f"got a single string/bytes value. Use {param_name}=[...]."
        )
    if not isinstance(value, sequence_types):
        raise ValueError(f"{param_name} must be a {sequence_label} of {item_type_desc}, got {type(value).__name__}")
    items = list(value)
    if not items:
        message = f"{param_name} must be non-empty"
        if empty_hint is not None:
            message = f"{message}; {empty_hint}"
        raise ValueError(message)
    return items


def _validate_general_sequence_not_scalar(
    value: object,
    param_name: str,
    item_type_desc: str,
    *,
    empty_hint: str | None = None,
) -> list[object]:
    """Validate that value is a non-empty general sequence, not a scalar.

    :param value: Value to validate.
    :param param_name: Parameter name for error messages.
    :param item_type_desc: Description of expected item types for error messages.
    :param empty_hint: Optional suffix for the empty-sequence error message.
    :returns: Value converted to list.
    :raises ValueError: If value is a scalar, wrong type, or empty.
    """
    return _validate_sequence_not_scalar(
        value,
        param_name,
        item_type_desc,
        empty_hint=empty_hint,
        sequence_types=(ABCSequence,),
        sequence_label="sequence",
    )


def _normalize_turbine_ids(
    items: list[object],
    *,
    available: tuple[str, ...] | None = None,
    unknown_hint: str = "atlas.wind.turbines",
) -> tuple[str, ...]:
    """Normalize turbine IDs and optionally validate them against an inventory.

    :param items: Turbine ID values to normalize.
    :param available: Optional available turbine inventory for membership checks.
    :param unknown_hint: Help text appended to unknown-ID errors.
    :returns: Normalized tuple of turbine IDs with order preserved.
    :raises ValueError: If any ID is not a string, empty, duplicate, or unknown.
    """
    available_set = set(available) if available is not None else None
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"Each turbine ID must be a string, got {type(item).__name__}")
        stripped = item.strip()
        if not stripped:
            raise ValueError("Turbine ID cannot be empty or whitespace-only")
        if stripped in seen:
            raise ValueError(f"Duplicate turbine ID: {stripped!r}")
        if available_set is not None and stripped not in available_set:
            raise ValueError(f"Unknown turbine ID: {stripped!r}; see {unknown_hint}")
        seen.add(stripped)
        cleaned.append(stripped)
    return tuple(cleaned)
