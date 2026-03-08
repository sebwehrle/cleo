"""Centralized unit utilities for Atlas workflows.

This module provides canonical unit metadata handling and conversion utilities
that are dask-friendly and preserve xarray attrs.

Canonical attr key is 'units' (plural).
"""

from __future__ import annotations

import xarray as xr
from pint import UnitRegistry

_UREG: UnitRegistry = UnitRegistry()

# Canonical attr key (normative)
UNIT_ATTR_KEY = "units"


def get_unit_attr(da: xr.DataArray) -> str | None:
    """Get canonical unit attr.

    Args:
        da: DataArray to read unit from

    Returns:
        Unit string or None if no unit attr present

    """
    return da.attrs.get(UNIT_ATTR_KEY)


def set_unit_attr(
    da: xr.DataArray,
    unit: str,
) -> xr.DataArray:
    """Set canonical unit attr on DataArray.

    Returns new DataArray with updated attrs (does not mutate input).

    Args:
        da: Input DataArray
        unit: Unit string to set
    Returns:
        New DataArray with updated attrs
    """
    new_attrs = dict(da.attrs)
    new_attrs[UNIT_ATTR_KEY] = unit
    new_attrs.pop("unit", None)
    # Create new DataArray with explicit attrs (not merge)
    return xr.DataArray(
        da.data,
        coords=da.coords,
        dims=da.dims,
        name=da.name,
        attrs=new_attrs,
    )


def assert_convertible(from_unit: str, to_unit: str) -> None:
    """Assert units are dimensionally compatible for conversion.

    Args:
        from_unit: Source unit string
        to_unit: Target unit string

    Raises:
        ValueError: If units are not dimensionally compatible
    """
    try:
        (1.0 * _UREG(from_unit)).to(to_unit)
    except Exception as e:
        raise ValueError(f"Cannot convert from {from_unit!r} to {to_unit!r}: {e}") from e


def conversion_factor(from_unit: str, to_unit: str) -> float:
    """Get scalar multiplication factor for unit conversion.

    Returns factor such that: value_in_to_unit = value_in_from_unit * factor

    Args:
        from_unit: Source unit string
        to_unit: Target unit string

    Returns:
        Scalar multiplication factor

    Raises:
        ValueError: If units are not dimensionally compatible
    """
    assert_convertible(from_unit, to_unit)
    return float((1.0 * _UREG(from_unit)).to(to_unit).magnitude)


def convert_dataarray(
    da: xr.DataArray,
    to_unit: str,
    *,
    from_unit: str | None = None,
) -> xr.DataArray:
    """Convert DataArray to new unit.

    Dask-friendly: computes scalar factor and multiplies.
    Preserves all attrs except updates 'units'.

    Args:
        da: Input DataArray
        to_unit: Target unit string
        from_unit: Source unit (reads from attrs if None)

    Returns:
        New DataArray with converted values and updated 'units' attr

    Raises:
        ValueError: If no unit source available or units incompatible
    """
    if from_unit is None:
        from_unit = get_unit_attr(da)
    if from_unit is None:
        raise ValueError(
            "No unit attr found on DataArray and from_unit not specified. "
            "Set 'units' attr or pass from_unit explicitly."
        )

    factor = conversion_factor(from_unit, to_unit)

    # Multiply preserves dask laziness
    converted_data = da.data * factor

    # Preserve original attrs, update unit
    new_attrs = dict(da.attrs)
    new_attrs[UNIT_ATTR_KEY] = to_unit
    new_attrs.pop("unit", None)

    # Create new DataArray with explicit attrs (not merge)
    return xr.DataArray(
        converted_data,
        coords=da.coords,
        dims=da.dims,
        name=da.name,
        attrs=new_attrs,
    )


def convert_dataset_variable(
    ds: xr.Dataset,
    variable: str,
    to_unit: str,
    *,
    from_unit: str | None = None,
) -> xr.Dataset:
    """Convert a single variable in a Dataset to new unit.

    Returns new Dataset with converted variable (does not mutate input).

    Args:
        ds: Input Dataset
        variable: Variable name to convert
        to_unit: Target unit string
        from_unit: Source unit (reads from variable attrs if None)

    Returns:
        New Dataset with converted variable

    Raises:
        ValueError: If variable not found, no unit source, or units incompatible
    """
    if variable not in ds:
        raise ValueError(f"Variable {variable!r} not found in Dataset. Available: {list(ds.data_vars)}")

    da = ds[variable]
    converted = convert_dataarray(da, to_unit, from_unit=from_unit)

    return ds.assign({variable: converted})


# =============================================================================
# Canonical Unit Registry
# =============================================================================

# Canonical units for public variables/metrics.
# None means the variable is dimensionless or unit is not applicable.
CANONICAL_UNITS: dict[str, str | None] = {
    # Wind metrics
    "wind_speed": "m/s",
    "mean_wind_speed": "m/s",
    "rotor_equivalent_wind_speed": "m/s",
    "rews_mps": "m/s",
    "capacity_factors": "1",  # dimensionless fraction
    # Economics metrics
    "lcoe": "EUR/MWh",
    "min_lcoe_turbine": None,  # index, no unit
    "optimal_power": "kW",
    "optimal_energy": "GWh/a",
    # Landscape metrics
    "elevation": "m",
    # Distance metrics (pattern: distance_*)
    "distance": "m",
    # Turbine metadata
    "turbine_capacity": "kW",
    "turbine_hub_height": "m",
    "turbine_rotor_diameter": "m",
    "turbine_commissioning_year": None,  # year, no unit
    # Wind physics (Weibull parameters)
    "weibull_A": "m/s",  # scale parameter
    "weibull_k": "1",  # shape parameter (dimensionless)
    "rho": "kg/m**3",  # air density
    # Power curve
    "power_curve": "1",  # dimensionless (capacity factor)
}


def get_canonical_unit(variable_name: str) -> str | None:
    """Get canonical unit for a known variable name.

    Also handles distance_* pattern matching.

    Args:
        variable_name: Variable name to look up

    Returns:
        Canonical unit string, None if dimensionless, or None if unknown
    """
    # Direct lookup
    if variable_name in CANONICAL_UNITS:
        return CANONICAL_UNITS[variable_name]

    # Pattern matching for distance_* variables
    if variable_name.startswith("distance_"):
        return CANONICAL_UNITS["distance"]

    return None


def is_known_variable(variable_name: str) -> bool:
    """Check if a variable name has a canonical unit definition.

    Args:
        variable_name: Variable name to check

    Returns:
        True if variable has canonical unit definition (including None for dimensionless)
    """
    if variable_name in CANONICAL_UNITS:
        return True
    if variable_name.startswith("distance_"):
        return True
    return False


def validate_unit_attr(
    da: xr.DataArray,
    variable_name: str,
    *,
    strict: bool = False,
) -> None:
    """Validate DataArray unit attr matches canonical expectation.

    Args:
        da: DataArray to validate
        variable_name: Variable name for canonical lookup
        strict: If True, raises for unknown variables; if False, skips unknown

    Raises:
        ValueError: If unit is present but doesn't match canonical
        ValueError: If canonical unit exists but attr is missing (when canonical is not None)
        ValueError: If strict=True and variable is unknown
    """
    canonical = get_canonical_unit(variable_name)

    # Unknown variable
    if not is_known_variable(variable_name):
        if strict:
            raise ValueError(
                f"Variable {variable_name!r} has no canonical unit definition. "
                f"Known variables: {sorted(CANONICAL_UNITS.keys())}"
            )
        return  # Skip validation for unknown variables

    # Dimensionless or no-unit variable
    if canonical is None:
        return  # No unit expected

    actual = get_unit_attr(da)

    if actual is None:
        raise ValueError(f"Variable {variable_name!r} missing required 'units' attr. Expected: {canonical!r}")

    if actual != canonical:
        raise ValueError(f"Variable {variable_name!r} has non-canonical unit: {actual!r}. Expected: {canonical!r}")


def list_canonical_units() -> dict[str, str | None]:
    """Return a copy of the canonical units registry.

    Returns:
        Dict mapping variable names to their canonical units
    """
    return dict(CANONICAL_UNITS)
