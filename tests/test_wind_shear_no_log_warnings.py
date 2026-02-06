"""Test that compute_wind_shear_coefficient does not emit RuntimeWarning from log(0)."""

import numpy as np
import xarray as xr
import warnings
from unittest.mock import MagicMock


def _create_mock_wind_atlas_with_zeros():
    """Create mock WindAtlas with mean_wind_speed containing zeros."""
    # u50 has a zero at index 0
    u50_values = [0.0, 5.0, 6.0]
    u100_values = [2.0, 10.0, 12.0]
    x = np.arange(len(u50_values), dtype=float)

    mean_wind_speed = xr.DataArray(
        data=np.array([u50_values, u100_values]),
        dims=["height", "x"],
        coords={"height": [50, 100], "x": x},
        name="mean_wind_speed",
    )

    mock_self = MagicMock()
    mock_self.data = xr.Dataset({"mean_wind_speed": mean_wind_speed})
    mock_self.parent = MagicMock()
    mock_self.parent.country = "AUT"

    return mock_self


def test_wind_shear_no_runtime_warning():
    """compute_wind_shear_coefficient should not emit RuntimeWarning from log(0)."""
    mock_self = _create_mock_wind_atlas_with_zeros()

    from cleo.assess import compute_wind_shear_coefficient

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_wind_shear_coefficient(mock_self)

        # Filter for RuntimeWarning about divide/invalid
        runtime_warnings = [
            warning for warning in w
            if issubclass(warning.category, RuntimeWarning)
            and ("divide by zero" in str(warning.message) or "invalid value" in str(warning.message))
        ]

        assert len(runtime_warnings) == 0, (
            f"Expected no RuntimeWarning from log, got {len(runtime_warnings)}: "
            f"{[str(w.message) for w in runtime_warnings]}"
        )


def test_wind_shear_nan_on_invalid_cells():
    """Wind shear should be NaN where input is invalid (u50=0)."""
    mock_self = _create_mock_wind_atlas_with_zeros()

    from cleo.assess import compute_wind_shear_coefficient

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    # Cell 0 has u50=0, should be NaN
    assert np.isnan(wind_shear.values[0]), "wind_shear[0] should be NaN (u50=0)"

    # Cells 1, 2 are valid
    assert np.isfinite(wind_shear.values[1]), "wind_shear[1] should be finite"
    assert np.isfinite(wind_shear.values[2]), "wind_shear[2] should be finite"


def test_wind_shear_no_inf_anywhere():
    """Wind shear should never contain inf values."""
    mock_self = _create_mock_wind_atlas_with_zeros()

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]
    assert not np.any(np.isinf(wind_shear.values)), "wind_shear should not contain inf"
