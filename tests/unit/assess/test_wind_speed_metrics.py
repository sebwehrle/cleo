"""assess: test_wind_speed_metrics.

Contracts covered:
- compute_mean_wind_speed:
  - preserves dataset coords (notably 'wind_speed')
  - is idempotent for an already-present height (no recompute)
  - adds new heights exactly once (no duplicates)
- default wind-speed grid contract (0..40 step 0.5) as used by atlas datasets / compute_weibull_pdf
- compute_weibull_pdf works without turbines (only needs template + wind_speed + weibull params)
- power-curve resampling to atlas grid (np.interp contract)
- compute_optimal_power_energy:
  - optimal_energy units and 8766 hours convention
  - optimal_power must use metadata (get_turbine_attribute), not parse label strings
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from tests.helpers.oracles import assert_close
from tests.helpers.factories import da_xy_with_crs, wind_speed_axis

from cleo.assess import (
    compute_mean_wind_speed,
    compute_optimal_power_energy,
    compute_weibull_pdf,
)

DEFAULT_WIND_SPEED = wind_speed_axis()


@dataclass
class _Atlas:
    data: xr.Dataset

    def load_weibull_parameters(self, height: int | float):  # noqa: ARG002
        raise NotImplementedError

    def get_turbine_attribute(self, label: str, key: str) -> float:  # noqa: ARG002
        raise NotImplementedError

    def _set_var(self, name, da):
        """Simple mock for _set_var - just assigns directly."""
        self.data[name] = da


def _template_ds(*, x: np.ndarray, y: np.ndarray, wind_speed: np.ndarray = DEFAULT_WIND_SPEED) -> xr.Dataset:
    # NOTE: tests.helpers.factories.da_xy_with_crs() currently assumes square n×n coords.
    # Here we want explicit x/y coords (often rectangular), so build directly.
    template = xr.DataArray(
        np.zeros((y.size, x.size), dtype=np.float32),
        dims = ("y", "x"),
        coords = {"y": y, "x": x},
        name = "template",
        ).rio.write_crs("EPSG:4326")
    return xr.Dataset(data_vars={"template": template}, coords={"wind_speed": wind_speed})


def test_default_wind_speed_grid_values() -> None:
    expected = wind_speed_axis()
    assert expected.size == 81
    assert expected[0] == 0.0
    assert expected[-1] == 40.0
    assert np.allclose(np.diff(expected), 0.5)
    assert np.array_equal(DEFAULT_WIND_SPEED, expected)


def test_compute_mean_wind_speed_preserves_dataset_coords() -> None:
    """
    Crucial regression: computing mean wind speed must not drop existing coords,
    especially the atlas-wide 'wind_speed' coord used elsewhere.
    """
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    ds = _template_ds(x=x, y=y, wind_speed=np.array([0.0, 1.0, 2.0], dtype=float))

    class Self(_Atlas):
        def load_weibull_parameters(self, height: int | float):  # noqa: ARG002
            a = xr.DataArray(
                np.ones((y.size, x.size), dtype=np.float32) * 8.0,
                dims = ("y", "x"),
                coords = {"y": y, "x": x},
                name = "a",
            ).rio.write_crs("EPSG:4326")
            k = xr.DataArray(
                np.ones((y.size, x.size), dtype=np.float32) * 2.0,
                dims = ("y", "x"),
                coords = {"y": y, "x": x},
                name = "k",
            ).rio.write_crs("EPSG:4326")
            return a, k

    self = Self(ds)

    coords_before = set(self.data.coords)
    compute_mean_wind_speed(self, height=50, inplace=True)
    compute_mean_wind_speed(self, height=100, inplace=True)
    coords_after = set(self.data.coords)

    assert "wind_speed" in coords_after
    assert coords_after.issuperset(coords_before.union({"height"}))


def test_wind_atlas_has_wind_speed_without_turbine_and_compute_weibull_pdf_works() -> None:
    """
    Minimal atlas: template + wind_speed + weibull parameters should be sufficient.
    No turbine definitions needed for compute_weibull_pdf.
    """
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 0.0], dtype=float)
    ds = _template_ds(x=x, y=y, wind_speed=DEFAULT_WIND_SPEED).rio.write_crs("EPSG:4326")
    ds.attrs.setdefault("country", "TST")

    class Self(_Atlas):
        def load_weibull_parameters(self, height: int | float):  # noqa: ARG002
            a = xr.DataArray(
                np.ones((y.size, x.size), dtype=np.float32) * 8.0,
                dims=("y", "x"),
                coords={"y": y, "x": x},
                name="a",
            ).rio.write_crs("EPSG:4326")
            k = xr.DataArray(
                np.ones((y.size, x.size), dtype=np.float32) * 2.0,
                dims=("y", "x"),
                coords={"y": y, "x": x},
                name="k",
            ).rio.write_crs("EPSG:4326")
            return a, k

    self = Self(ds)

    assert "wind_speed" in self.data.coords
    assert np.array_equal(self.data.coords["wind_speed"].values, DEFAULT_WIND_SPEED)

    compute_weibull_pdf(self)

    assert "weibull_pdf" in self.data.data_vars
    pdf = self.data["weibull_pdf"]
    assert "wind_speed" in pdf.dims
    assert np.array_equal(pdf.coords["wind_speed"].values, DEFAULT_WIND_SPEED)
    assert "y" in pdf.dims and "x" in pdf.dims
    assert np.all(np.isfinite(pdf.values))
    assert np.all(pdf.values >= 0.0)


def test_power_curve_resampling_contract_matches_numpy_interp() -> None:
    old_u = np.array([0.0, 10.0, 20.0, 30.0])
    old_p = np.array([0.0, 0.25, 0.75, 1.0])

    atlas_u = DEFAULT_WIND_SPEED
    expected_p = np.interp(atlas_u, old_u, old_p, left=0.0, right=0.0)

    # Spot checks (avoid magic indices by mapping to exact speeds)
    idx_5 = int(np.where(np.isclose(atlas_u, 5.0))[0][0])
    idx_15 = int(np.where(np.isclose(atlas_u, 15.0))[0][0])
    idx_40 = int(np.where(np.isclose(atlas_u, 40.0))[0][0])

    assert_close(expected_p[idx_5], 0.125, rtol=0.0, atol=1e-15)
    assert_close(expected_p[idx_15], 0.5, rtol=0.0, atol=1e-15)
    assert_close(expected_p[idx_40], 0.0, rtol=0.0, atol=1e-15)

    power_curve = xr.DataArray(expected_p, coords={"wind_speed": atlas_u}, dims=("wind_speed",), name="power_curve")
    assert np.array_equal(power_curve.coords["wind_speed"].values, atlas_u)
    assert np.allclose(power_curve.values, expected_p)


def test_compute_mean_wind_speed_idempotent_no_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If mean_wind_speed already contains a given height with valid signature,
    compute_mean_wind_speed(height) must not call load_weibull_parameters again (idempotent contract).

    Note: Semantic cache requires valid cleo:sigs attr; data without signatures WILL be recomputed.
    """
    from cleo.assess import _sig, _canon

    # Compute the expected signature for this height (no parent -> country=None, crs=None)
    params = {"country": None, "crs": None, "height": 100}
    expected_sig = _sig("compute_mean_wind_speed", "1", params)
    sigs_json = _canon({"100": expected_sig})

    existing = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 7.5,
        dims=("height", "y", "x"),
        coords={"height": [100.0], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
        attrs={
            "cleo:algo": "compute_mean_wind_speed",
            "cleo:algo_version": "1",
            "cleo:sigs": sigs_json,
        },
    )

    class Self(_Atlas):
        def __init__(self):
            super().__init__(xr.Dataset({"mean_wind_speed": existing}))
            self.calls = 0

        def load_weibull_parameters(self, height: int | float):
            self.calls += 1
            return xr.DataArray(np.ones((4, 4)) * 8.0, dims=("y", "x")), xr.DataArray(np.ones((4, 4)) * 2.0, dims=("y", "x"))

    self = Self()
    compute_mean_wind_speed(self, 100.0, inplace=True)
    compute_mean_wind_speed(self, 100.0, inplace=True)

    assert self.calls == 0
    heights = list(self.data["mean_wind_speed"].coords["height"].values)
    assert heights == [100.0]


def test_compute_mean_wind_speed_adds_new_height() -> None:
    existing = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 6.0,
        dims=("height", "y", "x"),
        coords={"height": [50.0], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
    )

    class Self(_Atlas):
        def __init__(self):
            super().__init__(xr.Dataset({"mean_wind_speed": existing}))
            self.calls = 0

        def load_weibull_parameters(self, height: int | float):  # noqa: ARG002
            self.calls += 1
            a = xr.DataArray(np.ones((4, 4)) * 8.0, dims=("y", "x"))
            k = xr.DataArray(np.ones((4, 4)) * 2.0, dims=("y", "x"))
            return a, k

    self = Self()
    compute_mean_wind_speed(self, 100.0, inplace=True)

    assert self.calls == 1
    heights = sorted(self.data["mean_wind_speed"].coords["height"].values.tolist())
    assert heights == [50.0, 100.0]


def test_compute_mean_wind_speed_no_duplicates_after_multiple_calls() -> None:
    existing = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 7.5,
        dims=("height", "y", "x"),
        coords={"height": [100.0], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
    )

    class Self(_Atlas):
        def __init__(self):
            super().__init__(xr.Dataset({"mean_wind_speed": existing}))

        def load_weibull_parameters(self, height: int | float):  # noqa: ARG002
            return xr.DataArray(np.ones((4, 4)) * 8.0, dims=("y", "x")), xr.DataArray(np.ones((4, 4)) * 2.0, dims=("y", "x"))

    self = Self()
    for _ in range(5):
        compute_mean_wind_speed(self, 100.0, inplace=True)

    heights = list(self.data["mean_wind_speed"].coords["height"].values)
    assert heights == [100.0]
    assert len(heights) == 1


def test_optimal_energy_is_gwh_per_year_and_has_units_attr() -> None:
    turbines = ["T1", "T2"]
    ds = xr.Dataset(
        {
            "lcoe": xr.DataArray(
                np.array([[[10.0]], [[20.0]]], dtype=np.float64),
                dims=("turbine", "y", "x"),
                coords={"turbine": turbines, "y": [0], "x": [0]},
            ),
            "capacity_factors": xr.DataArray(
                np.array([[[0.5]], [[0.25]]], dtype=np.float64),
                dims=("turbine", "y", "x"),
                coords={"turbine": turbines, "y": [0], "x": [0]},
            ),
        },
        attrs={"country": "AUT"},
    )

    class Self(_Atlas):
        def __init__(self):
            super().__init__(ds)

        def get_turbine_attribute(self, label: str, key: str) -> float:
            assert key == "capacity"
            return {"T1": 1000.0, "T2": 2000.0}[label]

    self = Self()
    compute_optimal_power_energy(self)

    out = self.data["optimal_energy"]
    assert out.attrs.get("units") == "GWh/a"

    expected = 0.5 * 1000.0 * 8766.0 / 1e6
    got = float(out.values.item())
    assert_close(got, expected, rtol=0.0, atol=1e-12)


def test_optimal_power_uses_metadata_not_label() -> None:
    """
    If someone parses the label "Vestas.V112.3.45", they'd pick 45.
    We require metadata capacity=3.45 to win.
    """
    turbine_label = "Vestas.V112.3.45"
    ds = xr.Dataset(
        {
            "lcoe": xr.DataArray(
                np.array([[[100.0]]], dtype=np.float64),
                dims=("turbine", "y", "x"),
                coords={"turbine": [turbine_label], "y": [0.0], "x": [0.0]},
            ),
            "capacity_factors": xr.DataArray(
                np.array([[[0.35]]], dtype=np.float64),
                dims=("turbine", "y", "x"),
                coords={"turbine": [turbine_label], "y": [0.0], "x": [0.0]},
            ),
        }
    )

    class Self(_Atlas):
        def __init__(self):
            super().__init__(ds)

        def get_turbine_attribute(self, label: str, key: str) -> float:
            assert key == "capacity"
            assert label == turbine_label
            return 3.45

    self = Self()
    compute_optimal_power_energy(self)

    got = float(self.data["optimal_power"].values.item())
    assert_close(got, 3.45, rtol=0.0, atol=0.0)


def test_optimal_power_multiple_turbines_varies_by_location() -> None:
    turbines = ["Enercon.E126.7.58", "Vestas.V90.3.0"]
    lcoe = xr.DataArray(
        np.array(
            [
                [[80.0, 120.0]],   # Enercon wins at x=0
                [[100.0, 70.0]],   # Vestas wins at x=1
            ],
            dtype=np.float64,
        ),
        dims=("turbine", "y", "x"),
        coords={"turbine": turbines, "y": [0.0], "x": [0.0, 1.0]},
    )
    cfs = xr.DataArray(
        np.ones((2, 1, 2), dtype=np.float64) * 0.3,
        dims=("turbine", "y", "x"),
        coords={"turbine": turbines, "y": [0.0], "x": [0.0, 1.0]},
    )

    capacity_map = {"Enercon.E126.7.58": 7.58, "Vestas.V90.3.0": 3.0}

    class Self(_Atlas):
        def __init__(self):
            super().__init__(xr.Dataset({"lcoe": lcoe, "capacity_factors": cfs}))

        def get_turbine_attribute(self, label: str, key: str) -> float:
            assert key == "capacity"
            return float(capacity_map[label])

    self = Self()
    compute_optimal_power_energy(self)

    out = self.data["optimal_power"]
    assert_close(float(out.isel(y=0, x=0).values), 7.58, rtol=0.0, atol=0.0)
    assert_close(float(out.isel(y=0, x=1).values), 3.0, rtol=0.0, atol=0.0)
