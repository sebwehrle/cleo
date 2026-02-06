from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import compute_optimal_power_energy


class DummyAtlas:
    def __init__(self):
        turbines = ["T1", "T2"]
        self.data = xr.Dataset(
            {
                "lcoe": xr.DataArray(
                    np.array([[[10.0]], [[20.0]]], dtype=np.float32),
                    dims=("turbine", "y", "x"),
                    coords={"turbine": turbines, "y": [0], "x": [0]},
                ),
                "capacity_factors": xr.DataArray(
                    np.array([[[0.5]], [[0.25]]], dtype=np.float32),
                    dims=("turbine", "y", "x"),
                    coords={"turbine": turbines, "y": [0], "x": [0]},
                ),
            },
            attrs={"country": "AUT"},
        )

    def get_turbine_attribute(self, label, key):
        # capacities in kW
        assert key == "capacity"
        return {"T1": 1000.0, "T2": 2000.0}[label]


def test_optimal_energy_is_gwh_per_year_and_has_units_attr():
    self = DummyAtlas()
    compute_optimal_power_energy(self)

    assert "optimal_energy" in self.data.data_vars
    out = self.data["optimal_energy"]
    assert out.attrs.get("units") == "GWh/a"

    # Least-cost turbine is T1 with power 1000 kW and CF 0.5
    expected = 0.5 * 1000.0 * 8766.0 / 1e6  # GWh/year
    got = float(out.values.item())
    assert abs(got - expected) < 1e-12
