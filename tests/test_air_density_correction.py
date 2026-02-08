r"""test_air_density_correction.py

Oracle tests for air density correction in capacity factor computation.

Air density correction applies equivalent wind speed INSIDE power curve lookup:
u_eq = u * (rho/rho0)^(1/3) = u * c

This is NOT a post-hoc CF scaling - it shifts the effective wind speed thresholds.

Oracle formulas (LaTeX):
- Weibull CDF: F(u) = 1 - \exp\left(-\left(\frac{u}{A}\right)^k\right)
- Weibull tail: P(U \geq u) = \exp\left(-\left(\frac{u}{A}\right)^k\right)
- Step curve with density: CF = P(U \geq u_0/c) = \exp\left(-\left(\frac{u_0}{c \cdot A}\right)^k\right)
- Top-hat with density: CF = \exp\left(-\left(\frac{u_{in}}{c \cdot A}\right)^k\right) - \exp\left(-\left(\frac{u_{out}}{c \cdot A}\right)^k\right)

No elevation/CopDEM is used - only GWA air_density rasters.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from types import SimpleNamespace

import pytest

from cleo.assess import (
    simulate_capacity_factors,
    compute_air_density_at_height,
    weibull_probability_density,
    _interp_da_to_height_log,
    _integrate_cf_with_density_correction,
    RHO_0,
    GWA_HEIGHTS,
)


# ============================================================================
# Helper functions
# ============================================================================

def weibull_tail_probability(u: float, k: float, a: float) -> float:
    """P(U >= u) = exp(-(u/a)^k)"""
    return np.exp(-((u / a) ** k))


def create_mock_atlas_with_density(
    *,
    y: np.ndarray,
    x: np.ndarray,
    wind_speed: np.ndarray,
    weibull_A: np.ndarray,  # shape (n_heights, ny, nx)
    weibull_k: np.ndarray,  # shape (n_heights, ny, nx)
    air_density: np.ndarray,  # shape (n_heights, ny, nx)
    heights: np.ndarray,
    power_curve: np.ndarray,  # shape (n_wind_speed,)
    turbine_name: str,
    hub_height: float,
):
    """Create a mock atlas-like object for testing."""
    # Weibull A and k at heights (for PDF interpolation)
    A_da = xr.DataArray(
        weibull_A,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k_da = xr.DataArray(
        weibull_k,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )

    # Air density at heights
    rho_da = xr.DataArray(
        air_density,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="air_density",
    )

    # Power curve
    pc_da = xr.DataArray(
        power_curve.reshape(1, -1),
        dims=("turbine", "wind_speed"),
        coords={"turbine": [turbine_name], "wind_speed": wind_speed},
        name="power_curve",
    )

    # Template
    template = xr.DataArray(
        np.ones((len(y), len(x)), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="template",
    )

    ds = xr.Dataset(
        data_vars={
            "air_density": rho_da,
            "power_curve": pc_da,
            "template": template,
        },
        coords={
            "turbine": [turbine_name],
            "wind_speed": wind_speed,
        },
        attrs={"country": "TEST"},
    )

    class MockAtlas:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")
            self._weibull_A = A_da
            self._weibull_k = k_da

        def get_turbine_attribute(self, turbine_id: str, key: str):
            if key == "hub_height":
                return hub_height
            raise KeyError(f"Unknown attribute: {key}")

        def _set_var(self, name, da):
            self.data[name] = da

        def load_weibull_parameters(self, height):
            """Return Weibull A, k at specified height."""
            return (
                self._weibull_A.sel(height=height),
                self._weibull_k.sel(height=height),
            )

    return MockAtlas(ds)


# ============================================================================
# Test 1: Step curve tail shift oracle
# ============================================================================

def test_step_curve_with_density_correction_oracle():
    r"""
    Test step curve CF with density correction equals shifted tail probability.

    Step curve: CF_ref(u) = 1{u >= u_0}
    With density: CF(u) = 1{u*c >= u_0} = 1{u >= u_0/c}

    Oracle: CF = P(U >= u_0/c) = \exp\left(-\left(\frac{u_0}{c \cdot A}\right)^k\right)
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)  # 2x2 grid with different densities
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    # Constant Weibull params across heights (trivial interpolation)
    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)

    # Spatially varying air density (constant over height)
    # Pixel [0,0]: rho = rho0 * 0.9 (low density)
    # Pixel [0,1]: rho = rho0 * 1.1 (high density)
    # Pixel [1,0]: rho = rho0 (reference)
    # Pixel [1,1]: rho = rho0 * 0.8 (very low)
    rho_vals = np.array([
        [[RHO_0 * 0.9, RHO_0 * 1.1], [RHO_0, RHO_0 * 0.8]],
    ])
    air_density = np.broadcast_to(rho_vals, (4, 2, 2)).copy()

    # Step power curve at u_0 = 6.0
    u_0 = 6.0
    p_step = np.where(wind_speed >= u_0, 1.0, 0.0)

    turbine_name = "TestTurbine"
    hub_height = 100.0

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=p_step, turbine_name=turbine_name,
        hub_height=hub_height,
    )

    # Run with density correction
    simulate_capacity_factors(
        atlas, chunk_size=None, loss_factor=1.0,
        force=True, weibull_height_mode="hub", air_density_mode="gwa"
    )

    cf = atlas.data["capacity_factors"].sel(turbine=turbine_name)

    # Compute oracle for each pixel
    for iy in range(2):
        for ix in range(2):
            rho = air_density[0, iy, ix]
            c = (rho / RHO_0) ** (1.0 / 3.0)
            # Oracle: CF = exp(-((u_0 / (c * A))^k))
            cf_oracle = weibull_tail_probability(u_0 / c, k, A)
            cf_numeric = float(cf.isel(y=iy, x=ix).values)

            assert abs(cf_numeric - cf_oracle) < 0.03, (
                f"Pixel [{iy},{ix}]: CF={cf_numeric:.6f}, oracle={cf_oracle:.6f}, "
                f"diff={abs(cf_numeric - cf_oracle):.6f}"
            )


# ============================================================================
# Test 2: Top-hat curve oracle
# ============================================================================

def test_tophat_curve_with_density_correction_oracle():
    r"""
    Test top-hat power curve CF with density correction.

    Top-hat: CF_ref(u) = 1{u_in <= u < u_out}
    With density: thresholds divide by c.

    Oracle: CF = \exp\left(-\left(\frac{u_{in}}{c \cdot A}\right)^k\right)
                - \exp\left(-\left(\frac{u_{out}}{c \cdot A}\right)^k\right)
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)

    # Spatially varying density
    rho_vals = np.array([
        [[RHO_0 * 0.85, RHO_0 * 1.15], [RHO_0 * 0.95, RHO_0 * 1.05]],
    ])
    air_density = np.broadcast_to(rho_vals, (4, 2, 2)).copy()

    # Top-hat power curve: 1 for u_in <= u < u_out
    u_in = 6.0
    u_out = 12.0
    p_tophat = np.where((wind_speed >= u_in) & (wind_speed < u_out), 1.0, 0.0)

    turbine_name = "TopHatTurbine"
    hub_height = 100.0

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=p_tophat, turbine_name=turbine_name,
        hub_height=hub_height,
    )

    simulate_capacity_factors(
        atlas, chunk_size=None, loss_factor=1.0,
        force=True, weibull_height_mode="hub", air_density_mode="gwa"
    )

    cf = atlas.data["capacity_factors"].sel(turbine=turbine_name)

    for iy in range(2):
        for ix in range(2):
            rho = air_density[0, iy, ix]
            c = (rho / RHO_0) ** (1.0 / 3.0)
            # Oracle: difference of tails
            cf_oracle = (
                weibull_tail_probability(u_in / c, k, A) -
                weibull_tail_probability(u_out / c, k, A)
            )
            cf_numeric = float(cf.isel(y=iy, x=ix).values)

            assert abs(cf_numeric - cf_oracle) < 0.03, (
                f"Pixel [{iy},{ix}]: CF={cf_numeric:.6f}, oracle={cf_oracle:.6f}"
            )


# ============================================================================
# Test 3: Identity at rho = rho0
# ============================================================================

def test_density_correction_identity_at_reference():
    """
    When air_density == rho0 everywhere, density correction has no effect.
    CF with air_density_mode="gwa" should equal CF with air_density_mode="none".
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 3), A)
    weibull_k = np.full((4, 2, 3), k)

    # Air density = rho0 everywhere
    air_density = np.full((4, 2, 3), RHO_0)

    # Step power curve
    u_0 = 6.0
    p_step = np.where(wind_speed >= u_0, 1.0, 0.0)

    turbine_name = "RefDensityTurbine"
    hub_height = 100.0

    # Run with density correction
    atlas_gwa = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=p_step, turbine_name=turbine_name,
        hub_height=hub_height,
    )
    simulate_capacity_factors(
        atlas_gwa, chunk_size=None, loss_factor=1.0,
        force=True, weibull_height_mode="hub", air_density_mode="gwa"
    )
    cf_gwa = atlas_gwa.data["capacity_factors"].sel(turbine=turbine_name).values

    # Run without density correction
    atlas_none = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=p_step, turbine_name=turbine_name,
        hub_height=hub_height,
    )
    simulate_capacity_factors(
        atlas_none, chunk_size=None, loss_factor=1.0,
        force=True, weibull_height_mode="hub", air_density_mode="none"
    )
    cf_none = atlas_none.data["capacity_factors"].sel(turbine=turbine_name).values

    # Should be identical
    np.testing.assert_allclose(
        cf_gwa, cf_none, rtol=0, atol=1e-12,
        err_msg="CF with rho=rho0 should be identical with and without density correction"
    )


# ============================================================================
# Test 4: Air density height interpolation uses shared helper
# ============================================================================

def test_air_density_interpolation_uses_linear_in_height():
    r"""
    Verify compute_air_density_at_height uses LINEAR interpolation in height
    (NOT log-height-linear like Weibull parameters).

    Build air_density(height) exactly linear: rho(h) = p0 + p1 * h
    Assert interpolated value matches oracle with tight tolerance.
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    # Define LINEAR model: rho(h) = p0 + p1 * h
    p0 = 1.3
    p1 = -0.001  # Density decreases with height

    def rho_linear(h):
        return p0 + p1 * h

    # Create air density at each height
    rho_vals = np.stack([
        np.full((2, 3), rho_linear(h)) for h in heights
    ], axis=0)

    rho_da = xr.DataArray(
        rho_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="air_density",
    )

    # Template for mock atlas
    template = xr.DataArray(
        np.ones((len(y), len(x)), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="template",
    )

    ds = xr.Dataset(
        data_vars={"air_density": rho_da, "template": template},
        coords={"wind_speed": wind_speed},
        attrs={"country": "TEST"},
    )

    class MockAtlas:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")

    atlas = MockAtlas(ds)

    # Target heights to test
    for target_height in [80.0, 120.0, 175.0]:
        rho_hub = compute_air_density_at_height(atlas, target_height)

        # Oracle: linear interpolation
        rho_oracle = rho_linear(target_height)

        # Check all pixels match oracle with tight tolerance
        np.testing.assert_allclose(
            rho_hub.values, rho_oracle, rtol=0, atol=1e-12,
            err_msg=f"Air density interpolation at h={target_height} does not match linear oracle"
        )


# ============================================================================
# Test 5: Air density mode validates correctly
# ============================================================================

def test_air_density_mode_validation():
    """Test that invalid air_density_mode raises ValueError."""
    import pytest

    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)
    air_density = np.full((4, 2, 2), RHO_0)
    p_step = np.where(wind_speed >= 6.0, 1.0, 0.0)

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=p_step, turbine_name="T1", hub_height=100.0,
    )

    # Invalid mode
    with pytest.raises(ValueError, match="air_density_mode must be"):
        simulate_capacity_factors(
            atlas, force=True, weibull_height_mode="hub", air_density_mode="invalid"
        )

    # gwa mode with 100m_shear should raise
    with pytest.raises(ValueError, match="only supported with weibull_height_mode='hub'"):
        simulate_capacity_factors(
            atlas, force=True, weibull_height_mode="100m_shear", air_density_mode="gwa"
        )


# ============================================================================
# Test 6: Missing air_density AND raw files raises actionable error
# ============================================================================

def test_missing_air_density_and_raw_files_raises():
    """
    Test that air_density_mode='gwa' without air_density data AND without raw GWA files
    raises an actionable FileNotFoundError mentioning materialize/_load_gwa.
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0

    # Create atlas WITHOUT air_density
    A_da = xr.DataArray(
        np.full((4, 2, 2), A),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )
    k_da = xr.DataArray(
        np.full((4, 2, 2), k),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )
    pc_da = xr.DataArray(
        np.where(wind_speed >= 6.0, 1.0, 0.0).reshape(1, -1),
        dims=("turbine", "wind_speed"),
        coords={"turbine": ["T1"], "wind_speed": wind_speed},
    )
    template = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )

    ds = xr.Dataset(
        data_vars={"power_curve": pc_da, "template": template},
        coords={"turbine": ["T1"], "wind_speed": wind_speed},
        attrs={"country": "TEST"},
    )

    class MockAtlasNoRho:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")
            self._weibull_A = A_da
            self._weibull_k = k_da

        def get_turbine_attribute(self, turbine_id, key):
            if key == "hub_height":
                return 100.0
            raise KeyError(key)

        def _set_var(self, name, da):
            self.data[name] = da

        def load_weibull_parameters(self, height):
            return self._weibull_A.sel(height=height), self._weibull_k.sel(height=height)

        def load_air_density(self, height):
            # Simulate missing raw files
            raise FileNotFoundError(
                f"Missing air-density raster for height={height}. "
                f"Run atlas.materialize() or atlas.wind._load_gwa() to download GWA data."
            )

    atlas = MockAtlasNoRho(ds)

    # Should raise FileNotFoundError with actionable message
    with pytest.raises(FileNotFoundError, match=r"(materialize|_load_gwa)"):
        simulate_capacity_factors(
            atlas, force=True, weibull_height_mode="hub", air_density_mode="gwa"
        )


# ============================================================================
# Test 6b: Lazy loader success path (no air_density in dataset, load from disk)
# ============================================================================

def test_lazy_load_air_density_success_path():
    """
    Test that compute_air_density_at_height works via lazy loading when
    air_density is NOT in the dataset but load_air_density returns valid data.

    Uses monkeypatch to provide fake load_air_density that returns LINEAR data.
    Verifies LINEAR interpolation matches oracle without requiring self.data["air_density"].
    """
    from cleo.assess import compute_air_density_at_height

    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    # LINEAR air density: rho(h) = p0 + p1 * h
    # (NOT log-linear - air density uses linear interpolation in height)
    p0 = 1.3
    p1 = -0.001  # Density decreases with height

    def rho_oracle(h):
        return p0 + p1 * h

    # Build DataArrays for each height
    def make_rho_da(h):
        rho_val = rho_oracle(h)
        return xr.DataArray(
            np.full((len(y), len(x)), rho_val, dtype=np.float64),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="air_density",
        )

    # Dataset WITHOUT air_density
    template = xr.DataArray(
        np.ones((len(y), len(x)), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="template",
    )

    ds = xr.Dataset(
        data_vars={"template": template},
        coords={"wind_speed": wind_speed},
        attrs={"country": "TEST"},
    )

    class MockAtlasLazyLoad:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")

        def load_air_density(self, height):
            """Fake loader returning LINEAR air density."""
            if height not in [50, 100, 150, 200]:
                raise FileNotFoundError(f"No data for height={height}")
            return make_rho_da(float(height))

    atlas = MockAtlasLazyLoad(ds)

    # Verify air_density is NOT in dataset
    assert "air_density" not in atlas.data.data_vars

    # Call compute_air_density_at_height with target between available heights
    target_height = 80.0
    rho_hub = compute_air_density_at_height(atlas, target_height)

    # Oracle: LINEAR interpolation
    expected_rho = rho_oracle(target_height)

    # Check result
    assert rho_hub.shape == (len(y), len(x))
    np.testing.assert_allclose(
        rho_hub.values,
        expected_rho,
        rtol=0,
        atol=1e-12,
        err_msg="Lazy-loaded air density interpolation does not match linear oracle"
    )

    # Also test at another height
    target_height_2 = 120.0
    rho_hub_2 = compute_air_density_at_height(atlas, target_height_2)
    expected_rho_2 = rho_oracle(target_height_2)

    np.testing.assert_allclose(
        rho_hub_2.values,
        expected_rho_2,
        rtol=0,
        atol=1e-12,
        err_msg="Lazy-loaded air density interpolation at h=120m does not match linear oracle"
    )


# ============================================================================
# Test 7: Boundary behavior - out-of-range power curve returns 0
# ============================================================================

def test_density_corrected_power_curve_out_of_range_is_zero():
    r"""
    Test that density correction with high rho (c > 1) and right=0 boundary
    correctly returns 0 for out-of-range wind speeds.

    Setup:
    - Power curve defined only for [0, 20] m/s, ends at 1.0 (no explicit cut-out)
    - High air density: rho = 2.0 kg/m³ => c = (2.0/1.225)^(1/3) ≈ 1.178
    - Weibull PDF concentrated at high wind speeds (A=20, k=10 => mass near 20 m/s)
    - u_eq = u * c ≈ u * 1.178, so u=20 maps to u_eq≈23.6, beyond curve range

    With right=0 in np.interp, u_eq > 20 returns PC=0.
    With right=1 (wrong), would overestimate by treating all high speeds as full power.

    We verify:
    1. cf_density < cf_no_correction (since correction pushes effective speeds above range)
    2. Numerical match to independent oracle using numpy directly
    """
    # Grid setup (use 2x2 grid for CRS compatibility)
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)
    heights = np.array([50.0, 100.0, 150.0, 200.0])

    # Weibull params: A=20, k=10 => PDF concentrated near 20 m/s
    A = 20.0
    k = 10.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)

    # High air density in valid range: c = (2.0/1.225)^(1/3) ≈ 1.178
    rho_high = 2.0  # kg/m³, at upper bound of valid range
    air_density = np.full((4, 2, 2), rho_high)

    # Power curve: ramps from 0 to 1 over [0, 20], undefined beyond
    pc_u = np.array([0.0, 20.0])
    pc_p = np.array([0.0, 1.0])
    # Resample to atlas grid
    power_curve = np.interp(wind_speed, pc_u, pc_p, left=0.0, right=0.0)

    hub_height = 100.0
    turbine_name = "BoundaryTestTurbine"

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=power_curve, turbine_name=turbine_name,
        hub_height=hub_height,
    )

    # Run WITH density correction
    simulate_capacity_factors(
        atlas, force=True, weibull_height_mode="hub", air_density_mode="gwa"
    )
    # Get CF at first pixel (all pixels are identical due to uniform inputs)
    cf_density = float(atlas.data["capacity_factors"].sel(turbine=turbine_name).isel(y=0, x=0).values)

    # Run WITHOUT density correction (same atlas, reset)
    atlas2 = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=power_curve, turbine_name=turbine_name,
        hub_height=hub_height,
    )
    simulate_capacity_factors(
        atlas2, force=True, weibull_height_mode="hub", air_density_mode="none"
    )
    cf_no_density = float(atlas2.data["capacity_factors"].sel(turbine=turbine_name).isel(y=0, x=0).values)

    # Assertion 1: density correction should reduce CF (since c>1 pushes u_eq out of range)
    assert cf_density < cf_no_density, (
        f"With c=2.0 and right=0 boundary, cf_density ({cf_density:.6f}) "
        f"should be less than cf_no_density ({cf_no_density:.6f})"
    )

    # Assertion 2: numerical oracle using numpy directly
    # Build PDF at hub height (same Weibull everywhere)
    u = wind_speed.astype(np.float64)
    pdf_oracle = (k / A) * ((u / A) ** (k - 1)) * np.exp(-((u / A) ** k))
    pdf_oracle[0] = 0.0  # f(0) = 0 for Weibull

    # Power curve evaluation with density correction: c = (rho/rho0)^(1/3)
    c = (rho_high / RHO_0) ** (1.0 / 3.0)
    u_eq = u * c
    pc_corrected = np.interp(u_eq, u, power_curve, left=0.0, right=0.0)

    # Trapezoidal integration
    cf_oracle = float(np.trapezoid(pdf_oracle * pc_corrected, u))

    np.testing.assert_allclose(
        cf_density, cf_oracle, rtol=1e-10, atol=1e-12,
        err_msg="Density-corrected CF does not match independent oracle"
    )


# ============================================================================
# Test 8: Non-positive air density raises ValueError
# ============================================================================

def test_air_density_nonpositive_raises():
    """
    Test that air_density with non-positive values raises ValueError.

    Setting air_density to 0 or negative should trigger validation error
    with message mentioning "> 0".
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)

    # Invalid: set air_density to 0
    air_density = np.zeros((4, 2, 2))

    power_curve = np.where(wind_speed >= 6.0, 1.0, 0.0)

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=power_curve, turbine_name="T1",
        hub_height=100.0,
    )

    with pytest.raises(ValueError, match=r"must be > 0"):
        simulate_capacity_factors(
            atlas, force=True, weibull_height_mode="hub", air_density_mode="gwa"
        )

    # Also test negative values
    air_density_neg = np.full((4, 2, 2), -1.0)
    atlas_neg = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density_neg, heights=heights,
        power_curve=power_curve, turbine_name="T1",
        hub_height=100.0,
    )

    with pytest.raises(ValueError, match=r"must be > 0"):
        simulate_capacity_factors(
            atlas_neg, force=True, weibull_height_mode="hub", air_density_mode="gwa"
        )


# ============================================================================
# Test 9: Air density median outside plausible range raises ValueError
# ============================================================================

def test_air_density_units_median_out_of_range_raises():
    """
    Test that air_density with median outside [0.5, 2.0] kg/m³ raises ValueError.

    This catches unit errors (e.g., g/m³ instead of kg/m³).
    Setting air_density = 10.0 everywhere should fail median check.
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(2)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    weibull_A = np.full((4, 2, 2), A)
    weibull_k = np.full((4, 2, 2), k)

    # Invalid: air_density = 10 kg/m³ (physically implausible, suggests unit error)
    air_density = np.full((4, 2, 2), 10.0)

    power_curve = np.where(wind_speed >= 6.0, 1.0, 0.0)

    atlas = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density, heights=heights,
        power_curve=power_curve, turbine_name="T1",
        hub_height=100.0,
    )

    with pytest.raises(ValueError, match=r"outside plausible range"):
        simulate_capacity_factors(
            atlas, force=True, weibull_height_mode="hub", air_density_mode="gwa"
        )

    # Also test too low (e.g., 0.1 kg/m³)
    air_density_low = np.full((4, 2, 2), 0.1)
    atlas_low = create_mock_atlas_with_density(
        y=y, x=x, wind_speed=wind_speed,
        weibull_A=weibull_A, weibull_k=weibull_k,
        air_density=air_density_low, heights=heights,
        power_curve=power_curve, turbine_name="T1",
        hub_height=100.0,
    )

    with pytest.raises(ValueError, match=r"outside plausible range"):
        simulate_capacity_factors(
            atlas_low, force=True, weibull_height_mode="hub", air_density_mode="gwa"
        )
