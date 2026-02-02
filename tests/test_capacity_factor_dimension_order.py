"""Test that capacity_factor works regardless of dimension order."""

import numpy as np
import xarray as xr

from cleo.assess import capacity_factor


def _create_test_inputs(dim_order):
    """Create test inputs with specified dimension order for weibull_pdf."""
    u = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    y = np.array([0.0, 1.0])
    x = np.array([0.0, 1.0, 2.0])

    # Create weibull_pdf with specified dimension order
    # Start with standard order then transpose
    pdf_data = np.random.rand(len(u), len(y), len(x)) * 0.1
    pdf_standard = xr.DataArray(
        pdf_data,
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u, "y": y, "x": x},
    )

    # Transpose to requested order
    pdf = pdf_standard.transpose(*dim_order)

    # Create wind_shear (spatial only)
    wind_shear = xr.DataArray(
        np.full((len(y), len(x)), 0.14),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )

    # Simple linear power curve
    p = np.array([0.0, 0.2, 0.6, 0.9, 1.0])

    return pdf, wind_shear, u, p


def test_capacity_factor_wind_speed_first():
    """capacity_factor should work with wind_speed as first dimension."""
    pdf, wind_shear, u, p = _create_test_inputs(["wind_speed", "y", "x"])

    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=100,
        h_reference=100,
    )

    assert cf.shape == wind_shear.shape
    assert np.all(np.isfinite(cf.values))


def test_capacity_factor_wind_speed_last():
    """capacity_factor should work with wind_speed as last dimension."""
    pdf, wind_shear, u, p = _create_test_inputs(["y", "x", "wind_speed"])

    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=100,
        h_reference=100,
    )

    assert cf.shape == wind_shear.shape
    assert np.all(np.isfinite(cf.values))


def test_capacity_factor_wind_speed_middle():
    """capacity_factor should work with wind_speed in middle."""
    pdf, wind_shear, u, p = _create_test_inputs(["y", "wind_speed", "x"])

    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=100,
        h_reference=100,
    )

    assert cf.shape == wind_shear.shape
    assert np.all(np.isfinite(cf.values))


def test_capacity_factor_same_result_regardless_of_order():
    """capacity_factor should produce same result regardless of dimension order."""
    # Use fixed seed for reproducibility
    np.random.seed(42)

    pdf1, wind_shear, u, p = _create_test_inputs(["wind_speed", "y", "x"])

    np.random.seed(42)
    pdf2, _, _, _ = _create_test_inputs(["y", "x", "wind_speed"])

    cf1 = capacity_factor(pdf1, wind_shear, u, p, h_turbine=100)
    cf2 = capacity_factor(pdf2, wind_shear, u, p, h_turbine=100)

    np.testing.assert_allclose(cf1.values, cf2.values, rtol=1e-10)
