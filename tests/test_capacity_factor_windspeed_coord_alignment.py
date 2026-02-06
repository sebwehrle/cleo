"""Oracle-driven test for capacity_factor wind-speed coordinate alignment.

This test proves whether capacity_factor uses label alignment or positional
indexing when accessing the weibull_pdf DataArray.

The test constructs a PDF with wind_speed coordinate reversed from u_power_curve.
If capacity_factor uses positional indexing (bug), it will compute the wrong
integral. If it properly aligns by wind-speed label, it will compute correctly.
"""

import numpy as np
import xarray as xr

from cleo.assess import capacity_factor


def test_capacity_factor_windspeed_coord_alignment():
    """
    Verify capacity_factor aligns PDF by wind-speed label, not positional index.

    Setup:
    - u_power_curve: [0, 5, 10, 15, 20] (ascending)
    - weibull_pdf wind_speed coord: [20, 15, 10, 5, 0] (descending)
    - wind_shear = 0 (so s = 1, no hub-height scaling)
    - p_power_curve: monotone increasing [0, 0.25, 0.5, 0.75, 1.0]

    Key: Use an ASYMMETRIC PDF to break symmetry and distinguish alignments.
    PDF is nonzero only for u < 12 (left-skewed).

    Oracle:
    - Align PDF values to match u_power_curve order by label
    - Integrate: CF = integral(pdf(u) * p(u) du)

    If production uses positional indexing:
    - PDF storage order [pdf(20), pdf(15), pdf(10), pdf(5), pdf(0)]
    - Gets paired with p_power_curve by position, not by wind speed label
    - This produces a different (incorrect) result

    The test FAILS if capacity_factor uses positional indexing.
    """
    # Wind speed coordinates
    u_ascending = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    u_descending = u_ascending[::-1].copy()  # [20, 15, 10, 5, 0]

    # Linear power curve: p(u) = u / 20
    # p = [0, 0.25, 0.5, 0.75, 1.0] for u = [0, 5, 10, 15, 20]
    p_power_curve = u_ascending / 20.0

    # Create an ASYMMETRIC PDF: uniform for u in [0, 12), zero elsewhere
    # This breaks the symmetry that made the previous test degenerate
    def asymmetric_pdf(u):
        # pdf = 1/12 for u < 12, 0 otherwise
        # Integrates to 1 over [0, 12]
        return np.where(u < 12, 1.0 / 12.0, 0.0)

    # PDF values at ascending wind speeds [0, 5, 10, 15, 20]
    pdf_at_ascending = asymmetric_pdf(u_ascending)
    # = [1/12, 1/12, 1/12, 0, 0]

    # PDF values at descending wind speeds [20, 15, 10, 5, 0]
    pdf_at_descending = asymmetric_pdf(u_descending)
    # = [0, 0, 1/12, 1/12, 1/12]

    # Verify PDFs have correct values and are DIFFERENT
    expected_ascending = np.array([1/12, 1/12, 1/12, 0, 0])
    expected_descending = np.array([0, 0, 1/12, 1/12, 1/12])
    assert np.allclose(pdf_at_ascending, expected_ascending)
    assert np.allclose(pdf_at_descending, expected_descending)
    assert not np.allclose(pdf_at_ascending, pdf_at_descending), \
        "Asymmetric PDF should give different values for ascending vs descending"

    # Create weibull_pdf DataArray with DESCENDING wind_speed coordinate
    # coords: [20, 15, 10, 5, 0]
    # values: [pdf(20), pdf(15), pdf(10), pdf(5), pdf(0)] = [0, 0, 1/12, 1/12, 1/12]
    weibull_pdf = xr.DataArray(
        data=pdf_at_descending[:, None, None],  # shape (5, 1, 1)
        dims=["wind_speed", "y", "x"],
        coords={
            "wind_speed": u_descending,
            "y": [0],
            "x": [0],
        },
    )

    # Verify coordinate assignment
    assert list(weibull_pdf.coords["wind_speed"].values) == [20.0, 15.0, 10.0, 5.0, 0.0]

    # Verify values are correctly associated with coords
    assert np.isclose(weibull_pdf.sel(wind_speed=0).values.item(), 1/12)
    assert np.isclose(weibull_pdf.sel(wind_speed=10).values.item(), 1/12)
    assert np.isclose(weibull_pdf.sel(wind_speed=20).values.item(), 0)

    # wind_shear = 0 means s = (h/h_ref)^0 = 1, no scaling
    wind_shear = xr.DataArray(
        data=np.array([[0.0]]),
        dims=["y", "x"],
        coords={"y": [0], "x": [0]},
    )

    # Oracle: Compute CF by explicitly aligning PDF to u_power_curve order
    # For each u in u_ascending, find the PDF value at that wind speed label
    pdf_aligned = np.array([
        float(weibull_pdf.sel(wind_speed=u, method=None).values.flat[0])
        for u in u_ascending
    ])
    # pdf_aligned = [pdf(0), pdf(5), pdf(10), pdf(15), pdf(20)]
    #             = [1/12, 1/12, 1/12, 0, 0]

    # Verify alignment worked correctly
    assert np.allclose(pdf_aligned, expected_ascending)

    # Oracle CF: integral(pdf(u) * p(u) du) using trapezoidal rule
    integrand_aligned = pdf_aligned * p_power_curve
    # = [1/12 * 0, 1/12 * 0.25, 1/12 * 0.5, 0 * 0.75, 0 * 1.0]
    # = [0, 1/48, 1/24, 0, 0]
    CF_oracle = np.trapezoid(integrand_aligned, u_ascending)

    # Sanity check: CF should be positive and less than 1
    assert 0 < CF_oracle < 1, f"Oracle CF={CF_oracle} should be in (0,1)"

    # What would positional indexing compute?
    # It would use pdf_at_descending (storage order) with p_power_curve
    # Storage order: [pdf(20), pdf(15), pdf(10), pdf(5), pdf(0)] = [0, 0, 1/12, 1/12, 1/12]
    # Paired with u_ascending positions: p(0), p(5), p(10), p(15), p(20)
    integrand_positional = pdf_at_descending * p_power_curve
    # = [0 * 0, 0 * 0.25, 1/12 * 0.5, 1/12 * 0.75, 1/12 * 1.0]
    # = [0, 0, 1/24, 1/16, 1/12]
    CF_positional = np.trapezoid(integrand_positional, u_ascending)

    # The two results MUST be different (this validates the test design)
    assert not np.isclose(CF_oracle, CF_positional), (
        f"Oracle CF={CF_oracle} should differ from positional CF={CF_positional}. "
        "Test setup is degenerate."
    )

    # Call capacity_factor
    result = capacity_factor(
        weibull_pdf,
        wind_shear,
        u_ascending,  # u_power_curve in ascending order
        p_power_curve,
        h_turbine=100.0,
        h_reference=100.0,  # Same as h_turbine, so s=1
        correction_factor=1.0,
    )

    # Extract scalar result
    cf_result = float(result.values.flat[0])

    # The test PASSES if result matches oracle (label alignment)
    # The test FAILS if result matches positional indexing (bug)
    assert np.isclose(cf_result, CF_oracle, atol=1e-10), (
        f"capacity_factor result {cf_result:.6f} does not match "
        f"label-aligned oracle {CF_oracle:.6f}. "
        f"If result ≈ positional={CF_positional:.6f}, "
        "then capacity_factor uses positional indexing (BUG)."
    )
