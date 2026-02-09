"""Guardrail tests for raster assignments.

Purpose: prevent regrowth of raster-alignment issues by ensuring:
1. Known raster vars are assigned via _set_var, not directly
2. Direct self.data["key"] = assignments are only in allowed locations
"""

import re
from pathlib import Path

import pytest

from tests.helpers.paths import cleo_path


def get_cleo_source_files():
    """Get all Python source files in cleo package (excluding tests)."""
    return list(cleo_path().glob("*.py"))


def read_source(path: Path) -> str:
    """Read source file, return empty string if unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


class TestRasterAssignmentGuardrails:
    """Static analysis tests to enforce _set_var usage for rasters."""

    # Known raster variable names that MUST go through _set_var
    RASTER_VAR_DENYLIST = {
        "air_density_correction",
        "mean_wind_speed",
        "wind_shear",
        "weibull_pdf",
        "capacity_factors",
        "lcoe",
        "optimal_power",
        "optimal_energy",
        "min_lcoe",
    }

    # Pattern for direct self.data["key"] = assignment
    DIRECT_ASSIGN_PATTERN = re.compile(
        r'self\.data\s*\[\s*["\']([^"\']+)["\']\s*\]\s*='
    )

    def test_no_direct_raster_assignment_outside_set_var(self):
        """
        Ensure raster vars are not assigned directly outside _set_var.

        Allowed exceptions:
        - "power_curve" (non-raster, dims are wind_speed/turbine)
        - "template" inside _set_var implementation only
        - Assignments inside _set_var method itself
        """
        violations = []

        for path in get_cleo_source_files():
            source = read_source(path)
            if not source:
                continue

            lines = source.splitlines()
            in_set_var_method = False

            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track if we're inside _set_var method
                if "def _set_var(" in stripped:
                    in_set_var_method = True
                elif in_set_var_method and stripped.startswith("def "):
                    in_set_var_method = False

                # Skip if inside _set_var
                if in_set_var_method:
                    continue

                # Skip comments
                code_part = line.split("#")[0]
                if not code_part.strip():
                    continue

                # Find direct assignments
                match = self.DIRECT_ASSIGN_PATTERN.search(code_part)
                if match:
                    var_name = match.group(1)

                    # Allow power_curve (non-raster)
                    if var_name == "power_curve":
                        continue

                    # Check if it's a denylisted raster var
                    if var_name in self.RASTER_VAR_DENYLIST:
                        violations.append(
                            f"{path.name}:{lineno}: direct assignment to raster var '{var_name}': "
                            f"{stripped[:80]}..."
                        )

        if violations:
            msg = (
                "Direct raster variable assignments found outside _set_var. "
                "Use self._set_var(name, da) instead:\n" + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_no_xr_merge_for_power_curve(self):
        """Ensure xr.merge is not used for power_curve assignment."""
        violations = []

        for path in get_cleo_source_files():
            source = read_source(path)
            if not source:
                continue

            # Look for xr.merge near power_curve context
            if "power_curve" in source and "xr.merge" in source:
                lines = source.splitlines()
                for lineno, line in enumerate(lines, 1):
                    if "xr.merge" in line and "power_curve" in line:
                        violations.append(
                            f"{path.name}:{lineno}: xr.merge with power_curve: {line.strip()[:80]}"
                        )

        if violations:
            msg = (
                "xr.merge found for power_curve. Use direct assignment instead:\n"
                + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_rasterize_uses_set_var(self):
        """Ensure rasterize method uses _set_var, not direct assignment."""
        classes_path = cleo_path("classes.py")
        source = read_source(classes_path)

        if not source:
            pytest.skip("Could not read classes.py")

        # Find rasterize method and check for _set_var usage
        in_rasterize = False
        found_set_var = False
        found_direct_assign = False

        for line in source.splitlines():
            if "def rasterize(" in line:
                in_rasterize = True
            elif in_rasterize and line.strip().startswith("def "):
                break
            elif in_rasterize:
                if "_set_var(" in line:
                    found_set_var = True
                if re.search(r'self\.data\s*\[\s*name_to_assign\s*\]\s*=', line):
                    found_direct_assign = True

        assert found_set_var, "rasterize method should use _set_var"
        assert not found_direct_assign, "rasterize should not use direct self.data[name_to_assign] = "


class TestEnforceExactGridExists:
    """Verify enforce_exact_grid helper exists and is used."""

    def test_enforce_exact_grid_in_spatial(self):
        """Verify enforce_exact_grid is defined in spatial.py."""
        from cleo.spatial import enforce_exact_grid

        assert callable(enforce_exact_grid)

    def test_set_var_uses_enforce_exact_grid(self):
        """Verify _set_var imports and uses enforce_exact_grid."""
        classes_path = cleo_path("classes.py")
        source = read_source(classes_path)

        assert "enforce_exact_grid" in source, "_set_var should import enforce_exact_grid"

    def test_coords_equal_in_spatial(self):
        """Verify coords_equal helper exists in spatial.py."""
        from cleo.spatial import coords_equal

        assert callable(coords_equal)
