"""Static boundary guardrails for atlas policy modules.

Uses shared utilities from tools.arch_check.
"""

from __future__ import annotations

from tools.arch_check import check_policy_module_boundaries


def test_atlas_policies_boundary_guardrails() -> None:
    """Atlas policy modules must not import atlas or perform direct I/O."""
    violations = check_policy_module_boundaries()
    assert violations == [], "atlas_policies boundary violated:\n" + "\n".join(violations)
