"""Landscape materialization policies.

This module is a facade that re-exports functions from private submodules.
The implementation is split into:
- _landscape_vector.py: Vector source handling helpers
- _landscape_sources.py: Source registration and resolution helpers
- _landscape_core.py: Core materialize_landscape function
- _landscape_api.py: Public API functions

All public and internal names are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Re-export from _landscape_vector (internal helpers used by tests)
from cleo.unification.materializers._landscape_vector import (
    _canonical_vector_source_artifact,
    _load_vector_shape,
    _vector_semantic_hash,
    _vector_semantic_payload,
    _vector_values_for_column,
)

# Re-export from _landscape_sources (internal helpers used by tests)
from cleo.unification.materializers._landscape_sources import (
    _current_landscape_source_fingerprint,
    _prepare_vector_landscape_variable_data,
    _register_landscape_source_entry,
    _resolve_landscape_source_for_variable,
)

# Re-export from _landscape_core
from cleo.unification.materializers._landscape_core import materialize_landscape

# Re-export from _landscape_api (public API)
from cleo.unification.materializers._landscape_api import (
    compute_air_density_correction,
    materialize_landscape_computed_variables,
    materialize_landscape_variable,
    prepare_landscape_variable_data,
    register_landscape_source,
    register_landscape_vector_source,
)

__all__ = [
    # Public API
    "materialize_landscape",
    "register_landscape_source",
    "register_landscape_vector_source",
    "prepare_landscape_variable_data",
    "materialize_landscape_computed_variables",
    "materialize_landscape_variable",
    "compute_air_density_correction",
    # Internal helpers (for test compatibility)
    "_load_vector_shape",
    "_vector_values_for_column",
    "_vector_semantic_payload",
    "_vector_semantic_hash",
    "_canonical_vector_source_artifact",
    "_register_landscape_source_entry",
    "_resolve_landscape_source_for_variable",
    "_current_landscape_source_fingerprint",
    "_prepare_vector_landscape_variable_data",
]
