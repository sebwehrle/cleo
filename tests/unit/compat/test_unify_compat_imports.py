"""Compatibility import tests for Phase 1 unification extraction."""

from __future__ import annotations


def test_legacy_imports_from_unify_still_work() -> None:
    from cleo.unify import GWA_HEIGHTS, Unifier, hash_grid_id, hash_inputs_id, init_manifest

    assert GWA_HEIGHTS == [10, 50, 100, 150, 200]
    assert Unifier is not None
    assert callable(hash_grid_id)
    assert callable(hash_inputs_id)
    assert callable(init_manifest)


def test_new_fingerprint_module_imports_work() -> None:
    from cleo.unification.fingerprint import (
        fingerprint_file,
        get_git_info,
        hash_grid_id,
        hash_inputs_id,
    )

    assert callable(get_git_info)
    assert callable(hash_grid_id)
    assert callable(hash_inputs_id)
    assert callable(fingerprint_file)


def test_new_manifest_module_imports_work() -> None:
    from cleo.unification.manifest import (
        init_manifest,
        write_manifest_sources,
        write_manifest_variables,
    )

    assert callable(init_manifest)
    assert callable(write_manifest_sources)
    assert callable(write_manifest_variables)


def test_unification_package_reexports_work() -> None:
    from cleo.unification import (
        Unifier,
        hash_grid_id,
        hash_inputs_id,
        init_manifest,
        write_manifest_sources,
        write_manifest_variables,
    )

    assert Unifier is not None
    assert callable(hash_grid_id)
    assert callable(hash_inputs_id)
    assert callable(init_manifest)
    assert callable(write_manifest_sources)
    assert callable(write_manifest_variables)


def test_legacy_private_nuts_imports_from_unify_still_work() -> None:
    from cleo.unify import _read_nuts_region_catalog, _read_vector_file

    assert callable(_read_vector_file)
    assert callable(_read_nuts_region_catalog)


def test_new_nuts_module_imports_work() -> None:
    from cleo.unification.nuts_io import _read_nuts_region_catalog, _read_vector_file

    assert callable(_read_vector_file)
    assert callable(_read_nuts_region_catalog)


def test_new_gwa_module_imports_work() -> None:
    from cleo.unification.gwa_io import GWA_HEIGHTS, _required_gwa_files

    assert GWA_HEIGHTS == [10, 50, 100, 150, 200]
    assert callable(_required_gwa_files)
