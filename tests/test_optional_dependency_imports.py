"""Tests that falsify optional-dependency import hygiene.

These tests demonstrate that the current cleo package does NOT properly
defer optional dependency imports - importing cleo modules fails immediately
when rioxarray/pint/xrspatial are unavailable.
"""
import builtins
import sys
import pytest


@pytest.fixture
def block_imports(monkeypatch):
    """Factory fixture to block specified module imports."""
    def _block(blocked_modules):
        # Remove any cached imports of blocked modules
        modules_to_remove = [
            name for name in list(sys.modules.keys())
            if any(name == b or name.startswith(b + ".") for b in blocked_modules)
        ]
        for name in modules_to_remove:
            sys.modules.pop(name, None)

        # Also remove cleo modules to force re-import
        cleo_modules = [name for name in list(sys.modules.keys()) if name.startswith("cleo")]
        for name in cleo_modules:
            sys.modules.pop(name, None)

        real_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            for blocked in blocked_modules:
                if name == blocked or name.startswith(blocked + "."):
                    raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

    return _block


def test_import_cleo_fails_without_xrspatial(block_imports):
    """Falsify: import cleo does NOT succeed when xrspatial is blocked.

    This test proves that cleo lacks lazy imports - xrspatial is imported
    at module load time in cleo.classes.
    """
    block_imports(["xrspatial"])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import cleo

    assert "xrspatial" in str(exc_info.value)


def test_import_cleo_fails_without_rioxarray(block_imports):
    """Falsify: import cleo does NOT succeed when rioxarray is blocked.

    This test proves that cleo lacks lazy imports - rioxarray is imported
    at module load time in cleo.class_helpers.
    """
    block_imports(["rioxarray"])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import cleo

    assert "rioxarray" in str(exc_info.value)


def test_import_cleo_fails_without_pint(block_imports):
    """Falsify: import cleo does NOT succeed when pint is blocked.

    This test proves that cleo lacks lazy imports - pint is imported
    at module load time in cleo.utils.
    """
    block_imports(["pint"])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import cleo

    assert "pint" in str(exc_info.value)


def test_import_cleo_assess_fails_without_rioxarray(block_imports):
    """Falsify: import cleo.assess does NOT succeed when rioxarray is blocked.

    This test proves that cleo.assess lacks lazy imports - importing it
    triggers cleo/__init__.py which imports from cleo.classes, which imports
    from cleo.class_helpers, which imports rioxarray at module level.
    """
    block_imports(["rioxarray"])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import cleo.assess

    assert "rioxarray" in str(exc_info.value)


def test_import_cleo_utils_fails_without_pint(block_imports):
    """Falsify: import cleo.utils does NOT succeed when pint is blocked.

    This test proves that cleo.utils lacks lazy imports - pint is imported
    at module level via 'from pint import UnitRegistry'.
    """
    block_imports(["pint"])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import cleo.utils

    assert "pint" in str(exc_info.value)
