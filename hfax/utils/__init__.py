"""Utilities namespace for both top-level and GM utils.

This package ensures that imports like ``hfax.utils.logging`` resolve to the
top-level implementation in this directory, while still allowing existing
imports such as ``from hfax.utils import _dtype_params`` to work by extending
the package search path to include ``hfax/gm/utils``.

Rationale: ``hfax/__init__.py`` appends ``hfax/gm`` to the top-level package
``__path__``. When ``hfax.utils`` had no ``__init__`` file, Python preferred the
concrete package at ``hfax/gm/utils`` over the implicit namespace at
``hfax/utils``. Creating this ``__init__`` and extending our own ``__path__``
restores the intended resolution order without breaking existing imports.
"""

from __future__ import annotations

from pathlib import Path

# Extend this package's search path to also include gm/utils so submodules like
# ``_dtype_params`` remain importable via ``hfax.utils._dtype_params``.
_PKG_DIR = Path(__file__).resolve().parent
_GM_UTILS_DIR = _PKG_DIR.parent / "gm" / "utils"

if _GM_UTILS_DIR.exists():
    try:
        __path__.append(str(_GM_UTILS_DIR))  # type: ignore[name-defined]
    except NameError:
        # __path__ is defined for packages; guard for static analyzers.
        pass

__all__ = ()

