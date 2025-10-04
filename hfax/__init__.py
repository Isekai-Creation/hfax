# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the public hfax API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = '3.2.1'

_PACKAGE_DIR = Path(__file__).resolve().parent
_GM_DIR = _PACKAGE_DIR / 'gm'
if _GM_DIR.exists():
  if __path__ is not None and str(_GM_DIR) not in __path__:
    __path__.append(str(_GM_DIR))

# The historical releases exposed modules under the ``hfax.hfax`` namespace.
# Some internal imports still rely on that layout, so we register an alias to
# keep backwards compatibility when working directly from the source tree.
sys.modules.setdefault(f'{__name__}.hfax', sys.modules[__name__])

# Map attribute names to their backing modules. Keeping the mapping here avoids
# importing any heavyweight dependencies until the attribute is actually used.
_API_MODULES: Dict[str, str] = {
    # Core Gemma modules
    'ckpts': 'hfax.gm.ckpts',
    'data': 'hfax.gm.data',
    'evals': 'hfax.gm.evals',
    'losses': 'hfax.gm.losses',
    'math': 'hfax.gm.math',
    'nn': 'hfax.gm.nn',
    'sharding': 'hfax.gm.sharding',
    'testing': 'hfax.gm.testing',
    'text': 'hfax.gm.text',
    'tools': 'hfax.gm.tools',
    'typing': 'hfax.gm.typing',
    # Namespaces exposed at the top level
    'gm': 'hfax.gm',
    'multimodal': 'hfax.multimodal',
    'peft': 'hfax.peft',
    'research': 'hfax.research',
    'metrics': 'hfax.metrics',
    'profiler': 'hfax.profiler',
}


def __getattr__(name: str):  # pylint: disable=invalid-name
  """Lazily import submodules on first attribute access."""
  try:
    module = importlib.import_module(_API_MODULES[name])
  except KeyError as exc:  # pragma: no cover - defensive guard
    raise AttributeError(
        f"module '{__name__}' has no attribute {name!r}"
    ) from exc

  setattr(sys.modules[__name__], name, module)
  sys.modules.setdefault(f'{__name__}.{name}', module)
  sys.modules.setdefault(f'{__name__}.hfax.{name}', module)
  return module


def __dir__() -> list[str]:  # pragma: no cover - small helper
  return sorted(list(globals().keys()) + list(_API_MODULES.keys()))


__all__ = tuple(sorted(_API_MODULES))
