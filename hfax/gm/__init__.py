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

"""Kauldron API for Hfax."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top

with _epy.lazy_api_imports(globals()):
  # API match the `kd` namespace.
  from hfax.gm import ckpts
  from hfax.gm import data
  from hfax.gm import evals
  from hfax.gm import losses
  from hfax.gm import math
  from hfax.gm import nn
  from hfax.gm import text
  from hfax.gm import tools
  from hfax.gm import sharding
  from hfax.gm import testing
  from hfax.gm import typing
  from hfax import peft
