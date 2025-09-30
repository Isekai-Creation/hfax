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

"""PEFT utils for `flax.linen`."""

# pylint: disable=g-importing-member,g-bad-import-order

# Module surgery utils
from hfax.peft._interceptors import Interceptor
from hfax.peft._interceptors import ModuleInterceptor

# LoRA utils
from hfax.peft._lora import LoRADense
from hfax.peft._lora import LoRADenseAdapter
from hfax.peft._lora import LoRAEinsum
from hfax.peft._lora import LoRAEinsumAdapter
from hfax.peft._lora import LoRADenseGeneral
from hfax.peft._lora import LoRADenseGeneralAdapter
from hfax.peft._tree_utils import fuse_params
from hfax.peft._tree_utils import merge_params
from hfax.peft._tree_utils import split_params
from hfax.peft._tree_utils import unfuse_params

# Quantization utils
from hfax.peft._quantization_utils import QuantizationMethod
from hfax.peft._quantization_utils import quantize
from hfax.peft._quantization import simulate_quantize
from hfax.peft._quantization import get_axis_to_reduce_from_einsum_str
from hfax.peft._quantization import SimulateQuantizedDense
from hfax.peft._quantization import SimulateQuantizedEinsum
from hfax.peft._quantization import IntDense
from hfax.peft._quantization import IntEinsum
