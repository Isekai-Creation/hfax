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

"""Gemma models."""

# pylint: disable=g-importing-member,g-import-not-at-top

from etils import epy as _epy


with _epy.lazy_api_imports(globals()):
  # ****************************************************************************
  # Gemma models
  # ****************************************************************************
  # Gemma 2
  from hfax.nn._gemma import Gemma2_2B
  from hfax.nn._gemma import Gemma2_9B
  from hfax.nn._gemma import Gemma2_27B
  # Gemma 3
  from hfax.nn._gemma import Gemma3_270M
  from hfax.nn._gemma import Gemma3_1B
  from hfax.nn._gemma import Gemma3_4B
  from hfax.nn._gemma import Gemma3_12B
  from hfax.nn._gemma import Gemma3_27B

  # ****************************************************************************
  # Wrapper (LoRA, quantization, DPO,...)
  # ****************************************************************************
  from hfax.nn._lora import LoRA
  from hfax.nn._quantization import QuantizationAwareWrapper
  from hfax.nn._quantization import IntWrapper
  from hfax.nn._policy import AnchoredPolicy
  from hfax.nn._transformer import Transformer

  # ****************************************************************************
  # Transformer building blocks
  # ****************************************************************************
  # Allow users to create their own transformers.
  # TODO(epot): Also expose the Vision encoder model as standalone.
  from hfax.nn._layers import Einsum
  from hfax.nn._layers import RMSNorm
  from hfax.nn._modules import Embedder
  from hfax.nn._modules import Attention
  from hfax.nn._modules import Block
  from hfax.nn._modules import FeedForward
  from hfax.nn._modules import AttentionType

  # Model inputs
  from hfax.nn._config import Cache

  # Model outputs
  from hfax.nn._transformer import Output
  from hfax.nn._policy import AnchoredPolicyOutput

  from hfax.nn import config
