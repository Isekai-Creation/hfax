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

"""Text processing utilities."""


from etils import epy as _epy

# pylint: disable=g-import-not-at-top,g-importing-member

with _epy.lazy_api_imports(globals()):
  # Tokenizers
  from hfax.text._tokenizer import Gemma2Tokenizer
  from hfax.text._tokenizer import Gemma3Tokenizer
  from hfax.text._tokenizer import Tokenizer
  from hfax.text._tokenizer import SpecialTokens

  # Samplers
  from hfax.text._sampler import Sampler
  from hfax.text._chat_sampler import ChatSampler
  from hfax.text._tool_sampler import ToolSampler

  # Sampling methods
  # TODO(mblondel): Add nucleus sampling
  from hfax.text._sampling import SamplingMethod
  from hfax.text._sampling import Greedy
  from hfax.text._sampling import RandomSampling
  from hfax.text._sampling import TopkSampling
  from hfax.text._sampling import TopPSampling

  # Other utils
  # from hfax.text import _template as template
