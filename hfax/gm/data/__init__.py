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

"""Data pipeline ops."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top

with _epy.lazy_api_imports(globals()):

  # pylint: disable=g-importing-member,g-bad-import-order

  # Data sources
  from hfax.data._parquet import Parquet

  # `transforms=`
  from hfax.data._tasks import ContrastiveTask
  from hfax.data._tasks import Seq2SeqTask
  from hfax.data._transforms import AddSeq2SeqFields
  from hfax.data._transforms import DecodeBytes
  from hfax.data._transforms import FormatText
  from hfax.data._transforms import MapInts
  from hfax.data._transforms import Pad
  from hfax.data._transforms import Tokenize

  # Functional API
  from hfax.data._functional import make_seq2seq_fields
  from hfax.data._functional import pad
