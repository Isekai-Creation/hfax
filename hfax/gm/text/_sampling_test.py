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

import hfax
import jax.numpy
import numpy as np


def test_greedy_sampling():
  sampling = hfax.text.Greedy()
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 3
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_random_sampling():
  sampling = hfax.text.RandomSampling()
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 3
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_topk_sampling():
  sampling = hfax.text.TopkSampling(k=3)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_topp_sampling():
  sampling = hfax.text.TopPSampling(p=0.9)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_topp_sampling_with_skewed_logits():
  sampling = hfax.text.TopPSampling(p=0.6)
  rng = jax.random.PRNGKey(2)
  # Probabilities after softmax: [0.64, 0.23, 0.09, 0.03, 0.01].
  logits = jax.numpy.array([
      [5.0, 4.0, 3.0, 2.0, 1.0],
  ])
  tokens = sampling.get_next_tokens(logits, rng)
  np.testing.assert_array_equal(tokens, [0])


def test_top1_sampling_matches_greedy_sampling():
  greedy = hfax.text.Greedy()
  top1_sampling = hfax.text.TopkSampling(k=1)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens_greedy = greedy.get_next_tokens(logits, rng)
  tokens_top1 = top1_sampling.get_next_tokens(logits, rng)
  np.testing.assert_array_equal(tokens_greedy, tokens_top1)

