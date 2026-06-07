# Copyright 2026 DeepMind Technologies Limited.
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

"""Numerical equivalence test for `KVCacheMode.LOCAL_WINDOW`.

The whole point of the local-window cache is that it must produce the *same*
decode behavior as the legacy full-length cache: a `LOCAL_SLIDING` layer never
attends outside its window, so ring-buffering those layers should be a pure
memory optimization with no effect on the math.

`examples/cache_local_window_test.py` checks this end-to-end, but it needs a
TPU and real checkpoints, so it cannot run in CI. This test reproduces the
same guarantee on CPU with a tiny randomly-initialized Gemma 4 model: it runs
the real `prefill` -> `SamplerLoop` decode path in both modes and asserts the
per-step logits match.

The regimes exercised are the ones where the ring buffer actually does
something:
  * prompt length P > window W, so prefill compaction must keep "the last W
    valid logical tokens" and the decode ring wraps repeatedly;
  * a padded batch where rows have different real lengths, which is where
    "last W *valid* tokens" differs from "last W *physical* slots" (the bug
    the implementation is specifically written to avoid).

A negative control (manually replacing the compaction with the wrong
"last W physical" strategy) was used during development to confirm this test
diverges by ~0.26 in logits on the short padded row, i.e. the test is
sensitive to the bug it guards.
"""

from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _transformer as _gemma4_transformer
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.utils import _cache_helper
from gemma.gm.utils import _types
import jax
import jax.numpy as jnp
import numpy as np
import pytest


_WINDOW = 4
_CACHE_LENGTH = 16
_PAD_LENGTH = 8
_NUM_DECODE_STEPS = 6
_SPECIAL_TOKENS = _tokenizer._Gemma3SpecialTokens  # pylint: disable=protected-access


def _build_model() -> _gemma4_transformer.Transformer:
  """A tiny Gemma 4 model with the local/global hybrid attention pattern.

  Layer 0 is `LOCAL_SLIDING` (as in every real Gemma 4 variant), so this also
  covers the sampler-loop stop-condition fix that must not key off layer 0's
  physical shape.
  """
  config = _config.TransformerConfig(
      num_embed=64,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=1,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      ),
      sliding_window_size=_WINDOW,
      global_key_size=8,
      num_global_kv_heads=1,
      # RoPE proportions must be set for the forward pass to run.
      local_rope_proportion=1.0,
      global_rope_proportion=1.0,
  )
  return _gemma4_transformer.Transformer(config=config)


def _decode_logits(
    *,
    model: _gemma4_transformer.Transformer,
    params,
    tokens: jax.Array,
    kv_cache_mode: _cache_helper.KVCacheMode,
) -> np.ndarray:
  """Prefill then greedily decode, returning the per-step logits `[B, N, V]`.

  Both modes are driven through the identical real code path
  (`_prefill.prefill` + `SamplerLoop._sample_step`); only `kv_cache_mode`
  differs. We re-run `model.apply` at each step to capture the logits that the
  sampler loop itself discards.
  """
  input_ = _types.Input(  # pylint: disable=redefined-builtin
      text=tokens,
      images=None,
      config=_types.InputConfig(
          support_images=False,
          num_tokens_per_image=0,
          special_tokens=_SPECIAL_TOKENS,
      ),
  )
  state = _prefill.prefill(
      model=model,
      params=params,
      input=input_,
      last_state=None,
      cache_length=_CACHE_LENGTH,
      max_out_length=_NUM_DECODE_STEPS,
      pad_length=(_PAD_LENGTH,),
      rng=jax.random.PRNGKey(0),
      sharding=None,
      kv_cache_mode=kv_cache_mode,
  )
  loop = _sampler_loop.SamplerLoop(
      model=model,
      end_tokens=(_SPECIAL_TOKENS.EOS,),
      forbidden_tokens=None,
      sampling=_sampling.Greedy(),
      cache_length=_CACHE_LENGTH,
      special_tokens=_SPECIAL_TOKENS,
  )

  per_step_logits = []
  for _ in range(_NUM_DECODE_STEPS):
    out = model.apply(
        {'params': params},
        tokens=state.last_token[..., None],
        cache=state.cache,
        positions=state.last_token_pos[..., None],
        attention_mask=state.attention_mask_for_step[:, None, :],
    )
    per_step_logits.append(np.asarray(out.logits[:, 0, :]))
    state = loop._sample_step(state, params=params)  # pylint: disable=protected-access

  return np.stack(per_step_logits, axis=1)


def _assert_modes_match(tokens: jax.Array):
  model = _build_model()
  params = model.init(
      jax.random.key(0), tokens=jnp.ones((1, _PAD_LENGTH), dtype=jnp.int32)
  )['params']

  legacy = _decode_logits(
      model=model,
      params=params,
      tokens=tokens,
      kv_cache_mode=_cache_helper.KVCacheMode.LEGACY,
  )
  local_window = _decode_logits(
      model=model,
      params=params,
      tokens=tokens,
      kv_cache_mode=_cache_helper.KVCacheMode.LOCAL_WINDOW,
  )

  # Sanity check that the logits are non-degenerate, so the equivalence below
  # is a meaningful comparison and not "all zeros == all zeros".
  assert np.abs(legacy).max() > 1e-3

  # The local-sliding mask discards everything outside the window in both
  # modes, so the result must be identical up to floating-point reduction
  # order. On CPU float32 this is exact, but we allow a small tolerance.
  np.testing.assert_allclose(local_window, legacy, atol=1e-4, rtol=0)


@pytest.mark.parametrize('prompt_len', [3, 4, 8])  # < W, == W, > W (ring wraps)
def test_local_window_matches_legacy_single_row(prompt_len: int):
  tokens = jnp.arange(3, 3 + 2 * prompt_len, 2, dtype=jnp.int32)[None, :]
  _assert_modes_match(tokens)


def test_local_window_matches_legacy_padded_batch():
  # Row 0 has 8 real tokens (wraps the window); row 1 has 5 real tokens then
  # padding. This is the case where "last W valid logical" must beat
  # "last W physical": row 1's live local context must survive row 0's longer
  # padded prefill timeline.
  tokens = jnp.asarray(
      [
          [3, 5, 7, 9, 11, 13, 15, 17],
          [3, 5, 7, 9, 11, 0, 0, 0],
      ],
      dtype=jnp.int32,
  )
  _assert_modes_match(tokens)
