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

from typing import Any

from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _gemma4 as gemma4_models
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _transformer as gt
from gemma.gm.utils import _cache_helper
import jax
import jax.numpy as jnp
import numpy as np
import pytest


BATCH_SIZE = 4
SEQ_LEN = 16


class _FakeMesh:

  def __init__(self, shape):
    self.shape = shape
    self.axis_names = tuple(shape)


class _FakeNamedSharding:

  def __init__(self, mesh, spec):
    self.mesh = mesh
    self.spec = spec


def _get_output(
    model: gt.Transformer, **kwargs
) -> tuple[gt.Output, Any]:

  def init_fn(**kwargs):
    out, params = model.init_with_output(jax.random.key(0), **kwargs)
    return out, params['params']

  return jax.eval_shape(init_fn, **kwargs)


@pytest.mark.parametrize(
    'model_cls',
    [
        gemma4_models.Gemma4_E2B,
        gemma4_models.Gemma4_E4B,
        gemma4_models.Gemma4_31B,
    ],
)
def test_transformer(model_cls: type[gt.Transformer]):
  model = model_cls()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, _ = _get_output(model, tokens=tokens)
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_text_only():
  model = gemma4_models.Gemma4_31B(text_only=True)  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, params = _get_output(model, tokens=tokens)
  assert 'vision_encoder' not in params
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_init_cache_local_window_shapes():
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      ),
      sliding_window_size=4,
      global_key_size=6,
      num_global_kv_heads=1,
  )

  cache = config.init_cache(
      batch_size=2,
      dtype=jnp.float32,
      cache_length=10,
      kv_cache_mode=_cache_helper.KVCacheMode.LOCAL_WINDOW,
  )

  assert cache['layer_0']['k'].shape == (2, 4, 1, 4)
  assert cache['layer_0']['v'].shape == (2, 4, 1, 4)
  assert cache['layer_0']['positions'].shape == (2, 4)
  assert cache['layer_0']['logical_index'].shape == (2, 4)
  assert cache['layer_0']['valid'].shape == (2, 4)
  np.testing.assert_array_equal(cache['layer_0']['logical_index'], -1)
  np.testing.assert_array_equal(cache['layer_0']['valid'], False)
  np.testing.assert_array_equal(cache['layer_0']['positions'], -(10**9))

  assert cache['layer_1']['k'].shape == (2, 10, 1, 6)
  assert 'logical_index' not in cache['layer_1']


def test_init_cache_local_window_uses_window_not_window_plus_one():
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(_modules.AttentionType.LOCAL_SLIDING,),
      sliding_window_size=4,
  )

  cache = config.init_cache(
      batch_size=1,
      dtype=jnp.float32,
      cache_length=10,
      kv_cache_mode=_cache_helper.KVCacheMode.LOCAL_WINDOW,
  )

  assert cache['layer_0']['k'].shape[1] == 4


def test_init_cache_local_window_does_not_shrink_global_without_global_key():
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(_modules.AttentionType.GLOBAL,),
      sliding_window_size=4,
  )

  cache = config.init_cache(
      batch_size=1,
      dtype=jnp.float32,
      cache_length=10,
      kv_cache_mode=_cache_helper.KVCacheMode.LOCAL_WINDOW,
  )

  assert cache['layer_0']['k'].shape == (1, 10, 1, 4)
  assert 'logical_index' not in cache['layer_0']


def test_cache_partition_spec_shards_divisible_kv_heads(monkeypatch):
  monkeypatch.setattr(_config, 'NamedSharding', _FakeNamedSharding)
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=4,
      head_dim=4,
      num_kv_heads=2,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      ),
      sliding_window_size=4,
      global_key_size=6,
      num_global_kv_heads=4,
  )

  spec = config.cache_partition_spec(_FakeMesh({'replicate': 2, 'tensor': 2}))

  assert tuple(spec['layer_0']['k'].spec) == (None, None, 'tensor', None)
  assert tuple(spec['layer_0']['v'].spec) == (None, None, 'tensor', None)
  assert tuple(spec['layer_0']['positions'].spec) == ()
  assert tuple(spec['layer_0']['end_index'].spec) == ()
  assert tuple(spec['layer_1']['k'].spec) == (None, None, 'tensor', None)


def test_cache_partition_spec_replicates_nondivisible_e2b(monkeypatch):
  monkeypatch.setattr(_config, 'NamedSharding', _FakeNamedSharding)
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=1,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(_modules.AttentionType.LOCAL_SLIDING,),
      sliding_window_size=4,
  )

  spec = config.cache_partition_spec(_FakeMesh({'replicate': 2, 'tensor': 2}))

  assert tuple(spec['layer_0']['k'].spec) == ()
  assert tuple(spec['layer_0']['v'].spec) == ()


def test_e4b_tp2_cache_partition_specs_shard_all_kv(monkeypatch):
  monkeypatch.setattr(_config, 'NamedSharding', _FakeNamedSharding)
  model = gemma4_models.Gemma4_E4B()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  config = model.config

  spec = config.cache_partition_spec(_FakeMesh({'replicate': 2, 'tensor': 2}))

  kv_specs = {
      tuple(layer_spec[name].spec)
      for layer_spec in spec.values()
      for name in ('k', 'v')
  }
  assert kv_specs == {(None, None, 'tensor', None)}


def test_cache_partition_spec_replicates_without_tensor_axis(monkeypatch):
  monkeypatch.setattr(_config, 'NamedSharding', _FakeNamedSharding)
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=4,
      head_dim=4,
      num_kv_heads=4,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(_modules.AttentionType.GLOBAL,),
  )

  spec = config.cache_partition_spec(_FakeMesh({'fsdp': 4}))

  assert tuple(spec['layer_0']['k'].spec) == ()
  assert tuple(spec['layer_0']['v'].spec) == ()


def test_cache_partition_spec_matches_local_window_tree(monkeypatch):
  monkeypatch.setattr(_config, 'NamedSharding', _FakeNamedSharding)
  config = _config.TransformerConfig(
      num_embed=32,
      embed_dim=8,
      hidden_dim=16,
      num_heads=2,
      head_dim=4,
      num_kv_heads=2,
      final_logit_softcap=None,
      use_post_attn_norm=False,
      use_post_ffw_norm=False,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      ),
      sliding_window_size=4,
  )

  legacy = config.cache_partition_spec(_FakeMesh({'tensor': 2}))
  local_window = config.cache_partition_spec(
      _FakeMesh({'tensor': 2}),
      kv_cache_mode=_cache_helper.KVCacheMode.LOCAL_WINDOW,
  )

  assert 'logical_index' not in legacy['layer_0']
  assert 'valid' not in legacy['layer_0']
  assert tuple(local_window['layer_0']['logical_index'].spec) == ()
  assert tuple(local_window['layer_0']['valid'].spec) == ()
  assert 'logical_index' not in local_window['layer_1']
  assert 'valid' not in local_window['layer_1']
