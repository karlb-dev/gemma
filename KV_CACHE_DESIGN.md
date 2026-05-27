# Gemma 4 Local-Window KV Cache

**Author:** Karl Burtram <kburtram@live.com>
**Status:** Code review for the change on this branch
**Target:** `github.com/google-deepmind/gemma`
**Date:** 2026-05-27

This document describes the change on this branch. It is not a proposal —
the code is here, the tests are here, and the measurements come from
runs against this branch on TPU v5e-4, v5e-8, and v5e-16 hardware. The
goal is to give a reviewer enough detail to make a decision: what the
change does, how it is gated, what it costs, where it has been
validated, and where it has not.

---

## 1. Summary

This branch adds an opt-in KV-cache layout for Gemma 4
(`KVCacheMode.LOCAL_WINDOW`) that caps each `LOCAL_SLIDING` attention
layer's persistent decode cache at `sliding_window_size` slots in a
per-row ring buffer, while keeping `GLOBAL` layers at the full
`cache_length`. The default behavior is unchanged: `KVCacheMode.LEGACY`
is the default at every call site, so existing users see the same
allocation, the same shapes, and the same sampler-loop semantics as
before.

The change is local: ~30 LOC in `gemma4/_config.py` decide cache shapes;
~110 LOC of new code in `gemma4/_modules.py` implement the ring-buffer
write and the logical-mask gather; ~300 LOC in `_prefill.py` implement
the scratch-then-compact prefill flow that keeps prefill attention
numerics identical to the legacy path. The remaining files are small
adjustments: a uniform-layer-shape assumption in `_cache_helper.py`
becomes a `max()` across layers; the sampler loop's stop condition
switches from layer-0 physical shape to the logical `cache_length`.

Measured on TPU v5e at batch=1, single-turn decode:

- The local-window fork scales the largest reliable application context
  with the slice size: $8$k on v5e-4 (unchanged from stock), $32$k on
  v5e-8, and $64$k on v5e-16. Stock serving caps at $8$k on v5e-4 and
  $16$k on both v5e-8 and v5e-16 — adding chips buys parameter
  headroom but no cache headroom under stock.
- At the clean serving cell (v5e-4, $L{=}8192$, $B{=}1$, $N{=}21$
  cases), both modes serve every case. The fork pays $1.22$ GiB of
  per-chip HBM headroom for $275$ ms of additional prefill
  ($+19\%$ on `service_p50`). Per-token decode is $12\%$ faster under
  the fork ($0.100$ vs $0.114$ ms) because the logical-mask gather
  attends a shorter physical span.
- Across the v5e-4 long-context grid ($L \ge 16{,}384$), stock
  saturates the per-chip HBM cap ($\approx 15.73$ GiB) while the fork
  keeps $1.5$–$3.7$ GiB of headroom. The fork's wall-clock cost in
  this regime is $19$–$32\%$ on `service_p50`, essentially all in
  prefill ring-buffer compaction.

The trade is: pay a measurable prefill cost to make long context
serviceable. The net is positive past the point where stock stops
serving, and a small but real cost at shorter contexts. The mode is
opt-in for that reason.

---

## 2. Background

### 2.1 The over-allocation

Gemma 4 uses a hybrid attention pattern: every Nth layer is `GLOBAL`,
and the rest are `LOCAL_SLIDING` with a fixed `sliding_window_size`. For
the published Gemma 4 variants:

| Model     | Local | Global | Window |
|-----------|-------|--------|--------|
| E4B       | 35    | 7      | 512    |
| 26B (MoE) | 25    | 5      | 1024   |
| 31B dense | 50    | 10     | 1024   |

The sampler at
[`gemma/gm/text/_sampler.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampler.py)
and the model's `init_cache` at
[`gemma/gm/nn/gemma4/_config.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_config.py)
allocate a `[batch, cache_length, num_kv_heads, head_dim]` buffer per
attention layer regardless of attention type. The sliding-window mask
at
[`gemma4/_modules.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py)
already discards every slot outside the window, so on a local-sliding
layer at $L{=}16{,}384$, the persistent storage carries $512$ slots that
the attention mask can read and $15{,}872$ slots that it cannot. The
ratio of unread to read storage per local layer is $L/W - 1$; at
$L{=}16{,}384, W{=}512$ that is $31{\times}$.

![Gemma 4 E4B attention layer pattern. The 42-layer model decomposes
into 7 repeats of (5 local + 1 global). The local-window fork caps each
local layer's persistent decode cache at min(L, 512) slots in a ring
buffer; the 7 global layers retain full-length
storage.](docs/figures/gemma_attention_layers.png)

### 2.2 Slot math

In aggregate cache slots, with $L$ the cache length, $W$ the sliding
window, and Gemma 4 E4B's $35{+}7$ layer split:

$$S_{\text{stock}} = 42L \quad\quad S_{\text{lw}} = 7L + 35\min(L, W)$$

For E4B with $W{=}512$:

| L       | $S_{\text{stock}}$ | $S_{\text{lw}}$ | Reduction |
|---------|-------------------:|----------------:|----------:|
| 4 096   | 172 032            | 46 592          | 3.7×      |
| 8 192   | 344 064            | 75 264          | 4.6×      |
| 16 384  | 688 128            | 132 608         | 5.2×      |
| 32 768  | 1 376 256          | 247 296         | 5.6×      |
| 65 536  | 2 752 512          | 476 672         | 5.8×      |

The reduction approaches $42/7 = 6{\times}$ asymptotically as $L$ grows
past the window: in the limit, only the 7 global layers scale with
$L$.

This is narrower than a full serving-system memory manager such as
PagedAttention. It exploits the Gemma-specific local/global split. The
pattern generalizes to Longformer- and Mistral-style architectures, but
this change does not attempt that generalization.

---

## 3. Mechanism

### 3.1 The cache-mode flag

A new enum, in
[`gemma/gm/utils/_cache_helper.py`](./gemma/gm/utils/_cache_helper.py):

```python
class KVCacheMode(enum.Enum):
  LEGACY = 'legacy'         # full [B, L, H, D] per layer (default)
  LOCAL_WINDOW = 'local_window'  # local layers ring-buffered at W

class KVPrefillMode(enum.Enum):
  LEGACY_SCRATCH = 'legacy_scratch'  # only mode for now
```

`KVCacheMode.LEGACY` is the default at every call site
(`Sampler.__init__`, `Gemma4Sampler.__init__`, `ChatSampler.__init__`,
`Gemma4Transformer.init_cache`, `TransformerConfig.init_cache`). The
mode can also be set via the `GEMMA_KV_CACHE_MODE` and
`GEMMA_KV_PREFILL_MODE` environment variables for A/B testing without
editing call sites.

`KVPrefillMode` is single-valued today; it exists as a hook for a
future prefill optimization (direct prefill into the ring buffer)
without changing the public sampler API.

### 3.2 Per-layer cache shapes

In
[`gemma/gm/nn/gemma4/_config.py`](./gemma/gm/nn/gemma4/_config.py),
`TransformerConfig.init_cache` selects between layer factories per
attention type:

```python
local_window_size = (
    min(cache_length, self.sliding_window_size)
    if use_local_window and self.sliding_window_size is not None
    else None
)
for i, attn_type in enumerate(self.attention_types):
  if attn_type == AttentionType.GLOBAL and self.global_key_size is not None:
    cache[f'layer_{i}'] = Attention.init_cache(           # full L
        cache_length, ..., self.global_key_size)
  elif use_local_window:
    cache[f'layer_{i}'] = Attention.init_local_window_cache(  # W
        local_window_size, self.num_kv_heads, self.head_dim, ...)
  else:
    cache[f'layer_{i}'] = Attention.init_cache(           # legacy: full L
        cache_length, self.num_kv_heads, self.head_dim, ...)
```

`Attention.init_local_window_cache`
([`gemma4/_modules.py`](./gemma/gm/nn/gemma4/_modules.py)) allocates a
`[B, W, H, D]` k/v pair plus two metadata arrays that legacy caches do
not have:

- `logical_index: [B, W] int32` — the logical token position stored at
  each physical slot. Initialized to `-1`.
- `valid: [B, W] bool` — whether the slot currently holds a real K/V.
  Initialized to `False`.

A simple duck-typed predicate
(`is_local_window_layer(layer_data)` returns `'logical_index' in
layer_data`) lets the rest of the code distinguish the two layouts
without a separate enum threaded through the cache pytree.

### 3.3 Decode-step write

The decode loop's per-step write
([`Attention.__call__` in `gemma4/_modules.py`](./gemma/gm/nn/gemma4/_modules.py))
becomes one branch when the layer carries `logical_index`/`valid`:

- The physical slot is `segment_pos % W` (ring write).
- `k`, `v`, and `positions` are updated by
  `dynamic_update_slice`-style scatter at the physical slot.
- `logical_index` at the physical slot is set to the absolute logical
  token position; `valid` is set to `True`.
- The post-write cache leaves are re-constrained against the incoming
  cache leaf sharding so the functional update doesn't fall back to
  replicated.

The attention computation gathers the sampler's full logical mask
through `logical_index` before applying the sliding-window mask:

```python
# Cache stores W physical slots; mask is sized to the logical cache
# length. Map each physical slot to its logical position and gather:
safe_index = jnp.clip(cache_logical_index, 0, logical_len - 1)
gathered_mask = jnp.take_along_axis(full_mask, safe_index, axis=-1)
gathered_mask = gathered_mask & cache_valid[..., None, :]  # invalidate empties
```

This means the sliding mask's `cache_position` comparison continues to
use the stored absolute `positions` (not physical slot index), so
ring-buffer wraparound and per-row independence work without any
special casing in the math.

### 3.4 Prefill: scratch then compact

Prefill remains correct-by-construction by routing local layers through
a full-size scratch cache that has no `logical_index`/`valid` metadata,
running `model.apply` against the scratch, and compacting each row's
last $W$ valid logical slots into the persistent ring at end of
prefill.

In [`gemma/gm/text/_prefill.py`](./gemma/gm/text/_prefill.py):

- `_make_prefill_input_local_window` builds the scratch cache. Each
  local layer gets a `[B, prefill_cache_length, H, D]` k/v buffer
  matching the prompt bucket length. If the persistent ring already
  holds valid logical slots (multi-turn case), those slots are
  scattered into the scratch at their `logical_index` so prefill
  attention sees the same history as a legacy run.
- `_compact_local_window_layer` is the load-bearing piece. It takes a
  full-size scratch layer plus a `logical_valid_mask` over prefill
  positions and, per batch row independently:
  1. Rank the valid logical positions and select the **last $W$ valid
     positions**. (Not "the last $W$ physical positions" — that would
     be wrong for padded batches where one row finished much earlier
     than another.)
  2. Place each selected entry into physical slot
     `selected_pos % W`.
  3. Write `(k, v, positions, logical_index)`, set `valid=True` for
     occupied physical slots, and leave the rest invalid.

The implementation is in
[`_prefill.py:589-668`](./gemma/gm/text/_prefill.py#L589-L668). The
unit test that exercises the compaction across the three regimes
(`prompt_len < W`, `== W`, `> W`) plus a padded-batch case is at
[`examples/cache_local_window_test.py`](./examples/cache_local_window_test.py).

Global layers do not go through compaction — they use the existing
prefill path unchanged.

### 3.5 Per-layer shapes and `is_full`

With local layers physically sized to $W$ and global layers sized to
$L$, the assumption that every layer has the same cache length no
longer holds. Two places that read it are fixed:

- **`Cache.total_cache_length`** in
  [`_cache_helper.py`](./gemma/gm/utils/_cache_helper.py) now returns
  `max(d['k'].shape[1] for d in self.cache.values())`. For legacy
  caches this is unchanged; for local-window caches this returns the
  global (logical) cache length, which is what callers want for mask
  sizing and stop-condition checks.
- **The sampler-loop stop condition** in
  [`_sampler_loop.py`](./gemma/gm/text/_sampler_loop.py) compares
  `used_cache_length >= self.cache_length - 1` against the static
  logical `cache_length`, not any layer's physical shape. (Under the
  old code, with `LOCAL_WINDOW` enabled, the loop would have halted
  at step $W{-}1$ because layer 0 is local-sliding in every Gemma 4
  variant.)

There is also a small latent bug in the legacy path: `is_full` was
defined against layer 0's physical shape, which is always a
local-sliding layer in Gemma 4. With `cache_length \gg W`, layer 0 was
never sized below `cache_length` so the bug was not user-visible. The
fix to read the max keeps legacy semantics intact while fixing the
latent issue.

---

## 4. Measured behavior on TPU v5e

The numbers below come from runs against this branch using the
benchmark suite under
[`cloud-deploy-agent`](https://gitlab.cs.washington.edu/kburtram/cloud-deploy-agent),
on TPU v5e-4, v5e-8, and v5e-16. The application workload is a
deterministic 21-case agent benchmark; cases are short, single-turn
chat completions against a strict-validation final-report gate. Stock
versus fork are differentiated only by `GEMMA_KV_CACHE_MODE`.

### 4.1 Cross-slice scaling

![Cross-slice scaling for Gemma 4 E4B at B=1. Left: peak per-chip HBM
versus cache length. Stock sampler (dashed) hits the chip cap as L
grows on each slice; local-window (solid) stays below the cap because
only the 7 global layers retain full-history KV state. Right: largest
reliable application context per slice and cache mode, measured as the
largest cache_length with ok_rate >= 0.95.](docs/figures/cross_slice_scaling.png)

Largest reliable cache length, $\texttt{ok\_rate} \ge 0.95$ on the
21-case benchmark:

| Slice  | Stock | Local-window | Within-slice gap |
|--------|------:|-------------:|-----------------:|
| v5e-4  |   8k  |   8k         | parity           |
| v5e-8  |  16k  |  32k         | 2×               |
| v5e-16 |  16k  |  64k         | 4×               |

On v5e-4 the fork is at parity in this metric — it does not raise the
largest serviceable context on a 4-chip slice, because the workload's
prompt and chat lengths fit within 8k regardless. The slice-scaling
story starts at v5e-8 and is clearest on v5e-16, where stock fails to
use a 4× larger HBM pool for cache headroom.

Peak per-chip HBM at $L{=}65{,}536$ on v5e-16 under the fork remains
in the $10$–$11$ GiB range for E4B, with $\sim 5$ GiB of headroom below
the v5e chip cap ($\approx 15.75$ GiB). Stock fails to load this cell.

### 4.2 v5e-4 clean serving cell

The cleanest comparison is on v5e-4 at $L{=}8192,\, B{=}1$, where both
modes serve every case. This isolates the prefill cost from the
correctness behavior at the cache-cap regime.

| Metric                          | Stock  | Local-window | Δ                 |
|---------------------------------|-------:|-------------:|-------------------|
| peak per-chip HBM (GiB)         |  12.80 |        11.58 | −1.22 GiB (−9.5%) |
| prefill $p_{50}$ (ms)           |  1,410 |        1,686 | +275 ms (+19%)    |
| decode per-token $p_{50}$ (ms)  |  0.114 |        0.100 | −0.014 ms (−12%)  |
| service $p_{50}$ (ms)           |  1,452 |        1,732 | +280 ms (+19%)    |
| `ok_rate`                       |  1.000 |        1.000 | +0.000            |
| `expected_ok_rate`              |  0.714 |        0.762 | +0.048            |
| elapsed $p_{95}$ (s)            |   83.0 |         92.7 | +9.7 s (+12%)     |

This is a regime where stock has not run out of room — the cost
appears as a real `service_p50` increase, not as an availability gain.
The trade is visible: roughly $1.2$ GiB of per-chip headroom for
roughly $19\%$ added prefill latency. Per-token decode is faster
because the gather attends $W{=}512$ physical slots rather than $L$,
which dominates at the per-step level. The `expected_ok_rate` gap from
$1.0$ is agent-quality, not serving-quality: four mixed-remediation
cases leave SQL merge conflicts open. Both modes show the same gap.

### 4.3 v5e-4 long-context grid

At $L \ge 16{,}384$ on v5e-4, the picture flips: stock saturates the
$\approx 15.73$ GiB per-chip cap and either fails to serve or runs
with no margin for activations; the fork keeps measurable headroom.

![Peak per-chip HBM versus cache length on v5e-4 (B=1). The stock
sampler (dashed) saturates the ~15.73 GiB chip cap by L=16384 and
loses headroom; the local-window fork (solid) keeps 1.5–3.7 GiB
headroom across the same
grid.](docs/figures/v5e4_hbm_vs_cache.png)

![Service-latency p50 versus cache length on v5e-4 (B=1). The fork's
prefill cost shows up as 19–32% on service_p50 across the long-context
grid, with the gap closing slightly at the longest L because the
per-token decode advantage starts to compensate.](docs/figures/v5e4_service_latency.png)

Across the grid the fork's wall-clock cost is $19$–$32\%$ on
`service_p50`. This is essentially all in the prefill compaction; the
per-token decode component is $\le 12\%$ faster under the fork at
every cell tested.

### 4.4 v5e-16 frontier behavior

At long context on v5e-16, the local-window cache is necessary but not
sufficient for the larger Gemma 4 variants — 26B MoE and 31B dense
also need parameter sharding to fit, which is handled by mesh-level
configuration in the agent CLI, not by this Gemma branch. For E4B the
fork alone is sufficient: at $L{=}65{,}536, B{=}1$, the v5e-16 chips
peak at $\sim 4.9$ GiB load + cache for parameters and KV combined,
versus $\sim 15.73$ GiB cap, with the 21-case benchmark passing every
case.

---

## 5. Cost characterization

The local-window mode is a tradeoff, not a free win. The honest
breakdown:

- **Prefill compaction cost.** At the v5e-4 clean cell, $+275$ ms on
  `prefill_p50` ($+19\%$). Across the v5e-4 long-context grid this
  scales to $19$–$32\%$ on `service_p50`. The dominant work is the
  per-row rank-and-place compaction in `_compact_local_window_layer`,
  which is $O(B \cdot P \cdot W)$ where $P$ is the prompt-bucket
  length. This cost is paid once per prefill (not per turn in
  multi-turn mode after the first turn — subsequent turns add only
  one local-window write per generated token).

- **Decode per-token cost.** $-12\%$ at the v5e-4 clean cell
  ($0.100$ vs $0.114$ ms). The win is structural: the local-attention
  gather reads $W$ physical slots rather than $L$, and the logical-mask
  gather is also $W$-bounded.

- **HBM trade.** At the clean cell, the fork frees $1.22$ GiB per
  chip. At the long-context grid the freed headroom is what makes the
  fork serviceable where stock is not.

- **Net.** At $L \le 8$k the fork costs `service_p50` without
  buying availability. Past $8$k the fork is what makes the request
  serve at all. The opt-in default reflects this: users who do not
  need long context should keep the default; users who do need long
  context can opt in.

- **What the fork does not change.** Numerics on the global layers
  are bit-identical. Numerics on the local layers are within bf16
  tolerance of legacy on the test set; the gather-then-mask order is
  algebraically equivalent to mask-then-implicit-truncate but
  reduction order can change. Greedy sampling tokens match legacy
  greedy for the test prompts in
  [`examples/cache_local_window_test.py`](./examples/cache_local_window_test.py).

---

## 6. API surface

The cache mode is plumbed through the samplers as a constructor kwarg.
The default is `KVCacheMode.LEGACY` everywhere, so an existing call
site that does not pass `kv_cache_mode=` sees stock behavior.

```python
from gemma import gm

# Default: legacy behavior, unchanged.
sampler = gm.text.ChatSampler(model=model, params=params, cache_length=8192)

# Opt-in to local-window for long context.
sampler = gm.text.ChatSampler(
    model=model, params=params, cache_length=65536,
    kv_cache_mode=gm.utils.KVCacheMode.LOCAL_WINDOW,
)
```

The mode also reads from `GEMMA_KV_CACHE_MODE` when no kwarg is
passed, which is the path the benchmark suite uses to A/B test without
editing call sites.

Three considerations a maintainer might prefer to handle differently,
all of which I would be happy to iterate on:

1. **Naming.** `LOCAL_WINDOW` is descriptive but not necessarily the
   right slot in the public taxonomy. `RingBufferLocal` or
   `WindowSizedLocalKV` are alternatives. The current naming was
   chosen to leave room for additional modes (e.g., a future
   direct-prefill variant) without renaming the enum.

2. **Placement of the enum.** `KVCacheMode` currently lives in
   `gemma/gm/utils/_cache_helper.py`. If the convention is for public
   enums to live in a different module (e.g., `gemma/gm/text/`), I can
   move it.

3. **Default.** The current default is `LEGACY`. If a maintainer
   prefers `LOCAL_WINDOW` to become the default for Gemma 4 — the data
   suggests it should be at $L \ge 16$k — that is a one-line change
   per call site, but it is a behavior change for existing users and
   I left it out of this PR to keep the diff strictly additive.

4. **`KVPrefillMode` exposure.** `LEGACY_SCRATCH` is the only mode,
   and the enum exists as a hook for a future direct-prefill
   optimization. If a maintainer would prefer not to expose a
   single-valued enum, the prefill branch can be selected internally
   from `KVCacheMode` and `KVPrefillMode` can be deleted.

---

## 7. Scope and limitations

What this branch has been tested against:

- **Models:** Gemma 4 E4B (primary), 26B MoE, 31B dense.
- **Hardware:** TPU v5e-4, v5e-8, v5e-16.
- **Decode:** batch=1 single-turn and multi-turn.
- **Prefill bucket lengths:** $\le W$, $= W$, $> W$, and padded
  batches where rows differ in prompt length, all in
  [`examples/cache_local_window_test.py`](./examples/cache_local_window_test.py).

What this branch has not been tested against:

- **GPU backends.** The implementation is JAX-pure (no Pallas
  kernels), so it should run on GPU, but it has not been measured
  there.
- **Other Gemma sizes.** Gemma 3, Gemma 3n, and other Gemma 4
  variants beyond the three above. The `gemma3n` path uses a similar
  hybrid attention pattern in
  [`gemma3n/_config.py`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma3n/_config.py),
  so a mirror change would be straightforward, but this branch leaves
  that for a follow-up rather than scope-creep this PR.
- **Other sliding-window architectures.** Longformer- and
  Mistral-style models have the same structural opportunity, but the
  code in this PR is specifically wired through Gemma 4's
  `Attention.__call__` and is not framed as a general mechanism.
- **Quantized cache.** Independent axis; can compose, has not been
  tested.
- **Cache sharding policies.** This branch's
  [`_transformer.py`](./gemma/gm/nn/gemma4/_transformer.py) accepts a
  mesh and forwards it to a per-layer partition spec for the
  local-window cache leaves, so the metadata fields are replicated
  while K/V can be sharded on the head axis when divisible. The
  active mesh choice (e.g., $f16t2$ for 31B on v5e-16) is set at the
  agent CLI layer, not inside this branch.

---

## 8. Adjacent change in this branch: bf16 checkpoint restore

A small adjacent change in
[`gemma/gm/ckpts/_checkpoint.py`](./gemma/gm/ckpts/_checkpoint.py)
(+31 LOC) adds an optional `dtype=` argument to `load_params` and
`LoadCheckpoint` that retargets only floating-point `ShapeDtypeStruct`
leaves and leaves integer and bool metadata leaves at their checkpoint
dtype. This is intended for inference-only runs where the on-disk
checkpoint stores fp32 weights and the doubling of loaded parameter
HBM is the binding factor (specifically, 31B on v5e-16 at long
context).

This change is not load-bearing for the local-window cache claim. It
is in the branch because the two changes are used together at the
serving level: bf16 restore makes 31B parameter footprint fit, and
the local-window cache makes the per-layer KV footprint fit at long
context. The bf16 restore could be split into a separate PR if
preferred; it is included here so the branch reflects the configuration
that produced the measured 31B/v5e-16 results in §4.4.

Test coverage:
[`gemma/gm/ckpts/_checkpoint_test.py`](./gemma/gm/ckpts/_checkpoint_test.py).

---

## 9. Code map

Files in this branch, by role:

**Core cache layout and mode flag**
- [`gemma/gm/utils/_cache_helper.py`](./gemma/gm/utils/_cache_helper.py)
  — `KVCacheMode`, `KVPrefillMode`, `is_local_window_layer`,
  `mesh_from_params`, `Cache.total_cache_length` fix.
- [`gemma/gm/nn/gemma4/_config.py`](./gemma/gm/nn/gemma4/_config.py)
  — per-layer cache size selection at `init_cache`.

**Per-layer mechanism**
- [`gemma/gm/nn/gemma4/_modules.py`](./gemma/gm/nn/gemma4/_modules.py)
  — `Attention.init_local_window_cache`, per-step ring-buffer write,
  logical-mask gather.
- [`gemma/gm/nn/gemma4/_transformer.py`](./gemma/gm/nn/gemma4/_transformer.py)
  — mode propagation and mesh inference for cache leaves.

**Prefill orchestration**
- [`gemma/gm/text/_prefill.py`](./gemma/gm/text/_prefill.py)
  — `_make_prefill_input_local_window`,
  `_make_local_window_prefill_scratch_layer`,
  `_merge_cache_local_window`, `_compact_local_window_layer`.

**Sampler-loop adjustments**
- [`gemma/gm/text/_sampler_loop.py`](./gemma/gm/text/_sampler_loop.py)
  — stop condition reads logical `cache_length`, not layer-0
  physical shape.
- [`gemma/gm/text/_chat_sampler.py`](./gemma/gm/text/_chat_sampler.py)
  and
  [`gemma/gm/text/_gemma4_sampler.py`](./gemma/gm/text/_gemma4_sampler.py)
  — `kv_cache_mode` constructor kwarg, defaulting to `LEGACY`.

**Tests**
- [`gemma/gm/nn/gemma4/_transformer_test.py`](./gemma/gm/nn/gemma4/_transformer_test.py)
  — per-layer cache shape invariants under both modes.
- [`gemma/gm/text/_prefill_test.py`](./gemma/gm/text/_prefill_test.py)
  — prefill compaction across `prompt_len < W`, `== W`, `> W`, and
  padded batches.
- [`examples/cache_local_window_test.py`](./examples/cache_local_window_test.py)
  — end-to-end local-window correctness against legacy on short
  prompts.

**Diagnostic tooling (read-only)**
- [`examples/cache_audit.py`](./examples/cache_audit.py) — inspects
  cache pytree shape, dtype, and sharding per leaf without loading
  params. Useful for verifying that opt-in produces the expected
  shapes.
- [`examples/cache_probe.py`](./examples/cache_probe.py) — runs a
  real `ChatSampler.chat()` with FSDP-sharded params and reports
  per-chip HBM deltas, separated by allocation phase. Used to derive
  the v5e-4 numbers in §4.

**Adjacent**
- [`gemma/gm/ckpts/_checkpoint.py`](./gemma/gm/ckpts/_checkpoint.py)
  — optional `dtype=` for floating-leaf restore (see §8).

---

## 10. Reproduction

The benchmark suite used for §4 lives at
[`gitlab.cs.washington.edu/kburtram/cloud-deploy-agent`](https://gitlab.cs.washington.edu/kburtram/cloud-deploy-agent).
The relevant entry points are:

```bash
# v5e-4 clean cell (§4.2):
python gemma4_jax.py serve --model gemma4-e4b-it --cache-length 8192
GEMMA_KV_CACHE_MODE=legacy        python run_benchmark.py --N 21
GEMMA_KV_CACHE_MODE=local_window  python run_benchmark.py --N 21

# v5e-4 long-context grid (§4.3):
for L in 16384 32768; do
  GEMMA_KV_CACHE_MODE=local_window \
    python gemma4_jax.py serve --model gemma4-e4b-it --cache-length $L
  python run_benchmark.py --N 21
done

# v5e-16 long-context E4B (§4.4):
GEMMA_KV_CACHE_MODE=local_window \
  python gemma4_jax.py serve --model gemma4-e4b-it --cache-length 65536 \
    --mesh-2d data=8,tensor=2 --mesh-2d-param-sharding full \
    --param-dtype bfloat16
python run_benchmark.py --N 21
```

The Gemma-side diagnostic that doesn't require the agent harness:

```bash
GEMMA_KV_CACHE_MODE=local_window python examples/cache_probe.py \
  --model gemma4_e4b --cache-length 16384
```

Trace bundles for the v5e-4, v5e-8, and v5e-16 runs (per-case
service-latency JSONs, per-chip HBM samples, sharding reports) are
preserved out-of-tree.

---

## 11. What this is not

To save reviewer time, this change is not:

- **PagedAttention.** No block tables, no Pallas gather kernels, no
  ragged attention. The eager-buffer approach with a per-layer
  ring-buffer is sufficient for batch=1 research-and-evaluation
  inference and is what fits cleanly into the existing sampler.
- **A cache-sharding refactor.** The default Gemma 4 cache placement
  is unchanged. The branch does expose a partition spec when a mesh
  is available, which lets a caller shard cache K/V on the head axis
  when divisible, but cache sharding policy is the caller's choice and
  is not load-bearing for the local-window claim.
- **A general sliding-window mechanism.** The plumbing is specific to
  `gemma4`. The pattern would port to `gemma3n` with a small change,
  and conceptually to Longformer/Mistral-style architectures, but
  this branch does not attempt that.
- **A quantization change.** The K/V dtype is unchanged.
- **A multi-tenant serving optimization.** The cost characterization
  applies to single-request decode. Heterogeneous-length batch
  serving has different tradeoffs that this branch does not address.
