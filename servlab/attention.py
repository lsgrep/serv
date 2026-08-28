"""Attention, built from the arithmetic up — because serving is downstream of it.

Every other lab in this repo consumes facts about attention: the KV cache
formula, why GQA shrinks it, why prefill is compute bound and decode is
bandwidth bound, why long context is expensive. Those facts are usually
memorised. Here they are derived, in numpy, on tensors small enough to print.

The five things worth taking away, in the order they build:

1. **Attention is a weighted lookup.** Scores, softmax, weighted sum of values.
   Three lines.
2. **Causal masking is what makes a cache possible.** Because token i can only
   attend to tokens <= i, the K and V computed for token i are *never revised*.
   That immutability — not an optimisation, a property of the mask — is the
   entire justification for the KV cache.
3. **Prefill and decode are the same function at different shapes.** Q is
   `[n, d]` for prefill and `[1, d]` for decode. Same code, opposite bottleneck.
4. **The score matrix is the quadratic term**, and it is *memory* before it is
   compute. Flash attention never materialises it; that is what it buys, and it
   buys no FLOPs at all.
5. **Positions are baked into K.** Cached keys are only reusable at the position
   they were computed for — which is exactly why prefix caching works on a
   shared prefix and cannot work on a shared suffix.

Pure numpy, tiny tensors, no GPU. If a claim here surprises you, change a
number and rerun it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Softmax, including the version that breaks
# --------------------------------------------------------------------------


def naive_softmax(x, axis=-1):
    """The textbook formula. Overflows on real logits — try it with x=1000."""
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax(x, axis=-1):
    """Subtract the max first. Mathematically identical, numerically survivable.

    This is not a detail: the max-subtraction is precisely the thing that makes
    the *streaming* version below possible, because a running max can be
    corrected after the fact.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def online_softmax(scores, block_size=4):
    """Softmax over blocks, holding only a running max and sum.

    This is the trick behind flash attention, and it is worth doing by hand
    once. Process the scores in tiles; keep `m` (running max) and `l` (running
    sum of exponentials). When a new tile has a larger max, rescale what you
    already accumulated by `exp(old_m - new_m)` and carry on.

    The consequence: you never need the whole row of scores in memory at the
    same time. Softmax stops being a barrier that forces materialisation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    *lead, n = scores.shape
    flat = scores.reshape(-1, n)
    out = np.empty_like(flat)
    for r, row in enumerate(flat):
        m, denom = -np.inf, 0.0
        acc = np.zeros(n)
        for start in range(0, n, block_size):
            tile = row[start:start + block_size]
            tile_max = tile.max()
            new_m = max(m, tile_max)
            # Rescale the accumulated sum to the new max, then add this tile.
            scale = np.exp(m - new_m) if m != -np.inf else 0.0
            denom = denom * scale + np.exp(tile - new_m).sum()
            acc[:start] *= scale
            acc[start:start + len(tile)] = np.exp(tile - new_m)
            m = new_m
        out[r] = acc / denom
    return out.reshape(*lead, n)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def causal_mask(q_len, k_len=None, offset=None):
    """True where attention is allowed.

    `offset` is where this query block starts in the sequence — it is what makes
    decode work: a single query at position 47 may attend to keys 0..47, so the
    mask is a row, not a triangle.
    """
    k_len = k_len if k_len is not None else q_len
    offset = (k_len - q_len) if offset is None else offset
    q_pos = np.arange(q_len)[:, None] + offset
    k_pos = np.arange(k_len)[None, :]
    return k_pos <= q_pos


def apply_mask(scores, mask, value=-1e9):
    return np.where(mask, scores, value)


# --------------------------------------------------------------------------
# Attention itself
# --------------------------------------------------------------------------


def attention(Q, K, V, mask=None, return_weights=False, scale=None):
    """Scaled dot-product attention. `Q [.., q, d]`, `K/V [.., k, d]`.

    The 1/sqrt(d) exists so the dot products do not grow with dimension and
    push softmax into a one-hot corner. Drop it and watch the weights collapse.
    """
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / (scale or np.sqrt(d))
    if mask is not None:
        scores = apply_mask(scores, mask)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return (out, weights) if return_weights else out


def flash_attention(Q, K, V, mask=None, block_size=4, scale=None):
    """Tiled attention that never materialises the full score matrix.

    Identical output to `attention()` — verify it, do not take it on faith. The
    saving is memory traffic, not arithmetic: the same multiplications happen,
    but the `q x k` intermediate never leaves fast on-chip memory.

    That distinction is the one people get wrong. Flash attention does not make
    attention cheaper in FLOPs. It makes it stop being memory bound.
    """
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    q_len, d = Q.shape[-2], Q.shape[-1]
    k_len = K.shape[-2]
    s = scale or np.sqrt(d)

    out = np.zeros_like(Q)
    for i in range(0, q_len, block_size):
        q_tile = Q[..., i:i + block_size, :]
        rows = q_tile.shape[-2]
        m = np.full((*q_tile.shape[:-1], 1), -np.inf)
        denom = np.zeros((*q_tile.shape[:-1], 1))
        acc = np.zeros_like(q_tile)
        for j in range(0, k_len, block_size):
            k_tile = K[..., j:j + block_size, :]
            v_tile = V[..., j:j + block_size, :]
            s_tile = q_tile @ np.swapaxes(k_tile, -1, -2) / s
            if mask is not None:
                s_tile = apply_mask(s_tile, mask[i:i + rows, j:j + k_tile.shape[-2]])
            tile_max = s_tile.max(axis=-1, keepdims=True)
            new_m = np.maximum(m, tile_max)
            correction = np.exp(m - new_m)
            correction = np.where(np.isfinite(correction), correction, 0.0)
            p = np.exp(s_tile - new_m)
            denom = denom * correction + p.sum(axis=-1, keepdims=True)
            acc = acc * correction + p @ v_tile
            m = new_m
        out[..., i:i + block_size, :] = acc / denom
    return out


def score_matrix_bytes(seq_len, n_heads, dtype_bytes=2, batch=1) -> float:
    """Bytes the naive `q x k` score matrix would occupy. Quadratic in context —
    this is the term flash attention removes."""
    return batch * n_heads * seq_len * seq_len * dtype_bytes


def flash_workspace_bytes(block_size, n_heads, dtype_bytes=2, batch=1) -> float:
    """What the tiled version holds instead: one block, not the matrix."""
    return batch * n_heads * block_size * block_size * dtype_bytes


# --------------------------------------------------------------------------
# Heads: MHA, MQA, GQA
# --------------------------------------------------------------------------


def split_heads(x, n_heads):
    """`[n, d_model] -> [n_heads, n, d_head]`."""
    n, d_model = x.shape
    return x.reshape(n, n_heads, d_model // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    """`[n_heads, n, d_head] -> [n, d_model]`."""
    n_heads, n, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n, n_heads * d_head)


def repeat_kv(kv, n_heads):
    """Broadcast `n_kv_heads` key/value heads up to `n_heads` query heads.

    This one function is all that grouped-query attention *is* at inference
    time. The KV cache stores `n_kv_heads` of them; the kernel repeats each one
    across its group of query heads. Nothing is recomputed and nothing is
    approximated — the cache is simply smaller, by exactly `n_heads/n_kv_heads`.
    """
    n_kv = kv.shape[0]
    if n_kv == n_heads:
        return kv
    if n_heads % n_kv:
        raise ValueError(f"{n_heads} query heads is not divisible by {n_kv} kv heads")
    return np.repeat(kv, n_heads // n_kv, axis=0)


@dataclass
class AttentionWeights:
    """One layer's projections, sized for MHA, MQA or GQA by `n_kv_heads`."""

    d_model: int
    n_heads: int
    n_kv_heads: int = 0
    seed: int = 0
    Wq: np.ndarray = field(default=None, repr=False)
    Wk: np.ndarray = field(default=None, repr=False)
    Wv: np.ndarray = field(default=None, repr=False)
    Wo: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.n_kv_heads = self.n_kv_heads or self.n_heads
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        rng = np.random.default_rng(self.seed)
        d, h, kv = self.d_model, self.n_heads, self.n_kv_heads
        d_head = d // h
        scale = 1 / np.sqrt(d)
        if self.Wq is None:
            self.Wq = rng.normal(scale=scale, size=(d, h * d_head))
            # The KV projections are narrower under GQA — this is where the
            # memory saving is created, not in the attention maths.
            self.Wk = rng.normal(scale=scale, size=(d, kv * d_head))
            self.Wv = rng.normal(scale=scale, size=(d, kv * d_head))
            self.Wo = rng.normal(scale=scale, size=(h * d_head, d))

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / self.n_kv_heads

    def project(self, x):
        """`x [n, d_model]` -> per-head Q, K, V."""
        q = split_heads(x @ self.Wq, self.n_heads)
        k = split_heads(x @ self.Wk, self.n_kv_heads)
        v = split_heads(x @ self.Wv, self.n_kv_heads)
        return q, k, v


def mha_forward(x, w: AttentionWeights, causal=True, return_weights=False):
    """Full-sequence attention — the prefill path."""
    q, k, v = w.project(x)
    mask = causal_mask(x.shape[0]) if causal else None
    out, weights = attention(q, repeat_kv(k, w.n_heads), repeat_kv(v, w.n_heads),
                             mask=mask, return_weights=True)
    y = merge_heads(out) @ w.Wo
    return (y, weights) if return_weights else y


# --------------------------------------------------------------------------
# The KV cache, built rather than assumed
# --------------------------------------------------------------------------


@dataclass
class KVCache:
    """Append-only per-layer store of keys and values.

    Append-only is the whole point, and it is a consequence of causal masking:
    once token i's key and value exist, no later token changes them. If
    attention were bidirectional this class could not exist.
    """

    n_kv_heads: int
    d_head: int
    dtype_bytes: int = 2
    keys: np.ndarray = None
    values: np.ndarray = None

    def append(self, k, v):
        """`k, v` are `[n_kv_heads, t, d_head]` for t new tokens."""
        self.keys = k if self.keys is None else np.concatenate([self.keys, k], axis=1)
        self.values = v if self.values is None else np.concatenate([self.values, v], axis=1)
        return self

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[1]

    @property
    def nbytes(self) -> float:
        """What the formula in every other lab is counting, measured here."""
        return 2 * self.n_kv_heads * self.d_head * self.length * self.dtype_bytes

    def bytes_per_token(self) -> float:
        return 2 * self.n_kv_heads * self.d_head * self.dtype_bytes


def prefill(x, w: AttentionWeights, cache: KVCache = None):
    """Process the whole prompt at once, filling the cache."""
    q, k, v = w.project(x)
    cache = cache or KVCache(w.n_kv_heads, w.d_head)
    cache.append(k, v)
    mask = causal_mask(x.shape[0])
    out = attention(q, repeat_kv(cache.keys, w.n_heads), repeat_kv(cache.values, w.n_heads),
                    mask=mask)
    return merge_heads(out) @ w.Wo, cache


def decode_step(x_t, w: AttentionWeights, cache: KVCache):
    """One new token against the cache.

    Note the shapes: Q is `[n_heads, 1, d_head]` — a single row — while K and V
    are the entire history. That asymmetry is the whole of the prefill/decode
    story. The matmul is a matrix-vector product, so it moves far more bytes
    than it does arithmetic, and the GPU sits waiting on memory.
    """
    q, k, v = w.project(np.atleast_2d(x_t))
    cache.append(k, v)
    # A single query at the end of the sequence attends to everything before it,
    # so no mask is needed — the cache contains only the past by construction.
    out = attention(q, repeat_kv(cache.keys, w.n_heads), repeat_kv(cache.values, w.n_heads))
    return merge_heads(out) @ w.Wo, cache


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def attention_flops(seq_len, d_model, n_heads=1, decode=False, ctx_len=None) -> float:
    """FLOPs for the attention *scores and weighted sum only* — projections excluded.

    Prefill: 2 * 2 * n^2 * d (scores, then the weighted sum).
    Decode:  2 * 2 * ctx * d for one query row.
    """
    if decode:
        ctx = ctx_len if ctx_len is not None else seq_len
        return 4 * ctx * d_model
    return 4 * seq_len * seq_len * d_model


def projection_flops(seq_len, d_model, n_heads, n_kv_heads=None) -> float:
    """Q, K, V and the output projection. Linear in sequence length."""
    n_kv = n_kv_heads or n_heads
    d_head = d_model // n_heads
    widths = d_model * (n_heads * d_head) * 2 + 2 * d_model * (n_kv * d_head)
    return 2 * seq_len * widths


def attention_share(seq_len, d_model, n_heads, n_kv_heads=None, ffn_mult=4) -> float:
    """Fraction of a layer's FLOPs spent inside attention scores.

    At short context this is a rounding error and the model is "just matmuls".
    It grows linearly with sequence length, and the point where it stops being
    ignorable is the answer to *why long context is expensive*.
    """
    attn = attention_flops(seq_len, d_model)
    proj = projection_flops(seq_len, d_model, n_heads, n_kv_heads)
    ffn = 2 * seq_len * (2 * ffn_mult * d_model * d_model)
    return attn / (attn + proj + ffn)


def quadratic_crossover(d_model, n_heads, n_kv_heads=None, ffn_mult=4, target=0.5) -> int:
    """Context length at which attention scores reach `target` of layer FLOPs.

    A number worth having for any model you discuss: below it, context is nearly
    free; above it, doubling context more than doubles the cost.
    """
    lo, hi = 8, 1 << 22
    while lo < hi:
        mid = (lo + hi) // 2
        if attention_share(mid, d_model, n_heads, n_kv_heads, ffn_mult) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Positions — why cached keys are not portable
# --------------------------------------------------------------------------


def rope(x, positions=None, base=10000.0):
    """Rotary position embedding, applied to `[.., n, d_head]`.

    Positions are rotated *into* the key and query vectors. The consequence for
    serving is the one that matters: a cached key carries the position it was
    computed at. Reuse it at a different offset and the geometry is wrong.

    That is why prefix caching is a *prefix* cache. A shared system prompt sits
    at positions 0..n every time, so its keys are reusable verbatim. A shared
    paragraph appearing midway through two different documents sits at different
    offsets, and its cached keys are worthless to the second one.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape[-2], x.shape[-1]
    if d % 2:
        raise ValueError("head dim must be even for RoPE")
    pos = np.arange(n) if positions is None else np.asarray(positions)
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
    theta = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(theta), np.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def relative_score_shift(d_head=8, gap=3, base=10000.0) -> dict:
    """Show that RoPE scores depend on the *distance* between positions.

    Two consequences, both practical: a cached key is valid only at its own
    offset, and a query/key pair separated by the same gap scores identically
    wherever it sits — which is what lets a model generalise over position at
    all.
    """
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, d_head))
    k = rng.normal(size=(1, d_head))
    same_gap = []
    for start in (0, 5, 50):
        qr = rope(q, positions=[start + gap], base=base)
        kr = rope(k, positions=[start], base=base)
        same_gap.append((qr @ kr.T).item())
    wrong_offset = (rope(q, positions=[gap]) @ rope(k, positions=[7]).T).item()
    return {"same_gap_scores": same_gap,
            "spread": max(same_gap) - min(same_gap),
            "wrong_offset_score": wrong_offset}


# --------------------------------------------------------------------------
# Looking at the weights
# --------------------------------------------------------------------------


def attention_entropy(weights) -> np.ndarray:
    """Entropy per query row, in nats. Low means the head has picked one token;
    high means it is averaging — often over the first token, which is the
    'attention sink' behaviour you see in trained models."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
    return -(w * np.log(w)).sum(axis=-1)


def sink_share(weights) -> float:
    """Average attention mass landing on position 0."""
    w = np.asarray(weights, dtype=np.float64)
    return float(w[..., 0].mean())
