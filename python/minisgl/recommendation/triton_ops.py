"""Optional CUDA/ROCm kernels. Import only when a GPU path is explicitly selected."""

import torch
import triton as tr
import triton.language as tl

from .precision import QuantizedRows, fp8_dtype


@tr.jit
def _quant(
    X,
    R,
    W,
    Y,
    S,
    RES,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    LIMIT: tl.constexpr,
    MODE: tl.constexpr,
    HAS_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * STRIDE + col, col < N, 0).to(tl.float32)
    if MODE == 1:
        if HAS_RES:
            value += tl.load(R + row * N + col, col < N, 0).to(tl.float32)
        value = value.to(X.dtype.element_ty).to(tl.float32)
        tl.store(RES + row * N + col, value, col < N)
        square = tl.where(col < N, value * value, 0.0)
        value = value * tl.rsqrt(tl.sum(square, 0) / N + EPS)
        value *= tl.load(W + col, col < N, 0).to(tl.float32)
    elif MODE == 2:
        up = tl.load(X + row * STRIDE + N + col, col < N, 0).to(tl.float32)
        value = value * tl.sigmoid(value) * up
    value = value.to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.max(tl.where(col < N, tl.abs(value), 0.0), 0), 1e-12) / LIMIT
    tl.store(S + row, scale)
    tl.store(Y + row * N + col, tl.minimum(tl.maximum(value / scale, -LIMIT), LIMIT), col < N)


def _quantize(x, mode=0, residual=None, weight=None, eps=0.0):
    if x.ndim != 2:
        raise ValueError("FP8 kernels require a matrix")
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.shape[-1] // (2 if mode == 2 else 1)
    values = torch.empty((x.shape[0], n), dtype=fp8_dtype(), device=x.device)
    scales = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    res = torch.empty_like(x) if mode == 1 else x
    _quant[(x.shape[0],)](
        x,
        residual if residual is not None else x,
        weight if weight is not None else x,
        values,
        scales,
        res,
        n,
        x.stride(0),
        eps,
        torch.finfo(fp8_dtype()).max,
        mode,
        residual is not None,
        tr.next_power_of_2(n),
    )
    return QuantizedRows(values, scales, x.dtype), res


def quantize(x):
    return _quantize(x)[0]


def rms_quantize(x, residual, weight, eps):
    return _quantize(x, 1, residual, weight, eps)


def silu_quantize(x):
    return _quantize(x, 2)[0]


@tr.jit
def _matmul(
    A,
    B,
    AS,
    BS,
    BIAS,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        a = tl.load(
            A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0.0
        )
        b = tl.load(
            B + cols[None, :] * K + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), 0.0
        )
        accum = tl.dot(a, b, accum)
    sa = tl.load(AS + rows, rows < M, 1.0)
    sb = tl.load(BS + cols, cols < N, 1.0)
    value = accum * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        value += tl.load(BIAS + cols, cols < N, 0.0)[None, :]
    tl.store(
        C + rows[:, None] * N + cols[None, :], value, (rows[:, None] < M) & (cols[None, :] < N)
    )


def matmul(x, weight, bias):
    m, k = x.values.shape
    n = weight.values.shape[0]
    output = torch.empty((m, n), dtype=x.output_dtype, device=x.values.device)
    _matmul[(tr.cdiv(m, 16), tr.cdiv(n, 64))](
        x.values,
        weight.values,
        x.scales,
        weight.scales,
        bias if bias is not None else output,
        output,
        m,
        n,
        k,
        bias is not None,
        16,
        64,
        64,
        num_warps=4,
    )
    return output


@tr.jit
def _legal_scores(
    X,
    SCORES,
    NODES,
    OFFSETS,
    COLS,
    OUT,
    V: tl.constexpr,
    DEGREE: tl.constexpr,
    BV: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    vocab = tl.arange(0, BV)
    logits = tl.load(X + row * V + vocab, vocab < V, -float("inf")).to(tl.float32)
    maximum = tl.max(logits, 0)
    logsum = tl.log(tl.sum(tl.exp(logits - maximum), 0))
    node = tl.load(NODES + row)
    start, end = tl.load(OFFSETS + node), tl.load(OFFSETS + node + 1)
    edge = tl.arange(0, BD)
    valid = (edge < DEGREE) & (start + edge < end)
    column = tl.load(COLS + start + edge, valid, 0)
    value = tl.load(X + row * V + column).to(tl.float32)
    score = value - maximum - logsum + tl.load(SCORES + row)
    tl.store(OUT + row * DEGREE + edge, tl.where(valid, score, -float("inf")), edge < DEGREE)


def legal_scores(logits, state, trie):
    """Fuse log-softmax reduction, CSR gather, mask and cumulative score addition."""
    degree = trie.edge_range.numel()
    result = torch.empty((*logits.shape[:2], degree), device=logits.device, dtype=torch.float32)
    _legal_scores[(logits.shape[0] * logits.shape[1],)](
        logits.contiguous(),
        state.scores.contiguous(),
        state.nodes.contiguous(),
        trie.offsets,
        trie.columns,
        result,
        logits.shape[-1],
        degree,
        tr.next_power_of_2(logits.shape[-1]),
        tr.next_power_of_2(degree),
    )
    return result


@tr.jit
def _qk_rope_store(
    X,
    QW,
    KW,
    COS,
    POS,
    LOC,
    Q,
    KC,
    VC,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    NORM_Q: tl.constexpr,
    NORM_K: tl.constexpr,
    QEPS: tl.constexpr,
    KEPS: tl.constexpr,
    LIMIT: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, D)
    partner = (col + D // 2) % D
    sign = tl.where(col < D // 2, -1.0, 1.0)
    pos = tl.load(POS + token)
    cos = tl.load(COS + pos * D + col % (D // 2)).to(tl.float32)
    sin = tl.load(COS + pos * D + D // 2 + col % (D // 2)).to(tl.float32)
    q = tl.load(X + token * STRIDE + head * D + col).to(tl.float32)
    qp = tl.load(X + token * STRIDE + head * D + partner).to(tl.float32)
    if NORM_Q:
        scale = tl.rsqrt(tl.sum(q * q, 0) / D + QEPS)
        q = (q * scale * tl.load(QW + col)).to(X.dtype.element_ty).to(tl.float32)
        qp = (qp * scale * tl.load(QW + partner)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Q + (token * NQ + head) * D + col, q * cos + sign * qp * sin)
    if head < NK:
        k = tl.load(X + token * STRIDE + (NQ + head) * D + col).to(tl.float32)
        kp = tl.load(X + token * STRIDE + (NQ + head) * D + partner).to(tl.float32)
        if NORM_K:
            scale = tl.rsqrt(tl.sum(k * k, 0) / D + KEPS)
            k = (k * scale * tl.load(KW + col)).to(X.dtype.element_ty).to(tl.float32)
            kp = (kp * scale * tl.load(KW + partner)).to(X.dtype.element_ty).to(tl.float32)
        k = (k * cos + sign * kp * sin).to(X.dtype.element_ty).to(tl.float32)
        v = tl.load(X + token * STRIDE + (NQ + NK + head) * D + col).to(tl.float32)
        slot = tl.load(LOC + token)
        target = (slot * NK + head) * D + col
        tl.store(KC + target, tl.minimum(tl.maximum(k, -LIMIT), LIMIT))
        tl.store(VC + target, tl.minimum(tl.maximum(v, -LIMIT), LIMIT))


def qk_rope_store(qkv, layer, ctx):
    q = torch.empty(
        (qkv.shape[0], layer.num_qo_heads, layer.head_dim), dtype=qkv.dtype, device=qkv.device
    )
    qweight = layer.q_norm.weight if layer.q_norm is not None else qkv
    kweight = layer.k_norm.weight if layer.k_norm is not None else qkv
    _qk_rope_store[(qkv.shape[0], layer.num_qo_heads)](
        qkv,
        qweight,
        kweight,
        layer.rotary._cos_sin_cache,
        ctx.batch.positions,
        ctx.batch.out_loc,
        q,
        ctx.kv_cache.k_cache(layer.layer_id),
        ctx.kv_cache.v_cache(layer.layer_id),
        layer.num_qo_heads,
        layer.num_kv_heads,
        layer.head_dim,
        qkv.stride(0),
        layer.q_norm is not None,
        layer.k_norm is not None,
        layer.q_norm.eps if layer.q_norm else 0.0,
        layer.k_norm.eps if layer.k_norm else 0.0,
        torch.finfo(ctx.kv_cache.dtype).max,
        num_warps=4,
    )
    return q
