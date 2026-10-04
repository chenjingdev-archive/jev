"""Numerical and dispatch checks; no model weights or server are loaded."""
import argparse
import json

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from sdpa_chunked import make_chunked_sdpa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--head-dim', type=int, default=32)
    args = parser.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.set_num_threads(4)
    torch.manual_seed(7231)
    dense = functional.scaled_dot_product_attention
    observed = []

    def spy(q, k, v, **kw):
        observed.append((q.shape[-2], k.shape[-2], v.shape[-2]))
        return dense(q, k, v, **kw)

    chunked = make_chunked_sdpa(spy, query_chunk_size=256, min_query_tokens=256, device_types=None)
    results = []

    def compare(name, batch, heads, kv_heads, length, source, kind, causal=False, scale=None):
        q = torch.randn(batch, heads, length, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, kv_heads, source, args.head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, kv_heads, source, args.head_dim, device=device, dtype=dtype)
        mask = None
        if kind == 'bool-full':
            mask = torch.rand(batch, 1, length, source, device=device) > 0.2
            mask[..., 13, :] = False  # fully masked row must keep the same semantics
        elif kind == 'float-full':
            mask = torch.randn(1, heads, length, source, device=device, dtype=torch.float32) * 0.2
            mask[..., 21, :] = float('-inf')
            mask[..., :, ::11] = float('-inf')
        elif kind == 'bool-key-broadcast':
            mask = torch.ones(batch, 1, 1, source, device=device, dtype=torch.bool)
            mask[..., ::7] = False
        elif kind == 'float-key-broadcast':
            mask = torch.randn(1, source, device=device, dtype=dtype) * 0.1
            mask[..., ::9] = float('-inf')
        kwargs = dict(attn_mask=mask, dropout_p=0.0, is_causal=causal,
                      scale=scale, enable_gqa=heads != kv_heads)
        observed.clear()
        # Force math to compare the very backend that would materialize the
        # large quadratic score matrix, instead of comparing unrelated kernels.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            expected = dense(q, k, v, **kwargs)
            actual = chunked(q, k, v, **kwargs)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert all(qn <= 256 and kn == source and vn == source for qn, kn, vn in observed)
        assert sum(qn for qn, _, _ in observed) == length
        # Same matrix operations can have BF16 kernel-rounding differences.
        tolerance = 0.008 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        delta = (actual.float() - expected.float()).abs()
        results.append({'case': name, 'length': length, 'source': source,
                        'max_abs_delta': delta.max().item(), 'mean_abs_delta': delta.mean().item(),
                        'chunks': len(observed), 'passed': True})

    compare('causal-offset-tail', 1, 4, 4, 1025, 1025, None, True)
    compare('gqa-causal-nonsquare-scale', 1, 4, 2, 513, 769, None, True, 0.23)
    compare('boolean-batched-full-mask', 2, 2, 2, 769, 513, 'bool-full')
    compare('float-head-specific-mask', 1, 2, 2, 1024, 769, 'float-full')
    compare('gqa-broadcast-key-mask', 2, 4, 2, 2048, 513, 'bool-key-broadcast')
    compare('float-broadcast-nonsquare', 1, 2, 2, 513, 1024, 'float-key-broadcast')

    # The production threshold must bypass all current game/CSAT lengths and
    # dropout/autograd without altering the arguments. This uses only a spy.
    calls = []
    def sentinel(*pos, **kw):
        calls.append((pos, kw))
        return pos[0]
    production = make_chunked_sdpa(sentinel, device_types=None)
    q = torch.zeros(1, 1, 4096, 1)
    with torch.inference_mode():
        assert production(q, q, q, is_causal=True) is q
        assert len(calls) == 1 and calls[0][1]['is_causal'] is True
        q = torch.zeros(1, 1, 4097, 1)
        assert production(q, q, q, dropout_p=0.1) is q
    assert production(q, q, q) is q  # autograd-enabled invocation delegates
    assert production.clef_sdpa_chunked_counters['chunked_calls'] == 0
    print(json.dumps({'device': str(device), 'dtype': str(dtype), 'torch': torch.__version__,
                      'head_dim': args.head_dim,
                      'numeric_cases': results, 'production_dispatch_checks': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
