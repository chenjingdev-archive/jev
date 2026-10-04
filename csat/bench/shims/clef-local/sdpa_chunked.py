"""Bound SDPA's temporary attention matrix by chunking query rows only.

The original PyTorch SDPA still computes every attention value. Each query row
sees the complete original K/V, scale, GQA grouping and allowed attention mask.
No context is removed, no windows or KV approximations are used, and no tensor
dtype changes. Softmax is over the same full key axis as in dense SDPA.

Explicit installation is process-local. By default it affects only inference
on CUDA/HIP with more than 4096 query rows; the chunk size is 512. Smaller calls,
autograd-enabled calls, and nonzero-dropout calls retain the original kernel.
Supplying both is_causal and an explicit mask also delegates: PyTorch's math
backend rejects that combination whereas some fused backends accept it.
"""
from functools import wraps
from typing import Callable

import torch
import torch.nn.functional as functional


def make_chunked_sdpa(
    original: Callable,
    *,
    query_chunk_size: int = 512,
    min_query_tokens: int = 4096,
    device_types: tuple[str, ...] | None = ('cuda',),
):
    if query_chunk_size < 1 or min_query_tokens < 0:
        raise ValueError('query_chunk_size must be positive; min_query_tokens cannot be negative')
    config = {'query_chunk_size': query_chunk_size, 'min_query_tokens': min_query_tokens,
              'device_types': list(device_types) if device_types is not None else None,
              'full_key_value_context': True, 'changes_dtype': False,
              'inference_only': True, 'dropout_p': 0.0}
    counters = {'chunked_calls': 0, 'delegated_calls': 0, 'largest_query_tokens': 0}

    @wraps(original)
    def chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, *, scale=None, enable_gqa=False):
        length, source_length = query.size(-2), key.size(-2)
        if (length <= min_query_tokens or length <= query_chunk_size
                or torch.is_grad_enabled() or dropout_p != 0.0
                or (is_causal and attn_mask is not None)
                or (device_types is not None and query.device.type not in device_types)):
            counters['delegated_calls'] += 1
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        # An LxS mask (optionally prefixed with batch/head axes) must be sliced
        # along L. A 1xS or S key mask is broadcast unchanged for every chunk.
        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor):
                raise TypeError('chunked SDPA requires a Tensor attention mask')
            if attn_mask.ndim >= 2 and attn_mask.size(-2) not in (1, length):
                raise ValueError('attention mask query dimension must be 1 or the full query length')
        counters['chunked_calls'] += 1
        counters['largest_query_tokens'] = max(counters['largest_query_tokens'], length)
        key_positions = torch.arange(source_length, device=query.device) if is_causal else None
        outputs = []
        for start in range(0, length, query_chunk_size):
            stop = min(start + query_chunk_size, length)
            mask = attn_mask
            if mask is not None and mask.ndim >= 2 and mask.size(-2) == length:
                mask = mask[..., start:stop, :]
            if is_causal:
                # PyTorch is_causal=True is the upper-left-aligned tril(L,S).
                # Chunk-local tril would reset row indices and discard valid
                # earlier context. Keep absolute original query row positions.
                query_positions = torch.arange(start, stop, device=query.device)
                causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                mask = causal
            outputs.append(original(
                query[..., start:stop, :], key, value,
                attn_mask=mask, dropout_p=0.0, is_causal=False,
                scale=scale, enable_gqa=enable_gqa,
            ))
        return torch.cat(outputs, dim=-2)

    chunked_sdpa.clef_sdpa_chunked_config = config
    chunked_sdpa.clef_sdpa_chunked_counters = counters
    chunked_sdpa.clef_sdpa_original = original
    return chunked_sdpa


def install(*, query_chunk_size=512, min_query_tokens=4096, device_types=('cuda',)):
    """Install once in the calling server process and return its wrapper.

    Importing this module alone changes nothing. The caller should disclose
    this allocation strategy and rerun model parity/long-input validation.
    """
    current = functional.scaled_dot_product_attention
    if hasattr(current, 'clef_sdpa_chunked_config'):
        expected = {'query_chunk_size': query_chunk_size, 'min_query_tokens': min_query_tokens,
                    'device_types': list(device_types) if device_types is not None else None,
                    'full_key_value_context': True, 'changes_dtype': False,
                    'inference_only': True, 'dropout_p': 0.0}
        if current.clef_sdpa_chunked_config != expected:
            raise RuntimeError('a differently configured Clef SDPA wrapper is already installed')
        return current
    functional.scaled_dot_product_attention = make_chunked_sdpa(
        current, query_chunk_size=query_chunk_size, min_query_tokens=min_query_tokens,
        device_types=device_types,
    )
    return functional.scaled_dot_product_attention
