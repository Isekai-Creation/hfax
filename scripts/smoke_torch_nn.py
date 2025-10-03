#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import torch


def main():
    # Prepend the torch port path so `gm` resolves to hfax-torch/gm
    repo_root = os.path.dirname(os.path.dirname(__file__))
    torch_pkg = os.path.join(repo_root, 'hfax-torch')
    sys.path.insert(0, torch_pkg)

    from gm.nn._modules import Block, AttentionType  # type: ignore
    from gm.math import _pos_utils  # type: ignore

    B, T = 2, 4
    embed_dim = 16
    num_heads = 4
    head_dim = 4
    hidden_dim = 32

    x = torch.randn(B, T, embed_dim)
    mask = torch.ones(B, T, dtype=torch.bool)
    positions = _pos_utils.build_positions_from_mask(mask)
    # causal mask: [B, T, T]
    causal = torch.ones(T, T, dtype=torch.bool).tril().unsqueeze(0).expand(B, -1, -1)

    block = Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attn_type=AttentionType.GLOBAL,
        query_pre_attn_scalar=1.0 / (head_dim ** 0.5),
        transpose_gating_einsum=False,
    )

    cache, y = block(x, positions, None, causal)
    assert y.shape == (B, T, embed_dim)
    print('OK: forward output shape', y.shape)


if __name__ == '__main__':
    main()

