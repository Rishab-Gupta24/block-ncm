#!/usr/bin/env python3
"""Reproduces every number quoted in paper/sections/complexity_and_bounds.tex.

Analytical FLOP / KV-cache accounting for Monolithic NCM vs. Hierarchical
Block-NCM. Pure Python, no dependencies.

    py scripts/complexity_constants.py
    py scripts/complexity_constants.py --N 2048 --M 8 --C 8 --K 6 --d 1024 --Lx 0
"""
import argparse
import math


def scales(K, kind="geometric", base_scale=0.5):
    """Level scale factors s_1..s_K with s_K = 1 (s_k = base_scale^(K-k) if geometric)."""
    if kind == "geometric":
        return [base_scale ** (K - k) for k in range(1, K + 1)]
    if kind == "uniform":
        return [1.0] * K
    raise ValueError(kind)


def sigmas(s, pattern="block_causal"):
    """sigma_1 = sum_k s_k.

    sigma_2 counts query-key pairs (in units of N^2):
      block_causal   : level k attends to levels j<=k -> sum_k s_k * sum_{j<=k} s_j
      block_diagonal : level k attends to itself only -> sum_k s_k^2   (released NCM configs)
    """
    sigma1 = sum(s)
    if pattern == "block_causal":
        sigma2 = sum(sk * sum(s[: k + 1]) for k, sk in enumerate(s))
        # identity check: sigma2 = (sigma1^2 + sum s^2) / 2
        assert math.isclose(sigma2, 0.5 * (sigma1 ** 2 + sum(x * x for x in s)))
    elif pattern == "block_diagonal":
        sigma2 = sum(x * x for x in s)
    else:
        raise ValueError(pattern)
    return sigma1, sigma2


def attn_flops(N, M, C, Lx, d, s1, s2):
    """Per-layer attention FLOPs (QK^T + AV, 2 FLOPs per MAC). M=1, C=0 -> monolithic."""
    return 4 * d * (N * N * s2 / M + N * (Lx + M * C) * s1)


def linear_flops(N, M, C, Lx, d, s1, c_lin=24, share_prefix=True):
    """Per-layer projection + MLP FLOPs (c_lin * d^2 per token processed)."""
    tokens = N * s1 + M * C + (Lx if share_prefix else M * Lx)
    return c_lin * d * d * tokens


def kv_tokens(N, M, C, Lx, s1, share_prefix=True, share_blueprint=True, pattern="block_causal"):
    """Tokens resident in the KV cache at the last level (all blocks resident).

    block_causal caches every level (N*sigma_1 output tokens); block_diagonal
    only ever holds the current level (N tokens at level K).
    """
    bp = M * C if share_blueprint else M * M * C
    px = Lx if share_prefix else M * Lx
    out = N * s1 if pattern == "block_causal" else N
    return out + bp + px


def main():
    ap = argparse.ArgumentParser(description="Compute Block-NCM complexity constants.")
    ap.add_argument("--N", type=int, default=2048, help="Target sequence length N")
    ap.add_argument("--M", type=int, default=8, help="Number of macro-blocks M")
    ap.add_argument("--C", type=int, default=8, help="Blueprint codes per block C")
    ap.add_argument("--K", type=int, default=6, help="Number of hierarchy levels K")
    ap.add_argument("--d", type=int, default=1024, help="Hidden dimension d")
    ap.add_argument("--Lx", type=int, default=0, help="Prompt length L_x")
    ap.add_argument("--c-lin", type=float, default=24, help="Projection+MLP FLOPs per token, in units of d^2")
    ap.add_argument("--scales", default="geometric", choices=["geometric", "uniform"])
    ap.add_argument("--base-scale", type=float, default=0.5, help="Ratio between consecutive geometric levels")
    ap.add_argument("--pattern", default="block_causal", choices=["block_causal", "block_diagonal"],
                    help="Cross-level attention pattern (released NCM configs use block_diagonal)")
    ap.add_argument("--levels", default=None,
                    help="Comma-separated level lengths, e.g. the generated levels of a released config. "
                         "Overrides --K/--scales/--N (N = last level length).")
    a = ap.parse_args()

    if a.levels:
        lv = [int(x) for x in a.levels.split(",")]
        a.N, a.K = lv[-1], len(lv)
        s = [x / lv[-1] for x in lv]
        a.scales = "explicit"
    else:
        s = scales(a.K, a.scales, a.base_scale)
    s1, s2 = sigmas(s, a.pattern)
    print(f"--- N={a.N} d={a.d} M={a.M} C={a.C} K={a.K} Lx={a.Lx} scales={a.scales} pattern={a.pattern} ---")
    print(f"sigma1={s1:.4f} sigma2={s2:.4f} sigma1/sigma2={s1/s2:.4f}")

    A_mono = attn_flops(a.N, 1, 0, a.Lx, a.d, s1, s2)
    A_blk = attn_flops(a.N, a.M, a.C, a.Lx, a.d, s1, s2)
    r_attn = A_blk / A_mono
    print(f"attention ratio (exact)        = {r_attn:.4f}  -> {1/r_attn:.2f}x reduction")
    print(f"  naive 1/M                    = {1/a.M:.4f}")

    Mstar = math.sqrt(a.N * s2 / (a.C * s1)) if a.C else float("inf")
    print(f"attention-optimal M* = sqrt(N s2 / (C s1)) = {Mstar:.1f}")

    L_mono = linear_flops(a.N, 1, 0, a.Lx, a.d, s1, a.c_lin)
    L_blk = linear_flops(a.N, a.M, a.C, a.Lx, a.d, s1, a.c_lin)
    phi = A_mono / (A_mono + L_mono)
    r_tot = (A_blk + L_blk) / (A_mono + L_mono)
    print(f"attention share of mono FLOPs  phi = {phi:.3f}")
    print(f"end-to-end FLOP ratio          = {r_tot:.4f}  -> {1/r_tot:.2f}x reduction")

    for sp, sb, tag in [(True, True, "prefix+blueprint shared"),
                        (False, False, "naive batch fold")]:
        kb = kv_tokens(a.N, a.M, a.C, a.Lx, s1, sp, sb, a.pattern)
        km = kv_tokens(a.N, 1, 0, a.Lx, s1, pattern=a.pattern)
        print(f"KV-cache ratio ({tag:24s}) = {kb/km:.4f}")

    # materialised score matrix at level K, per head (only relevant without fused attention)
    keys = a.N * s1 if a.pattern == "block_causal" else a.N
    S_mono = a.N * (a.Lx + keys)
    S_blk = a.N * (a.Lx + a.M * a.C) + a.N * keys / a.M
    print(f"score-matrix ratio (level K)   = {S_blk/S_mono:.4f}")


if __name__ == "__main__":
    main()
