"""Roofline serving-cost model: autoregressive (AR) vs. NCM variants (B1, B2, Block-NCM).

Every number this script prints follows from the equations below and the command-line inputs. Hardware
specs are public; utilisation, price and model size are ASSUMPTIONS and should be reported as such.

Time for one forward pass is set by whichever resource binds (roofline):

    t_pass = max( bytes_moved / BW_eff ,  flops / F_eff )        BW_eff = BW * mbu,  F_eff = peak * mfu

  flops per position     = 2P (dense matmuls) + 4 * L * d * S   (QK^T and AV over S attended positions)
  bytes per pass         = 2P (bf16 weights)  [+ KV-cache reads for AR]

AR (batch B, N output tokens): N passes; pass n processes B positions with context n and reads the KV cache
    of B sequences: B * n * 2 * L * d_kv * 2 bytes.
NCM (K target levels of lengths l_1..l_K per C-token chunk, M chunks, prefix p):
    pass t processes B*M*l_t positions (B2 / Block-NCM: all chunks in one pass)
    B1  : chunks one after another -> M*K passes of B*l_t positions
    B2  : every level global, S_t = p + M*l_t
    Ours: levels t < G global (S_t = p + M*l_t); levels t >= G local with a KV-cached plan,
          S_t = p + M*l_{G-1} + l_t (plan attended, not recomputed)
    rho = sum(l_t) / C  is the number of positions NCM runs per output token (5.0 for TinyStories levels).

Cost per output token = price_per_second * wall_time / tokens.   Latency = wall time of one batch.
NOT modelled: tokenizer decode, kernel launch overhead, utilisation falling on tiny passes, quality.

    python paper/scripts/serving_cost.py                      # 7B on H100, M=16 (2,048 tokens)
    python paper/scripts/serving_cost.py --M 64 --price 4.0 --mfu 0.5
"""
import argparse

LEVELS = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]  # released TinyStories target levels (C = 128)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--P", type=float, default=7e9, help="parameters (same for AR and NCM: an assumption)")
ap.add_argument("--L", type=int, default=32, help="layers")
ap.add_argument("--d", type=int, default=4096, help="model width")
ap.add_argument("--kv-ratio", type=float, default=1.0, help="KV heads / query heads (1 = MHA, 0.125 = GQA-8)")
ap.add_argument("--peak", type=float, default=989e12, help="peak bf16 FLOP/s (H100 SXM dense: 989e12)")
ap.add_argument("--bw", type=float, default=3.35e12, help="memory bandwidth B/s (H100 SXM: 3.35e12)")
ap.add_argument("--mfu", type=float, default=0.40, help="achieved fraction of peak FLOP/s (assumption)")
ap.add_argument("--mbu", type=float, default=0.80, help="achieved fraction of bandwidth (assumption)")
ap.add_argument("--hbm", type=float, default=80e9, help="GPU memory bytes (H100: 80e9); AR rows whose weights + "
                                                        "full KV cache exceed it are marked infeasible (*)")
ap.add_argument("--price", type=float, default=2.50, help="$ per GPU-hour (assumption)")
ap.add_argument("--M", type=int, default=16, help="chunks per output (output tokens = M * C)")
ap.add_argument("--C", type=int, default=128, help="tokens per chunk")
ap.add_argument("--G", type=int, default=3, help="global levels in Block-NCM")
ap.add_argument("--prefix", type=int, default=1)
ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
ap.add_argument("--Ms", default="1,2,4,8,16,32,64", help="M values for the latency sweep at batch 1")
a = ap.parse_args()

F = a.peak * a.mfu
BW = a.bw * a.mbu
W = 2 * a.P                        # bf16 weight bytes
lt = [l * a.C // 128 for l in LEVELS]
K = len(lt)
rho = sum(lt) / a.C
usd_per_s = a.price / 3600


def attn_flops(S):
    return 4 * a.L * a.d * S


def t_pass(positions, S, extra_bytes=0.0):
    flops = positions * (2 * a.P + attn_flops(S))
    return max((W + extra_bytes) / BW, flops / F)


def ar(B, N):
    kv_per_tok = 2 * a.L * a.d * a.kv_ratio * 2  # K and V, bf16
    t = sum(t_pass(B, n, B * n * kv_per_tok) for n in range(1, N + 1))
    return t


def ncm(B, M, variant):
    p, t = a.prefix, 0.0
    plan = M * lt[a.G - 1] if 0 < a.G < K else 0
    for i, l in enumerate(lt):
        if variant == "B1":
            t += M * t_pass(B * l, p + l)               # one chunk at a time
        elif variant == "B2" or i < a.G:
            t += t_pass(B * M * l, p + M * l)           # global level, all chunks
        else:
            t += t_pass(B * M * l, p + plan + l)        # local level, cached plan
    return t


def row(name, t, tokens):
    return name, t, usd_per_s * t / tokens * 1e6


N = a.M * a.C
B_star = (W / BW) * F / (2 * a.P)
print(f"# Serving-cost roofline: P={a.P:.2g}, L={a.L}, d={a.d}, kv_ratio={a.kv_ratio}")
print(f"# hardware: peak {a.peak:.3g} FLOP/s x mfu {a.mfu} = {F:.3g};  BW {a.bw:.3g} B/s x mbu {a.mbu} = {BW:.3g}")
print(f"# price ${a.price}/GPU-h (assumption); output {N} tokens = M {a.M} x C {a.C}; Block-NCM G={a.G}")
print(f"# weight-load time W/BW = {W / BW * 1e3:.2f} ms; dense compute 2P/F = {2 * a.P / F * 1e3:.4f} ms/position")
print(f"# AR memory->compute crossover B* = {B_star:.0f};  NCM positions per token rho = {rho:.2f};  "
      f"predicted AR-cheaper-than-NCM batch ~ B*/rho = {B_star / rho:.0f}\n")

print("## Latency (s per batch) and cost ($ per 1M output tokens) vs batch size\n")
print("| batch | AR latency | AR $/1M | B1 latency | B1 $/1M | B2 latency | B2 $/1M | Block-NCM latency | Block-NCM $/1M |")
print("|---|---|---|---|---|---|---|---|---|")
cross = None
kv_full = N * 2 * a.L * a.d * a.kv_ratio * 2  # bytes of KV cache for one finished sequence
for B in [int(x) for x in a.batches.split(",")]:
    r = [row("AR", ar(B, N), B * N), row("B1", ncm(B, a.M, "B1"), B * N),
         row("B2", ncm(B, a.M, "B2"), B * N), row("Ours", ncm(B, a.M, "Ours"), B * N)]
    fits = W + B * kv_full <= a.hbm
    cells = [f"{t:.3f} | {c:.3f}" for _, t, c in r]
    if not fits:
        cells[0] = cells[0].replace(" | ", "* | ") + "*"
    print(f"| {B} | " + " | ".join(cells) + " |")
    if cross is None and fits and r[0][2] < r[3][2]:
        cross = B
print(f"\n(*) AR weights + KV cache ({kv_full / 1e9:.2f} GB per sequence) exceed {a.hbm / 1e9:.0f} GB: infeasible batch.")
print(f"AR becomes cheaper per token than Block-NCM from batch {cross} (feasible batches swept; None = never).\n")

print("## Batch-1 latency vs output length (s)\n")
print("| M | tokens | AR | B1 | B2 | Block-NCM | AR / Block-NCM | B1 / Block-NCM | B2 / Block-NCM |")
print("|---|---|---|---|---|---|---|---|---|")
for M in [int(x) for x in a.Ms.split(",")]:
    t_ar, t_b1, t_b2, t_o = ar(1, M * a.C), ncm(1, M, "B1"), ncm(1, M, "B2"), ncm(1, M, "Ours")
    print(f"| {M} | {M * a.C} | {t_ar:.3f} | {t_b1:.3f} | {t_b2:.3f} | {t_o:.3f} | {t_ar / t_o:.1f}x | "
          f"{t_b1 / t_o:.2f}x | {t_b2 / t_o:.2f}x |")
