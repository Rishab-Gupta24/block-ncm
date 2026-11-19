"""Simulate serving latency/cost of Block-NCM method variants with the roofline model of serving_cost.py.

Same equations and defaults as serving_cost.py (7B, H100, 40% MFU, 80% MBU, $2.50/GPU-h, GQA-8 AR), plus knobs:
  levels        target level lengths per C-token chunk
  depth[t]      fraction of the network run for level t (level-adaptive depth / early exit); scales both
                FLOPs and weight bytes of that pass
  halo h        local levels also process h positions of each neighbouring chunk (extra queries and keys)
These are projections: they say what each method would cost IF it kept quality, not whether it does.

    python paper/scripts/method_scenarios.py
"""
import argparse

TS = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]

ap = argparse.ArgumentParser()
ap.add_argument("--P", type=float, default=7e9)
ap.add_argument("--L", type=int, default=32)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--kv-ratio", type=float, default=0.125)
ap.add_argument("--peak", type=float, default=989e12)
ap.add_argument("--bw", type=float, default=3.35e12)
ap.add_argument("--mfu", type=float, default=0.40)
ap.add_argument("--mbu", type=float, default=0.80)
ap.add_argument("--price", type=float, default=2.50)
ap.add_argument("--C", type=int, default=128)
a = ap.parse_args()
F, BW, W, usd = a.peak * a.mfu, a.bw * a.mbu, 2 * a.P, a.price / 3600


def t_pass(pos, S, frac=1.0, extra=0.0):
    flops = pos * frac * (2 * a.P + 4 * a.L * a.d * S)
    return max((W * frac + extra) / BW, flops / F)


def ar(B, N):
    kv = 2 * a.L * a.d * a.kv_ratio * 2
    return sum(t_pass(B, n, 1.0, B * n * kv) for n in range(1, N + 1))


def ncm(B, M, levels, G, depth=None, halo=0, variant="ours"):
    depth = depth or [1.0] * len(levels)
    K, t = len(levels), 0.0
    plan = M * levels[G - 1] if 0 < G < K else 0
    for i, l in enumerate(levels):
        f = depth[i]
        if variant == "B1":
            t += M * t_pass(B * l, 1 + l, f)
        elif variant == "B2" or i < G:
            t += t_pass(B * M * l, 1 + M * l, f)
        else:
            h = min(halo, l // 4)
            t += t_pass(B * M * (l + 2 * h), 1 + plan + l + 2 * h, f)
    return t


def rho_eff(levels, depth=None, halo=0, G=3):
    depth = depth or [1.0] * len(levels)
    pos = sum((l + (2 * min(halo, l // 4) if i >= G else 0)) * f for i, (l, f) in enumerate(zip(levels, depth)))
    return pos / a.C


half_fine = [1.0] * 6 + [0.5] * 5                    # levels 7-11 at half depth
few = [8, 16, 32, 64, 96, 128]                       # 6 levels (hypothetical tokenizer re-design)
scen = [
    ("Block-NCM G=3 (current)", dict(levels=TS, G=3)),
    ("Block-NCM G=6", dict(levels=TS, G=6)),
    ("G=3 + halo h=8", dict(levels=TS, G=3, halo=8)),
    ("G=3 + half-depth levels 7-11", dict(levels=TS, G=3, depth=half_fine)),
    ("G=2 + 6 levels [8..128]", dict(levels=few, G=2)),
    ("G=2 + 6 levels + half-depth last 3", dict(levels=few, G=2, depth=[1, 1, 1, .5, .5, .5])),
    ("G=3 + 0.25x refiner on local levels", dict(levels=TS, G=3, depth=[1.0] * 3 + [0.25] * 8)),
    ("G=3 + 0.15x refiner on local levels", dict(levels=TS, G=3, depth=[1.0] * 3 + [0.15] * 8)),
    ("B2 fully global", dict(levels=TS, G=3, variant="B2")),
    ("B1 chunks in sequence", dict(levels=TS, G=3, variant="B1")),
]
M = 16
N = M * a.C
ar1, ar256 = ar(1, N), ar(256, N)
print(f"# 7B on H100 (assumptions as serving_cost.py), output {N} tokens (M={M}); AR = GQA-8")
print(f"# AR: batch-1 latency {ar1:.2f} s, ${usd * ar1 / N * 1e6:.3f}/1M at batch 1, "
      f"${usd * ar256 / (256 * N) * 1e6:.3f}/1M at batch 256\n")
print("| method | positions per token | batch-1 latency, 2k tok | vs AR | batch-1 latency, 8k tok | $/1M batch 1 | "
      "$/1M batch 256 | AR cheaper from batch |")
print("|---|---|---|---|---|---|---|---|")
for name, kw in scen:
    t1 = ncm(1, M, **kw)
    t8k = ncm(1, 64, **kw)
    c1 = usd * t1 / N * 1e6
    c256 = usd * ncm(256, M, **kw) / (256 * N) * 1e6
    cross = next((B for B in range(1, 513) if usd * ar(B, N) / (B * N) < usd * ncm(B, M, **kw) / (B * N)), None) \
        if kw.get("variant") != "B1" or True else None
    rp = rho_eff(kw["levels"], kw.get("depth"), kw.get("halo", 0), kw["G"])
    print(f"| {name} | {rp:.2f} | {t1:.3f} s | {ar1 / t1:.0f}x | {t8k:.2f} s | {c1:.3f} | {c256:.3f} | "
          f"{cross if cross else '>512'} |")


# ---------------------------------------------------------------- streaming and speculative decoding (batch 1)
def ar_spec(N, P_draft=0.5e9, gamma=4, alpha=0.8):
    """AR + speculative decoding at batch 1: gamma draft passes (memory-bound draft model) + one verification
    pass of gamma+1 positions per round; expected accepted tokens per round (1 - alpha^(gamma+1)) / (1 - alpha)."""
    acc = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    t_draft = gamma * (2 * P_draft / BW)
    rounds, t, n = 0, 0.0, 0
    while n < N:
        t += t_draft + t_pass(gamma + 1, n + gamma + 1, 1.0, n * 2 * a.L * a.d * a.kv_ratio * 2)
        n += acc
        rounds += 1
    return t


print("\n## Batch-1 streaming: time to first token (TTFT) and to last token (TTLT), 2,048 tokens\n")
print("| method | TTFT | TTLT | streams? |\n|---|---|---|---|")
t_ar_tok = t_pass(1, 1, 1.0)
rows = [
    ("AR (GQA)", t_ar_tok, ar(1, N), "per token"),
    ("AR + speculative decoding (0.5B draft, gamma 4, acceptance 0.8)", t_ar_tok, ar_spec(N), "per round"),
    ("B1 chunks in sequence", ncm(1, 1, TS, 3), ncm(1, M, TS, 3, variant="B1"), "per chunk (128 tok)"),
    ("Block-NCM G=3", ncm(1, M, TS, 3), ncm(1, M, TS, 3), "no: all at once"),
    ("Super-block streaming: Block-NCM on 4-chunk blocks, in sequence", ncm(1, 4, TS, 3), 4 * ncm(1, 4, TS, 3),
     "per super-block (512 tok)"),
]
for name, ttft, ttlt, s in rows:
    print(f"| {name} | {ttft * 1e3:.0f} ms | {ttlt:.2f} s | {s} |")
print("\n(prefill and the super-block's conditioning prefix are not modelled)")
