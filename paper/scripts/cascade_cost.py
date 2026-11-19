"""Roofline cost of Paper 2 (Cascade-NCM) variants vs AR and Block-NCM: same model as serving_cost.py, extended
with per-level depth (fraction of the 12/12 layers a level runs) and skipped levels (passes never run).

A level running depth fraction f loads f of the weights and does f of the per-position FLOPs:
    t_pass = max( f*W / BW_eff ,  positions * f*(2P + 4 L d S) / F_eff )
Skipped levels cost nothing. Halo adds 2h computed positions per chunk on the levels it is applied to.
Quality is NOT modelled: what each variant costs in quality is the open question (see RESULTS.md, Milestone 4 and
the level-skip probe).

    python paper/scripts/cascade_cost.py
"""
import argparse

LEVELS = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]  # target levels 1..11 (level 0, l=1, is the prefix)

ap = argparse.ArgumentParser()
ap.add_argument("--P", type=float, default=7e9)
ap.add_argument("--L", type=int, default=32)
ap.add_argument("--d", type=int, default=4096)
ap.add_argument("--kv-ratio", type=float, default=0.125, help="AR baseline uses GQA-8")
ap.add_argument("--peak", type=float, default=989e12)
ap.add_argument("--bw", type=float, default=3.35e12)
ap.add_argument("--mfu", type=float, default=0.40)
ap.add_argument("--mbu", type=float, default=0.80)
ap.add_argument("--price", type=float, default=2.50)
ap.add_argument("--power", type=float, default=700.0)
ap.add_argument("--M", type=int, default=16)
ap.add_argument("--C", type=int, default=128)
ap.add_argument("--G", type=int, default=3)
ap.add_argument("--batches", default="1,8,16,32,64,128,256")
a = ap.parse_args()

F, BW, W = a.peak * a.mfu, a.bw * a.mbu, 2 * a.P
N = a.M * a.C


def t_pass(pos, S, f=1.0, extra=0.0):
    return max((f * W + extra) / BW, pos * f * (2 * a.P + 4 * a.L * a.d * S) / F)


def ar(B):
    kv = 2 * a.L * a.d * a.kv_ratio * 2
    return sum(t_pass(B, n, 1.0, B * n * kv) for n in range(1, N + 1))


def ncm(B, depth=None, skip=(), halo=None):
    """depth: {level: fraction}; skip: levels never run; halo: {level: h}. Levels 1..11; levels <= G are global."""
    depth, halo = depth or {}, halo or {}
    plan = a.M * LEVELS[a.G - 1]
    t = 0.0
    for k, l in enumerate(LEVELS, start=1):
        if k in skip:
            continue
        f = depth.get(k, 1.0)
        if k <= a.G:
            t += t_pass(B * a.M * l, 1 + a.M * l, f)
        else:
            lh = l + 2 * halo.get(k, 0)
            t += t_pass(B * a.M * lh, 1 + plan + lh, f)
    return t


LOCAL = range(a.G + 1, 12)
V = {
    "Block-NCM + halo 4 (Paper 1)": dict(halo={k: 4 for k in LOCAL}),
    "Cascade as proposed: 4-6 at 6/12 + halo, 9 & 11 at 3/12, skip 7,8,10":
        dict(depth={4: .5, 5: .5, 6: .5, 9: .25, 11: .25}, skip=(7, 8, 10), halo={4: 4, 5: 4, 6: 4}),
    "Evidence-adjusted: full depth, skip 7,8,10, halo on 4-6":
        dict(skip=(7, 8, 10), halo={4: 4, 5: 4, 6: 4}),
    "Conservative: full depth, skip 7,9, halo on 4-6":
        dict(skip=(7, 9), halo={4: 4, 5: 4, 6: 4}),
    "Aggressive skip: full depth, skip 7-10, halo on 4-6":
        dict(skip=(7, 8, 9, 10), halo={4: 4, 5: 4, 6: 4}),
}

usd = a.price / 3600
print(f"# 7B-class roofline (P={a.P:.2g}), H100, {N} output tokens (M={a.M}), price ${a.price}/GPU-h, {a.power:.0f} W")
print(f"# AR baseline: GQA kv_ratio {a.kv_ratio}. Quality differences between variants are NOT modelled.\n")
bs = [int(x) for x in a.batches.split(",")]
t_ar = {B: ar(B) for B in bs}
print("| variant | passes | positions/chunk | batch-1 latency (s) | x faster than AR | $/1M at batch 1 | "
      "kWh/1M at batch 1 | AR cheaper from batch |")
print("|---|---|---|---|---|---|---|---|")
print(f"| AR (GQA) | {N} | 128 | {t_ar[1]:.3f} | 1.0x | {usd * t_ar[1] / N * 1e6:.3f} | "
      f"{t_ar[1] / N * 1e6 * a.power / 3.6e6:.3f} | — |")
for name, kw in V.items():
    skip = kw.get("skip", ())
    passes = 11 - len(skip)
    pos = sum(l + 2 * kw.get("halo", {}).get(k, 0) for k, l in enumerate(LEVELS, 1) if k not in skip)
    t1 = ncm(1, **kw)
    cross = next((B for B in bs if t_ar[B] / (B * N) < ncm(B, **kw) / (B * N)), None)
    print(f"| {name} | {passes} | {pos} | {t1:.3f} | {t_ar[1] / t1:.1f}x | {usd * t1 / N * 1e6:.3f} | "
          f"{t1 / N * 1e6 * a.power / 3.6e6:.4f} | {cross if cross else f'never (<= {bs[-1]})'} |")

print("\n## Cost, $ per 1M output tokens, by batch\n")
print("| variant | " + " | ".join(f"batch {B}" for B in bs) + " |")
print("|---|" + "---|" * len(bs))
print("| AR (GQA) | " + " | ".join(f"{usd * t_ar[B] / (B * N) * 1e6:.3f}" for B in bs) + " |")
for name, kw in V.items():
    print(f"| {name} | " + " | ".join(f"{usd * ncm(B, **kw) / (B * N) * 1e6:.3f}" for B in bs) + " |")
