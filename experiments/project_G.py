"""Project the NLL of untrained G values from measured per-level NLL (a simulation, not a measurement).

Assumption: in a Block-NCM with G global levels, level t <= G behaves like the fully global run (B2) and
level t > G like the local levels of the measured Block-NCM run (local with a plan). A larger G also gives a
richer plan, which should help local levels further, so the projection is conservative for local levels and
optimistic for global ones (B2 trained every level globally).

    python experiments/project_G.py --breakdown ~/ncm/runs/nll_breakdown_M4.json \
        --b0 b0_M4_G0 --ours ours_M4_G3 --b2 b2_M4_G11
"""
import argparse
import json
import os

LEVELS = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]

ap = argparse.ArgumentParser()
ap.add_argument("--breakdown", default="~/ncm/runs/nll_breakdown_M4.json")
ap.add_argument("--b0", default="b0_M4_G0")
ap.add_argument("--ours", default="ours_M4_G3")
ap.add_argument("--b2", default="b2_M4_G11")
ap.add_argument("--tag", default="nonpad", choices=["all", "nonpad"])
a = ap.parse_args()

r = json.load(open(os.path.expanduser(a.breakdown)))
pl = {k: r[n][a.tag]["per_level"] for k, n in (("b0", a.b0), ("ours", a.ours), ("b2", a.b2))}
G_ours = r[a.ours]["G"]
w = LEVELS  # codes per level per chunk (padding share differs slightly by level; see check below)


def total(per_level):
    return sum(x * l for x, l in zip(per_level, w)) / sum(w)


print(f"check (level-weighted total vs measured, {a.tag}):")
for k, n in (("b0", a.b0), ("ours", a.ours), ("b2", a.b2)):
    print(f"   {k:5s} weighted {total(pl[k]):.4f}   measured {r[n][a.tag]['total']:.4f}")
print("\nper level: which is better, global (B2) or local-with-plan (Ours, levels > G_ours)?")
for t in range(len(w)):
    loc = pl["ours"][t] if t >= G_ours else float("nan")
    print(f"   level {t + 1:2d} (l={w[t]:3d})  B0 {pl['b0'][t]:.4f}  Ours {pl['ours'][t]:.4f}  B2 {pl['b2'][t]:.4f}  "
          f"{'global better' if pl['b2'][t] < pl['ours'][t] else 'local better'}  (diff {pl['b2'][t] - pl['ours'][t]:+.4f})")

print(f"\nprojected total NLL by G (levels <= G from B2, levels > G from Ours' local levels; G < {G_ours} uses B0 "
      f"for levels between G and {G_ours}):")
b0t, b2t = total(pl["b0"]), total(pl["b2"])
for G in range(0, len(w) + 1):
    proj = []
    for t in range(len(w)):
        if t < G:
            proj.append(pl["b2"][t])
        elif t >= G_ours:
            proj.append(pl["ours"][t])
        else:  # local level below the measured plan level: no measurement with a plan, use B0 (no plan)
            proj.append(pl["b0"][t])
    p = total(proj)
    mark = "  <- measured" if G == G_ours else ""
    print(f"   G={G:2d}  {p:.4f}   vs B2 {p - b2t:+.4f}   share of B0->B2 gap {(b0t - p) / (b0t - b2t):6.0%}{mark}")
