"""Depth probe (no training): per-level NLL when the local levels run only a subset of the layers.

For each depth (6 and 3 of 12) and each layer-selection pattern (prefix / uniform-keep-last / symmetric),
local levels >= start_level run that subset (untrained layer drop); global levels run all layers. Reports
per-level NLL against full depth, and writes the best pattern per depth (lowest code-weighted NLL on the
refiner's levels) to JSON for the refiner training run.

    python experiments/depth_probe.py --run ~/ncm/runs/oursc_M4_G3 --G 3 --n 128
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import amp_dtype, load_released  # noqa: E402
from blockncm.refiner import PATTERNS, refiner_from_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--G", type=int, default=3)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--start-level", type=int, default=0, help="default G+1 (all local levels)")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="")
ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                help="autocast dtype; auto = bf16 on GPUs with native bf16, else fp16")
a = ap.parse_args()
AMP = amp_dtype(a.amp)
start = a.start_level or a.G + 1

cfg, tk, net, sd = load_released(os.path.expanduser(a.ckpt), device=a.device)
st = torch.load(os.path.join(os.path.expanduser(a.run), "model.pt"), map_location="cpu", weights_only=False)
val = np.load(os.path.expanduser(f"{a.chunked}/val_M{a.M}.npy"))[: a.n]
LT = None


def per_level(model):
    model.eval()
    tot = None
    with torch.no_grad():
        for s in range(0, len(val), a.bs):
            c = torch.from_numpy(val[s:s + a.bs].astype(np.int64)).to(a.device)
            with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None and a.device != "cpu"):
                _, stt = model.forward_loss(c, a.G)
            pl = np.array(stt["per_level"]) * len(c)
            tot = pl if tot is None else tot + pl
    return tot / len(val)


configs = [("full", None)] + [(f"{d}/12 {name}", layers) for d in (6, 3) for name, layers in PATTERNS[d].items()]
rows = {}
for name, layers in configs:
    m = refiner_from_run(net, tk, sd, st["model"], layers=layers, start_level=start).to(a.device)
    LT = m.lt
    rows[name] = per_level(m)
    del m
    torch.cuda.empty_cache() if a.device != "cpu" else None
    print(f"{name:22s} " + " ".join(f"{x:.3f}" for x in rows[name]), flush=True)

w = np.array(LT, dtype=float)
sel = np.arange(len(LT)) >= start - 1


def score(v):
    return float((v[sel] * w[sel]).sum() / w[sel].sum())


print(f"\nrefiner levels {start}-{len(LT)} (code-weighted NLL): full {score(rows['full']):.4f}")
best = {}
for d in (6, 3):
    cands = {n: score(rows[f'{d}/12 {n}']) for n in PATTERNS[d]}
    for n, v in cands.items():
        print(f"   {d}/12 {n:10s} {v:.4f}   (+{v - score(rows['full']):.3f})")
    bn = min(cands, key=cands.get)
    best[d] = {"pattern": bn, "layers": PATTERNS[d][bn], "nll": cands[bn]}
    print(f"   best {d}/12: {bn} {PATTERNS[d][bn]}")
out = a.out or os.path.join(os.path.expanduser(a.run), f"depth_probe_start{start}.json")
json.dump({"start_level": start, "per_level": {k: v.tolist() for k, v in rows.items()}, "best": best,
           "full": score(rows["full"])}, open(out, "w"), indent=2)
print("->", out)
