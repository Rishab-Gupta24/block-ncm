"""Same-GPU latency of B1 (chunks generated one after another) vs Block-NCM, for several story lengths M.

B1 has no trained model of its own, and latency does not depend on the weights. Its time is measured as M sequential
single-chunk generations (11 passes each) plus decoding, with the released generator. A real B1 would also condition
each chunk on the text before it, which makes every pass longer, so this is a LOWER bound on B1's latency.
Block-NCM (G=5, halo 4) is timed in the same process with the same harness as time_gen.py (codes + decode,
lookup fp32, decoder fp16).

    python experiments/time_b1.py --Ms 2,4,8 --batches 1,32
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import amp_dtype, load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="2,4,8")
ap.add_argument("--G", type=int, default=5)
ap.add_argument("--halo", type=int, default=4)
ap.add_argument("--batches", default="1,32")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
ap.add_argument("--out", default="~/ncm/runs/time_b1.json")
a = ap.parse_args()
AMP = amp_dtype(a.amp)


def timed(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        fn()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return float(np.median(ts))


_, tk, net, sd = load_released(a.ckpt)
ncm = FastChunkedNCM(net, tk, sd, halo=a.halo).cuda().eval()
one = FastChunkedNCM(net, tk, sd, halo=0).cuda().eval()  # single chunk: halo and plan play no role


def decode(model, codes, n):
    with torch.no_grad():
        _, zl = model.vq.idx_to_zhats(codes.reshape(n, -1), return_zlevels=True)
        with torch.autocast("cuda", dtype=torch.float16):
            return model.vqvae.decoder(zl[-1].float()).argmax(-1)


def block(B, M):
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
        codes = ncm.generate(B, M, a.G)
    return decode(ncm, codes, B * M)


def b1(B, M):
    out = []
    for _ in range(M):  # one chunk after another: M x 11 sequential passes
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            codes = one.generate(B, 1, 0)
        out.append(decode(one, codes, B))
    return out


res = {"gpu": torch.cuda.get_device_name(0), "amp": str(AMP), "G": a.G, "halo": a.halo, "rows": []}
for M in [int(x) for x in a.Ms.split(",")]:
    for B in [int(x) for x in a.batches.split(",")]:
        tb, t1 = timed(lambda: block(B, M), a.reps), timed(lambda: b1(B, M), a.reps)
        row = {"M": M, "tokens": M * ncm.chunk_len, "batch": B, "block_s": tb, "b1_lower_bound_s": t1,
               "block_s_per_sample": tb / B, "b1_s_per_sample": t1 / B, "b1_over_block": t1 / tb}
        res["rows"].append(row)
        print(row, flush=True)
print("\n| M | tokens | batch | Block-NCM s | B1 (lower bound) s | B1 / Block-NCM |\n|---|---|---|---|---|---|")
for r in res["rows"]:
    print(f"| {r['M']} | {r['tokens']} | {r['batch']} | {r['block_s']:.3f} | {r['b1_lower_bound_s']:.3f} | "
          f"{r['b1_over_block']:.2f}x |")
json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
