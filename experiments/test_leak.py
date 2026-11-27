"""Leak test for the attention masks of trained chunked models (teacher-forced, eval mode, fp32).

Replace chunk 0's codes at one level j with another document's codes, then compare every chunk's per-code NLL:
  - j is a LOCAL level (j >= G): chunks that are not neighbours of chunk 0 must be exactly unchanged at all levels;
    the neighbour (chunk 1) may change only if the model has a halo; all chunks must be unchanged at levels <= j-1
    except chunk 0 itself (inputs of a level only use coarser levels).
  - j is a GLOBAL level (j < G): other chunks should change at later levels (the plan is shared), a sanity check
    that the test can see a change at all.

    python experiments/test_leak.py --runs oursc_M4_G5_halo4 oursc_M4_G5 b0c_M4_G0 b2c_M4_G11
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.fast import FastChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()

cfg, tk, net, sd = load_released(a.ckpt)
val = np.load(os.path.expanduser(f"{a.chunked}/val_M4.npy"))
codes = torch.from_numpy(val[:a.bs].astype(np.int64)).cuda()
donor = torch.from_numpy(val[a.bs:2 * a.bs].astype(np.int64)).cuda()
failures = 0

for run in a.runs:
    st = torch.load(os.path.expanduser(f"~/ncm/runs/{run}/model.pt"), map_location="cpu", weights_only=False)
    G, halo = st["args"]["G"], st["args"].get("halo", 0)
    model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
    model.load_state_dict(st["model"], strict=True)
    K, M = model.K, codes.shape[1]

    def per_code(c):
        with torch.no_grad():
            _, s = model.forward_loss(c, G, per_code=True)
        return s["codes_nll"]  # list over levels of (B, M, l)

    base = per_code(codes)
    # perturb the TARGET codes of level j (raw columns l0 + offsets[j]); they are the input of level j + 1.
    # "earlier" = levels 0..j (cannot depend on them for other chunks), "later" = levels j+1..K-1.
    # The perturbation reaches other chunks through level j+1's input, so its kind is that of level j+1.
    levels = sorted({j for j in (0, G - 2, G - 1, G, G + 2, K - 3) if 0 <= j < K - 1})
    print(f"\n== {run}: G={G} halo={halo} K={K} M={M}")
    for j in levels:
        o, l = model.l0 + model.offsets[j], model.lt[j]
        pert = codes.clone()
        pert[:, 0, o:o + l] = donor[:, 0, o:o + l]
        new = per_code(pert)
        diff = [(n - b).abs().amax(dim=(0, 2)) for n, b in zip(new, base)]  # per level: (M,) max change per chunk
        D = torch.stack(diff).cpu().numpy()  # (K, M)
        changed = lambda m, ts: float(D[ts, m].max()) if len(ts) else 0.0
        later = list(range(j + 1, K)); earlier = list(range(0, j + 1))
        kind = "global" if j + 1 < G else "local"
        row = "  ".join(f"c{m}: earlier {changed(m, earlier):.1e} later {changed(m, later):.1e}" for m in range(M))
        print(f"  perturb chunk 0 level-{j} codes (feeds {kind} level {j + 1}): {row}")
        for m in range(1, M):
            if changed(m, earlier) > a.tol:
                print(f"    FAIL: chunk {m} changed at level <= {j} (inputs must only use coarser levels)")
                failures += 1
        if kind == "local":
            for m in range(1, M):
                allowed = (m == 1 and halo > 0)
                if not allowed and changed(m, later) > a.tol:
                    print(f"    FAIL: chunk {m} saw chunk 0 through local level {j + 1} (leak)")
                    failures += 1
            if halo > 0 and changed(1, later) <= a.tol:
                print(f"    note: halo model, but neighbour chunk 1 did not change")
        else:
            if all(changed(m, later) <= a.tol for m in range(1, M)):
                print(f"    FAIL: global level {j + 1} changed nothing in other chunks (plan not shared?)")
                failures += 1

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURES'}")
sys.exit(1 if failures else 0)
