"""Checks for blockncm.fast.FastChunkedNCM against ChunkedNCM (runs on CPU or GPU).

  [1] forward_loss equal (KV-cached plan == recomputed plan), M=4, G in {0, 1, 3, 6, 11}
  [2] halo with M=1 changes nothing (no neighbours), and halo changes the loss when M > 1
  [3] generation with the same seed gives the same codes (halo 0)
  [4] plan search (n_plans=4) runs and returns the right shape
  [5] timing of forward+backward, ChunkedNCM vs FastChunkedNCM

    python experiments/test_fast.py --device cpu
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.model import ChunkedNCM  # noqa: E402
from blockncm.fast import FastChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--device", default="cpu")
ap.add_argument("--bs", type=int, default=2, help="documents per check (1 keeps CPU memory low)")
ap.add_argument("--no-timing", action="store_true", help="skip [5] (forward+backward timing)")
a = ap.parse_args()
dev = a.device

cfg, tk, net, sd = load_released(os.path.expanduser(a.ckpt), device=dev)
ref = ChunkedNCM(net, tk, sd).to(dev).eval()
fast = FastChunkedNCM(net, tk, sd).to(dev).eval()
torch.manual_seed(0)
w = torch.randn_like(ref.chunk_emb.weight) * 0.02  # non-zero chunk embeddings so the path is exercised
ref.chunk_emb.weight.data.copy_(w); fast.chunk_emb.weight.data.copy_(w)
codes = torch.from_numpy(np.load(os.path.expanduser(f"{a.chunked}/val_M4.npy"))[:a.bs].astype(np.int64)).to(dev)

with torch.no_grad():
    for G in (0, 1, 3, 6, ref.K):
        lr_, sr = ref.forward_loss(codes, G)
        lf, sf = fast.forward_loss(codes, G)
        dl = max(abs(p - q) for p, q in zip(sr["per_level"], sf["per_level"]))
        print(f"[1] G={G:2d} ref {lr_.item():.6f} fast {lf.item():.6f} max level diff {dl:.2e}")
        assert dl < 1e-4, "fast path differs"

    c1 = codes[:, :1]
    fast.halo = 8
    for G in (0, 3):
        l0_, _ = ref.forward_loss(c1, G)
        lh, _ = fast.forward_loss(c1, G)
        print(f"[2] M=1 G={G} halo=8: ref {l0_.item():.6f} halo {lh.item():.6f} diff {abs(l0_.item() - lh.item()):.2e}")
        assert abs(l0_.item() - lh.item()) < 1e-4
    lh4, sh = fast.forward_loss(codes, 3)
    fast.halo = 0
    l4, s4 = fast.forward_loss(codes, 3)
    print(f"[2] M=4 G=3: halo=8 {lh4.item():.6f} vs halo=0 {l4.item():.6f} (should differ; untrained halo)")

    for G in (0, 3, ref.K):
        torch.manual_seed(1); g1 = ref.generate(2, 2, G, device=dev)
        torch.manual_seed(1); g2 = fast.generate(2, 2, G, device=dev)
        print(f"[3] generate M=2 G={G:2d}: code match {(g1 == g2).float().mean().item():.4f}")
    torch.manual_seed(2); gp = fast.generate(2, 2, 3, device=dev, n_plans=4)
    print(f"[4] plan search n_plans=4: codes {tuple(gp.shape)}")
    fast.halo = 8; torch.manual_seed(3); gh = fast.generate(2, 4, 3, device=dev); fast.halo = 0
    print(f"[4] halo=8 generation M=4: codes {tuple(gh.shape)}")

for name, m in (() if a.no_timing else (("ChunkedNCM", ref), ("FastChunkedNCM", fast))):
    m.train()
    ts = []
    for _ in range(3):
        t0 = time.time()
        loss, _ = m.forward_loss(codes, 3)
        loss.backward()
        net.zero_grad(set_to_none=True)
        if dev != "cpu":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    m.eval()
    print(f"[5] {name:15s} forward+backward G=3 M=4 B=2: {min(ts):.2f} s")
print("all fast-path checks passed")
