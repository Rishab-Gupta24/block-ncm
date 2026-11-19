"""Checks for blockncm.refiner.RefinerNCM (light: 1 document, M=2; safe on CPU).

  [1] all 12 layers == FastChunkedNCM (exact)
  [2] layer subsets change only the refiner's levels (levels below start_level unchanged)
  [3] with freeze_base + distillation, only the refiner copy gets gradients; KL is finite and positive
  [4] plan K/V are taken from the kept layer ids (a subset whose ids are permuted gives a different loss)
  [5] generation runs with a 6-layer refiner

    python experiments/test_refiner.py --device cpu
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.refiner import RefinerNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
dev = a.device

cfg, tk, net, sd = load_released(os.path.expanduser(a.ckpt), device=dev)
fast = FastChunkedNCM(net, tk, sd).to(dev).eval()
codes = torch.from_numpy(np.load(os.path.expanduser(f"{a.chunked}/val_M4.npy"))[:1, :2].astype(np.int64)).to(dev)

with torch.no_grad():
    full = RefinerNCM(net, tk, sd, layers=None, start_level=4).to(dev).eval()
    lf, sf = fast.forward_loss(codes, 3)
    lr_, sr = full.forward_loss(codes, 3)
    d = max(abs(x - y) for x, y in zip(sf["per_level"], sr["per_level"]))
    print(f"[1] 12/12 layers: fast {lf.item():.6f} refiner {lr_.item():.6f} max level diff {d:.2e}")
    assert d < 1e-4

    six = RefinerNCM(net, tk, sd, layers=[0, 2, 4, 6, 8, 11], start_level=7).to(dev).eval()
    _, s6 = six.forward_loss(codes, 3)
    below = max(abs(x - y) for x, y in zip(sf["per_level"][:6], s6["per_level"][:6]))
    print(f"[2] 6-layer refiner from level 7: levels 1-6 diff {below:.2e} (expect 0); levels 7-11 "
          f"{[round(x, 3) for x in s6['per_level'][6:]]} vs full {[round(x, 3) for x in sf['per_level'][6:]]}")
    assert below < 1e-4

six.train()
six.freeze_base()
six.distill = 1.0
loss, st = six.forward_loss(codes, 3)
loss.backward()
base_g = sum(p.grad is not None for p in six.net.parameters()) + (six.chunk_emb.weight.grad is not None)
ref_g = sum(float(p.grad.abs().sum()) for p in six.ref_blocks.parameters() if p.grad is not None)
print(f"[3] distill: KL {st['kl']:.4f}; base params with grad {base_g} (expect 0); |grad| refiner {ref_g:.3e}")
assert base_g == 0 and ref_g > 0 and st["kl"] > 0
six.eval()

with torch.no_grad():
    a_ = RefinerNCM(net, tk, sd, layers=[0, 2, 4, 6, 8, 11], start_level=4).to(dev).eval()
    b_ = RefinerNCM(net, tk, sd, layers=[0, 2, 4, 6, 8, 11], start_level=4).to(dev).eval()
    b_.layers = [11, 8, 6, 4, 2, 0]  # same blocks, wrong plan K/V layer ids
    la, _ = a_.forward_loss(codes, 3)
    lb, _ = b_.forward_loss(codes, 3)
    print(f"[4] plan K/V from kept layer ids {la.item():.4f} vs mismatched ids {lb.item():.4f} (should differ)")
    assert abs(la.item() - lb.item()) > 1e-3
    torch.manual_seed(0)
    out = a_.generate(1, 2, 3, device=dev)
    print(f"[5] generate with 6-layer refiner: {tuple(out.shape)}")
print("all refiner checks passed")
