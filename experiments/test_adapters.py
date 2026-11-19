"""Checks for blockncm.adapters.PlanAdapterNCM (light: 1 document, M=2; safe on CPU).

  [1] zero LoRA + zero plan bias == FastChunkedNCM (G=3)
  [2] plan bias -8 at init: local levels ~= no-plan model (G=0); global levels == FastChunkedNCM
  [3] gradients reach LoRA (via global AND local loss) and the plan bias; generator weights get none
  [4] local-only loss still trains the global LoRA (through the cached plan K/V)
  [5] generation runs

    python experiments/test_adapters.py --device cpu
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.adapters import PlanAdapterNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
dev = a.device

cfg, tk, net, sd = load_released(os.path.expanduser(a.ckpt), device=dev)
fast = FastChunkedNCM(net, tk, sd).to(dev).eval()
pa = PlanAdapterNCM(net, tk, sd, plan_bias=0.0).to(dev).eval()
codes = torch.from_numpy(np.load(os.path.expanduser(f"{a.chunked}/val_M4.npy"))[:1, :2].astype(np.int64)).to(dev)

with torch.no_grad():
    lf, sf = fast.forward_loss(codes, 3)
    lp, sp = pa.forward_loss(codes, 3)
    d = max(abs(x - y) for x, y in zip(sf["per_level"], sp["per_level"]))
    print(f"[1] zero adapters, zero bias: fast {lf.item():.6f} adapter {lp.item():.6f} max level diff {d:.2e}")
    assert d < 1e-4

    pa.plan_bias.data.fill_(-8.0)
    _, s8 = pa.forward_loss(codes, 3)
    _, s0 = fast.forward_loss(codes, 0)
    loc8, loc0, locf = (np.mean(s["per_level"][3:]) for s in (s8, s0, sf))
    glob = max(abs(x - y) for x, y in zip(s8["per_level"][:3], sf["per_level"][:3]))
    print(f"[2] bias -8: local-level NLL {loc8:.4f} | no plan (G=0) {loc0:.4f} | plan, no bias {locf:.4f}; "
          f"global levels diff vs fast {glob:.2e}")

pa.train()
for p in pa.lora.parameters():  # non-zero B so A also receives gradient in the check
    p.data.add_(torch.randn_like(p) * 1e-3)
loss, _ = pa.forward_loss(codes, 3)
loss.backward()
gen_grads = sum(p.grad is not None for p in pa.net.parameters())
lora_g = sum(float(p.grad.abs().sum()) for p in pa.lora.parameters() if p.grad is not None)
print(f"[3] generator params with grad: {gen_grads} (expect 0); |grad| LoRA {lora_g:.3e}; "
      f"plan bias {float(pa.plan_bias.grad.abs().sum()):.3e}; chunk emb {float(pa.chunk_emb.weight.grad.abs().sum()):.3e}")
assert gen_grads == 0 and lora_g > 0 and pa.plan_bias.grad.abs().sum() > 0

pa.zero_grad(set_to_none=True)
_, st = pa.forward_loss(codes, 3, per_code=True)
local_loss = torch.cat([c.flatten() for c in st["codes_nll"][3:]]).mean()
local_loss.backward()
g_lora = sum(float(p.grad.abs().sum()) for p in pa.lora.parameters() if p.grad is not None)
print(f"[4] local-level loss only: |grad| on global LoRA {g_lora:.3e} (> 0 means the plan K/V carry gradient)")
assert g_lora > 0
pa.eval()

with torch.no_grad():
    torch.manual_seed(0)
    out = pa.generate(1, 2, 3, device=dev)
    print(f"[5] generate M=2 G=3: {tuple(out.shape)}")
    # [6] after "training" (random adapters, chunk embeddings and an open bias), a single chunk with G=0
    # has no plan: its local levels must equal the released model exactly
    pa.chunk_emb.weight.data.normal_(0, 0.1)
    pa.plan_bias.data.fill_(2.0)
    c1 = codes[:, :1]
    lr_, sr = fast.forward_loss(c1, 0)
    la, sa = pa.forward_loss(c1, 0)  # adapters stay random: the prefix gets no adapter delta
    print(f"[6] single chunk, no plan, random adapters + chunk emb + open bias: released {lr_.item():.6f} "
          f"adapter {la.item():.6f}")
    assert abs(lr_.item() - la.item()) < 1e-4
    # [7] with G=3, local levels with the bias closed are the released local levels (prefix unadapted)
    pa.plan_bias.data.fill_(-30.0)
    _, s3 = pa.forward_loss(codes, 3)
    _, s0 = fast.forward_loss(codes, 0)
    d7 = max(abs(x - y) for x, y in zip(s3["per_level"][3:], s0["per_level"][3:]))
    print(f"[7] G=3, bias closed (-30), random adapters: local levels vs released (no plan) max diff {d7:.2e}")
    assert d7 < 1e-3
print("all adapter checks passed")
