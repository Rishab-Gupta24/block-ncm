"""Checks for the level-weighted and KL-anchored training objective (FastChunkedNCM).

    python experiments/test_anchor.py
1. alpha = 0, no anchor: the objective is the plain code-average NLL.
2. alpha = 1: the objective is the mean over levels of each level's mean NLL.
3. Anchor with dropout off, G = 0: the model equals the released model, so KL = 0 and the objective is unchanged.
4. Anchor at G = 3: KL > 0 (local levels read the plan) and gradients reach the model but not the reference.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402

torch.manual_seed(0)
ck = "~/ncm/checkpoints/tinystories"
_, tk, net, sd = load_released(ck)
model = FastChunkedNCM(net, tk, sd).cuda()
_, _, rnet, rsd = load_released(ck)
ref = FastChunkedNCM(rnet, tk, rsd).cuda().eval()
for p in ref.parameters():
    p.requires_grad_(False)
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0
    if hasattr(m, "dropout") and isinstance(m.dropout, float):
        m.dropout = 0.0  # attention-probability dropout (functional, in block_fwd)
val = np.load(os.path.expanduser("~/ncm/data/chunked/val_M4.npy"), mmap_mode="r")
codes4 = torch.from_numpy(np.asarray(val[:2], dtype=np.int64)).cuda()
codes1 = codes4[:, :1].contiguous()
model.train()
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")


with torch.no_grad():
    base, st = model.forward_loss(codes4, 0)
    model.level_alpha = 0.0
    model.kl_weight = 0.0
    torch.manual_seed(1); l0, _ = model.forward_loss(codes4, 0)
    check("alpha 0 = code average", abs(l0.item() - st["nll"]) < 1e-4, f"{l0.item():.5f} vs {st['nll']:.5f}")
    model.level_alpha = 1.0
    l1, st1 = model.forward_loss(codes4, 0)
    exp = float(np.mean(st1["per_level"]))
    check("alpha 1 = mean of level means", abs(l1.item() - exp) < 1e-4, f"{l1.item():.5f} vs {exp:.5f}")
    model.level_alpha = 0.0
    model.__dict__["ref"] = ref
    model.kl_weight = 1.0
    la, sta = model.forward_loss(codes1, 0)
    check("anchor, G=0, identical models: KL ~ 0", sta["kl"] < 1e-3, f"kl {sta['kl']:.2e}")
    check("anchor, G=0: objective ~ NLL", abs(la.item() - sta["nll"]) < 1e-3, f"{la.item():.5f} vs {sta['nll']:.5f}")

model.chunk_emb.weight.data.normal_(0, 0.02)  # make the plan matter
la, sta = model.forward_loss(codes4, 3)
la.backward()
g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
gr = sum(1 for p in ref.parameters() if p.grad is not None)
check("anchor, G=3: KL > 0 and finite", 0 < sta["kl"] < 50, f"kl {sta['kl']:.4f}")
check("gradients reach model, not reference", g > 0 and gr == 0, f"|grad| {g:.3g}, ref grads {gr}")
check("reference not in state_dict", not any(k.startswith("ref") for k in model.state_dict()))
print("ALL PASS" if ok else "SOME FAILED")
