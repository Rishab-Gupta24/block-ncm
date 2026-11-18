"""Fine-tune the released NCM generator as a chunked model (Block-NCM, or a baseline via G).

    G = 0        independent chunks (B0)
    0 < G < K    Block-NCM: levels 1..G global, plan = level G, rest local
    G = K        fully global (B2)

    python experiments/train.py --M 4 --G 3 --name ours_M4_G3 --steps 3000
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.model import ChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--G", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--runs", default="~/ncm/runs")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--chunk-emb-lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--val-docs", type=int, default=512)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time-limit", type=float, default=0, help="stop after this many minutes (0 = none)")
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
run_dir = os.path.join(os.path.expanduser(a.runs), a.name)
os.makedirs(run_dir, exist_ok=True)
json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=2)

cfg, tk, net, sd = load_released(a.ckpt)
model = ChunkedNCM(net, tk, sd).cuda()
model.grad_ckpt = a.grad_ckpt
assert 0 <= a.G <= model.K, f"G must be in [0, {model.K}]"

chunked = os.path.expanduser(a.chunked)
train = np.load(os.path.join(chunked, f"train_M{a.M}.npy"), mmap_mode="r")
val = np.load(os.path.join(chunked, f"val_M{a.M}.npy"))[: a.val_docs]
print(f"train docs {len(train):,}  val docs {len(val):,}  M={a.M} G={a.G} K={model.K}", flush=True)

chunk_params = list(model.chunk_emb.parameters())
other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("chunk_emb")]
opt = torch.optim.AdamW(
    [{"params": other, "lr": a.lr}, {"params": chunk_params, "lr": a.chunk_emb_lr, "weight_decay": 0.0}],
    lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd,
)
base_lrs = [g["lr"] for g in opt.param_groups]


def lr_scale(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    prog = (step - a.warmup) / max(1, a.steps - a.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def batch(arr, idx):
    return torch.from_numpy(np.asarray(arr[np.sort(idx)], dtype=np.int64)).cuda(non_blocking=True)


@torch.no_grad()
def evaluate():
    model.eval()
    tot = {"nll": 0.0, "nll_global": 0.0, "nll_local": 0.0}
    n = 0
    for s in range(0, len(val), a.bs):
        codes = torch.from_numpy(val[s:s + a.bs].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, st = model.forward_loss(codes, a.G)
        w = len(codes)
        for k in tot:
            tot[k] += (st[k] if not math.isnan(st[k]) else 0.0) * w
        n += w
    model.train()
    return {k: v / n for k, v in tot.items()}


log = open(os.path.join(run_dir, "log.jsonl"), "a")
model.train()
t0 = time.time()
ev = evaluate()
print(f"step 0 val {ev}", flush=True)
log.write(json.dumps({"step": 0, "val": ev, "time": 0}) + "\n"); log.flush()

for step in range(a.steps):
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * lr_scale(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(a.accum):
        codes = batch(train, np.random.randint(0, len(train), a.bs))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, st = model.forward_loss(codes, a.G)
        (loss / a.accum).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 25 == 0:
        el = time.time() - t0
        print(f"step {step:5d} loss {st['nll']:.4f} glob {st['nll_global']:.4f} loc {st['nll_local']:.4f} "
              f"gn {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} {el / 60:.1f}min "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
        log.write(json.dumps({"step": step, "train": {k: st[k] for k in ("nll", "nll_global", "nll_local")},
                              "gn": float(gn), "time": el}) + "\n"); log.flush()
    last = step == a.steps - 1 or (a.time_limit and time.time() - t0 > a.time_limit * 60)
    if (step + 1) % a.eval_every == 0 or last:
        ev = evaluate()
        print(f"step {step + 1} val {ev}", flush=True)
        log.write(json.dumps({"step": step + 1, "val": ev, "time": time.time() - t0}) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "args": vars(a), "step": step + 1}, os.path.join(run_dir, "model.pt"))
    if last:
        break
print(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")
