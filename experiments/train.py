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
from blockncm.load import amp_dtype, load_released  # noqa: E402
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
ap.add_argument("--corrupt-mean", type=float, default=0.0,
                help="conditioning corruption, mean replacement rate over levels (NCM paper selected 0.5)")
ap.add_argument("--corrupt-span", type=float, default=0.5,
                help="rate falls linearly from mean+span/2 (level 0) to mean-span/2 (level K-1)")
ap.add_argument("--keep-ckpts", action="store_true", help="also save model_step{N}.pt at every evaluation")
ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                help="autocast dtype; auto = bf16 on GPUs with native bf16, else fp16 with gradient scaling")
ap.add_argument("--data-suffix", default="", help="e.g. _sent for sentence-aligned chunks (train_M4_sent.npy)")
ap.add_argument("--fast", action="store_true", help="FastChunkedNCM: KV-cached plan (exact, faster)")
ap.add_argument("--halo", type=int, default=0, help="halo attention width for local levels (implies --fast)")
ap.add_argument("--plan-adapter", type=int, default=0,
                help="LoRA rank for plan-only adapters: frozen generator, LoRA on global passes, local passes frozen")
ap.add_argument("--lora-alpha", type=float, default=0, help="LoRA alpha (default = rank, i.e. scale 1)")
ap.add_argument("--plan-bias", type=float, default=-6.0, help="initial logit bias of local queries on plan keys")
ap.add_argument("--plan-bias-lr", type=float, default=1e-2, help="learning rate of the plan bias")
ap.add_argument("--refiner-layers", default="", help="layer-drop refiner: comma-separated layer ids, e.g. 0,2,4,6,8,11")
ap.add_argument("--refiner-start-level", type=int, default=0, help="first (1-based) target level using the refiner "
                                                                     "(default G+1 = all local levels)")
ap.add_argument("--distill", type=float, default=1.0, help="weight of KL(full-depth teacher || refiner)")
ap.add_argument("--init-from", default="", help="run dir whose model.pt initialises the model (refiner: the teacher)")
ap.add_argument("--min-len", type=int, default=0,
                help="keep only documents with at least this many tokens (needs *_len.npy from prepare_chunks)")
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
run_dir = os.path.join(os.path.expanduser(a.runs), a.name)
os.makedirs(run_dir, exist_ok=True)
json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=2)

cfg, tk, net, sd = load_released(a.ckpt)
if a.refiner_layers:
    from blockncm.refiner import refiner_from_run
    assert a.init_from, "--refiner-layers needs --init-from (the full-depth teacher run)"
    a.refiner_start_level = a.refiner_start_level or a.G + 1
    assert a.refiner_start_level > a.G, "the refiner replaces local levels only"
    init = torch.load(os.path.join(os.path.expanduser(a.init_from), "model.pt"), map_location="cpu", weights_only=False)
    model = refiner_from_run(net, tk, sd, init["model"], layers=[int(x) for x in a.refiner_layers.split(",")],
                             start_level=a.refiner_start_level, distill=a.distill).cuda()
    model.freeze_base()
    print(f"refiner: layers {model.layers} from level {model.start_level}, distill {a.distill}, teacher {a.init_from}; "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
elif a.plan_adapter:
    from blockncm.adapters import PlanAdapterNCM
    a.lora_alpha = a.lora_alpha or a.plan_adapter
    model = PlanAdapterNCM(net, tk, sd, halo=a.halo, r=a.plan_adapter, alpha=a.lora_alpha,
                           plan_bias=a.plan_bias).cuda()
    print(f"plan-only adapters: trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"of {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
elif a.fast or a.halo:
    from blockncm.fast import FastChunkedNCM
    model = FastChunkedNCM(net, tk, sd, halo=a.halo).cuda()
else:
    model = ChunkedNCM(net, tk, sd).cuda()
model.grad_ckpt = a.grad_ckpt
if a.corrupt_mean > 0:
    model.corrupt = ChunkedNCM.linear_corruption(model.K, a.corrupt_mean, a.corrupt_span)
    print("conditioning corruption per level:", [round(p, 3) for p in model.corrupt], flush=True)
assert 0 <= a.G <= model.K, f"G must be in [0, {model.K}]"

chunked = os.path.expanduser(a.chunked)
train = np.load(os.path.join(chunked, f"train_M{a.M}{a.data_suffix}.npy"), mmap_mode="r")
val = np.load(os.path.join(chunked, f"val_M{a.M}{a.data_suffix}.npy"))
if a.min_len:
    tl = np.load(os.path.join(chunked, f"train_M{a.M}{a.data_suffix}_len.npy"))
    vl = np.load(os.path.join(chunked, f"val_M{a.M}{a.data_suffix}_len.npy"))
    assert len(tl) == len(train) and len(vl) == len(val), "length files do not match the code arrays"
    train = train[np.nonzero(tl >= a.min_len)[0]]  # fancy indexing loads the subset into RAM
    val = val[vl >= a.min_len]
val = val[: a.val_docs]
print(f"train docs {len(train):,}  val docs {len(val):,}  M={a.M} G={a.G} K={model.K}", flush=True)

chunk_params = list(model.chunk_emb.parameters())
other = [p for n, p in model.named_parameters()
         if p.requires_grad and not n.startswith("chunk_emb") and n != "plan_bias"]
groups = [{"params": other, "lr": a.lr}, {"params": chunk_params, "lr": a.chunk_emb_lr, "weight_decay": 0.0}]
if hasattr(model, "plan_bias"):  # Adam moves a parameter ~lr per step: the bias needs a large lr to open
    groups.append({"params": [model.plan_bias], "lr": a.plan_bias_lr, "weight_decay": 0.0})
opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
base_lrs = [g["lr"] for g in opt.param_groups]
AMP = amp_dtype(a.amp)
scaler = torch.amp.GradScaler("cuda", enabled=AMP is torch.float16)
print(f"autocast: {AMP}", flush=True)


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
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            _, st = model.forward_loss(codes, a.G)
        if not math.isfinite(st["nll"]):  # nll_global/local are nan by design when G=0 or G=K; nll never is
            raise RuntimeError(f"non-finite validation loss {st['nll']} (autocast {AMP}): numerical overflow")
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
        with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
            loss, st = model.forward_loss(codes, a.G)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step} (autocast {AMP}): numerical overflow")
        scaler.scale(loss / a.accum).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)  # without fp16 the scaler is disabled: plain opt.step()
    scaler.update()

    if step % 25 == 0:
        el = time.time() - t0
        pb = float(model.plan_bias.mean()) if hasattr(model, "plan_bias") else None
        print(f"step {step:5d} loss {st['nll']:.4f} glob {st['nll_global']:.4f} loc {st['nll_local']:.4f} "
              f"gn {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} {el / 60:.1f}min "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB" + (f" plan_bias {pb:.2f}" if pb is not None else "")
              + (f" kl {st['kl']:.4f}" if "kl" in st else ""), flush=True)
        log.write(json.dumps({"step": step, "train": {k: st[k] for k in ("nll", "nll_global", "nll_local")},
                              "gn": float(gn), "time": el, "plan_bias": pb}) + "\n"); log.flush()
    last = step == a.steps - 1 or (a.time_limit and time.time() - t0 > a.time_limit * 60)
    if (step + 1) % a.eval_every == 0 or last:
        ev = evaluate()
        print(f"step {step + 1} val {ev}", flush=True)
        log.write(json.dumps({"step": step + 1, "val": ev, "time": time.time() - t0}) + "\n"); log.flush()
        ck = {"model": model.state_dict(), "args": vars(a), "step": step + 1}
        torch.save(ck, os.path.join(run_dir, "model.pt"))
        if a.keep_ckpts and not last:
            torch.save(ck, os.path.join(run_dir, f"model_step{step + 1}.pt"))
    if last:
        break
print(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")
