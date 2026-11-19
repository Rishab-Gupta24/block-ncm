"""Teacher-forced NLL breakdown of trained chunked models on the validation split.

For each run reports, over all codes and over non-padding codes only:
  * total NLL, coarse levels (1..G_ref) vs fine levels, every level separately
  * NLL per chunk position (chunk 1..M)

A level-t code (length l_t) is treated as padding when every token of its chunk it covers,
[floor(j*C/l_t), ceil((j+1)*C/l_t)), is PAD. Tokens come from the
source token windows, rebuilt with prepare_chunks.py's selection.

    python experiments/nll_breakdown.py --M 4 --runs b0_M4_G0:0 ours_M4_G3:3 b2_M4_G11:11
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import amp_dtype, load_released  # noqa: E402
from blockncm.model import ChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--runs", nargs="+", required=True, help="name:G pairs (dirs under --runs-dir)")
ap.add_argument("--runs-dir", default="~/ncm/runs")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--chunked", default="~/ncm/data/chunked")
ap.add_argument("--data", default="~/ncm/data/tinystories")
ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--max-docs", type=int, default=0, help="as passed to prepare_chunks.py (0 = all)")
ap.add_argument("--coarse", type=int, default=3, help="levels 1..coarse count as coarse")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", default="")
ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                help="autocast dtype; auto = bf16 on GPUs with native bf16, else fp16")
a = ap.parse_args()
AMP = amp_dtype(a.amp)

cfg, tk, net, sd = load_released(a.ckpt)
from blockncm.fast import FastChunkedNCM  # noqa: E402  (exactly ChunkedNCM at halo 0; also scores halo runs)
model = FastChunkedNCM(net, tk, sd).cuda().eval()
base_state = {k: v.clone() for k, v in model.state_dict().items()}
val = np.load(os.path.join(os.path.expanduser(a.chunked), f"val_M{a.M}.npy"))
C, K, lt, PAD = model.chunk_len, model.K, model.lt, model.pad
print(f"val docs {len(val)}  M={a.M}", flush=True)

# padding masks per level: (N, M, l_t) True = real content. The token windows are rebuilt from the
# source data with prepare_chunks.py's exact selection (the tokenizer does not decode PAD reliably).
data = os.path.expanduser(a.data)
off = np.load(os.path.join(data, "train_bpe_8192_offsets.npy")).astype(np.int64)
tok = np.memmap(os.path.join(data, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
EOS = model.eos
eligible = np.nonzero(np.diff(off) > (a.M - 1) * C)[0]
if a.max_docs:  # must match prepare_chunks.py --max-docs
    eligible = eligible[: a.max_docs]
vidx = eligible[eligible % a.val_every == 0]
assert len(vidx) == len(val), f"rebuilt {len(vidx)} val docs, file has {len(val)}"


def window(i):
    ids = tok[off[i]:off[i + 1]].tolist()
    if len(ids) < a.M * C and (not ids or ids[-1] != EOS):
        ids = ids + [EOS]
    ids = ids[: a.M * C]
    return ids + [PAD] * (a.M * C - len(ids))


toks = torch.tensor([window(i) for i in vidx]).view(len(vidx), a.M, C)
with torch.no_grad():  # sanity check: re-encoding the first docs reproduces the stored codes
    n_chk = min(32, len(val))
    chk = model.vqvae.encode(toks[:n_chk].reshape(-1, C).cuda()).view(n_chk, a.M, -1).cpu().numpy()
    per_doc = (chk == val[:n_chk]).reshape(n_chk, -1).mean(1)
    lvl0 = (chk[..., :model.l0] == val[:n_chk, :, :model.l0]).mean()
    print(f"stored vs re-encoded codes, {n_chk} docs: mean {per_doc.mean():.3f}  min doc {per_doc.min():.3f}  "
          f"level 0 {lvl0:.3f}  (a mismatched document would be near 0)")
real_tok = toks != PAD
real = []
for l in lt:
    m = torch.zeros(len(val), a.M, l, dtype=torch.bool)
    for j in range(l):
        lo, hi = math.floor(j * C / l), math.ceil((j + 1) * C / l)
        m[:, :, j] = real_tok[:, :, lo:hi].any(-1)
    real.append(m)
print("fraction of non-padding codes per chunk:",
      [round(float(torch.cat([r[:, k].flatten() for r in real]).float().mean()), 3) for k in range(a.M)])

results, doc_nll = {}, {}
for spec in a.runs:
    name, G = spec.rsplit(":", 1)
    G = int(G)
    m_ = model
    if name == "released":  # the released weights, zero-shot
        st = {"step": 0}
        model.load_state_dict(base_state, strict=True)
    else:
        st = torch.load(os.path.join(os.path.expanduser(a.runs_dir), name, "model.pt"), map_location="cpu",
                        weights_only=False)
        ra = st.get("args", {})
        if ra.get("refiner_layers"):  # layer-drop refiner run: its own model class (extra refiner blocks)
            from blockncm.refiner import RefinerNCM
            m_ = RefinerNCM(net, tk, sd, layers=[int(x) for x in ra["refiner_layers"].split(",")],
                            start_level=ra["refiner_start_level"]).cuda().eval()
        elif ra.get("plan_adapter", 0):  # plan-only adapter run: its own model class (extra LoRA / bias params)
            from blockncm.adapters import PlanAdapterNCM
            m_ = PlanAdapterNCM(net, tk, sd, r=ra["plan_adapter"], alpha=ra.get("lora_alpha", ra["plan_adapter"]),
                                plan_bias=ra.get("plan_bias", -8.0)).cuda().eval()
        m_.load_state_dict(st["model"], strict=True)
    m_.halo = st.get("args", {}).get("halo", 0) if name != "released" else 0
    nll = [[] for _ in range(K)]
    with torch.no_grad():
        for s in range(0, len(val), a.bs):
            c = torch.from_numpy(val[s:s + a.bs].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
                _, stt = m_.forward_loss(c, G, per_code=True)
            for t in range(K):
                nll[t].append(stt["codes_nll"][t].float().cpu())
    nll = [torch.cat(x) for x in nll]  # each (N, M, l_t)

    def agg(levels, use_mask, chunk=None):
        num = den = 0.0
        for t in levels:
            x, m = nll[t], (real[t] if use_mask else torch.ones_like(real[t]))
            if chunk is not None:
                x, m = x[:, chunk], m[:, chunk]
            num += float((x * m).sum()); den += float(m.sum())
        return num / den

    allv, coarse, fine = range(K), range(a.coarse), range(a.coarse, K)
    r = {"G": G, "step": st.get("step")}
    for tag, mk in (("all", False), ("nonpad", True)):
        r[tag] = {
            "total": agg(allv, mk), "coarse": agg(coarse, mk), "fine": agg(fine, mk),
            "per_level": [agg([t], mk) for t in allv],
            "per_chunk": [agg(allv, mk, k) for k in range(a.M)],
        }
    # per-document NLL over non-padding codes (for paired comparisons)
    num = sum((nll[t] * real[t]).sum((1, 2)) for t in allv)
    den = sum(real[t].sum((1, 2)) for t in allv)
    doc_nll[name] = (num / den).numpy()
    results[name] = r
    print(name, json.dumps({k: round(v, 4) for k, v in r["nonpad"].items() if not isinstance(v, list)}), flush=True)

model.load_state_dict(base_state)
names = [s.rsplit(":", 1)[0] for s in a.runs]


def table(tag):
    rows = [f"| | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    def row(label, f):
        rows.append(f"| {label} | " + " | ".join(f"{f(results[n][tag]):.4f}" for n in names) + " |")
    row("total", lambda d: d["total"])
    row(f"coarse (levels 1-{a.coarse})", lambda d: d["coarse"])
    row(f"fine (levels {a.coarse + 1}-{K})", lambda d: d["fine"])
    for t in range(K):
        row(f"level {t + 1} (l={lt[t]})", lambda d, t=t: d["per_level"][t])
    for k in range(a.M):
        row(f"chunk {k + 1}", lambda d, k=k: d["per_chunk"][k])
    return "\n".join(rows)


for tag in ("all", "nonpad"):
    print(f"\n### NLL ({'all codes' if tag == 'all' else 'non-padding codes only'})\n")
    print(table(tag))

# paired per-document differences vs the first and last run (B0 and B2), 95% bootstrap CI over documents
rng = np.random.default_rng(0)
boot = rng.integers(0, len(val), size=(a.n_boot, len(val)))
print(f"\n### Paired per-document NLL difference (non-padding), 95% bootstrap CI, {len(val)} docs\n")
print("| run | vs " + names[0] + " | vs " + names[-1] + " |\n|---|---|---|")
for n in names:
    cells = []
    for ref in (names[0], names[-1]):
        d = doc_nll[n] - doc_nll[ref]
        bs = d[boot].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cells.append(f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  wins {np.mean(d < 0):.0%}")
        results[n].setdefault("paired", {})[ref] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                                                   "frac_docs_better": float(np.mean(d < 0))}
    print(f"| {n} | {cells[0]} | {cells[1]} |")
out = a.out or os.path.join(os.path.expanduser(a.runs_dir), f"nll_breakdown_M{a.M}.json")
json.dump(results, open(out, "w"), indent=2)
print("->", out)
