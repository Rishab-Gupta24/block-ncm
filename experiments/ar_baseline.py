"""Autoregressive reference: sample stories from a public TinyStories AR model and score them exactly like
sample_eval.py (GPT-2 Large genPPL via the NCM repo's generation_perplexity, unigram entropy of the BPE-8192 tokens,
boundary/inner windows at multiples of 128 BPE-8192 tokens).

The AR model has no chunks, so its "seams" are arbitrary cut points: its boundary ratio is a control (should be
about 1.0, like real text). Samples are cut to the same budget as the NCM runs, M*128 BPE-8192 tokens, and at the
first end-of-text.

Not the NCM paper's AR baseline (88.6M, same architecture as the NCM generator; no public checkpoint): the default
is roneneldan/TinyStories-33M (GPT-Neo, 33M parameters, trained on TinyStories). Run at --M 1 to compare with the
paper's AR row (128 tokens, genPPL 8.89).

    python experiments/ar_baseline.py --M 4
    python experiments/ar_baseline.py --M 1
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="roneneldan/TinyStories-33M")
ap.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125M", help="used if the model repo has no tokenizer")
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--n", type=int, default=256, help="samples per seed")
ap.add_argument("--seeds", default="42,0")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--device", default="cuda")
ap.add_argument("--min-new", type=int, default=0,
                help="block end-of-text for this many AR tokens (match the NCM samples' length, ~425 BPE tokens)")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = a.device
C, M, W = 128, a.M, a.window

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:
    atok = AutoTokenizer.from_pretrained(a.model)
except Exception:
    atok = AutoTokenizer.from_pretrained(a.tokenizer)
ar = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
n_par = sum(p.numel() for p in ar.parameters())
print(f"AR model {a.model}: {n_par / 1e6:.1f}M parameters", flush=True)

_, tk, net, _ = load_released(a.ckpt, device="cpu")  # only the BPE-8192 pre-tokenizer is needed
del net
EOS = tk.pre_tokenizer.eos_token_id
enc = lambda s: tk.pre_tokenizer.encode(s)
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

# generous AR budget: GPT-Neo tokens per BPE-8192 token is about 1 on TinyStories; trimmed below
max_new = int(M * C * 1.5) + 16
seeds = [int(s) for s in a.seeds.split(",")]
samples, lat = [], []
for seed in seeds:
    torch.manual_seed(seed)
    for s in range(0, a.n, a.bs):
        B = min(a.bs, a.n - s)
        start = torch.full((B, 1), atok.eos_token_id, dtype=torch.long, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = ar.generate(start, attention_mask=torch.ones_like(start), do_sample=True, temperature=a.temp,
                              top_k=0, top_p=1.0, max_new_tokens=max_new, pad_token_id=atok.eos_token_id,
                              eos_token_id=atok.eos_token_id, min_new_tokens=a.min_new)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append((time.time() - t0, B))
        for row in out[:, 1:].tolist():
            if atok.eos_token_id in row:
                row = row[:row.index(atok.eos_token_id)]
            samples.append((seed, atok.decode(row, skip_special_tokens=True)))
    print(f"seed {seed}: {a.n} samples", flush=True)
del ar
if dev == "cuda":
    torch.cuda.empty_cache()

texts, toks, bwin, iwin, n_ar_tok = [], [], [], [], []
for _, text in samples:
    ids = [i for i in enc(text) if i != EOS][: M * C]  # same budget as the NCM samples
    texts.append(dec(ids)); toks.append(ids)
    end = len(ids)
    for m in range(M - 1):
        b = (m + 1) * C
        if b + W <= end:
            bwin.append(dec(ids[b - W:b + W]))
    for m in range(M):
        mid = m * C + C // 2
        if mid + W <= end:
            iwin.append(dec(ids[mid - W:mid + W]))

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
sys.path.insert(0, NCM_ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).to(dev).eval()


def gppl(ts):
    ts = [t for t in ts if t.strip()]
    if not ts:
        return float("nan")
    return float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs).mean_ppl)


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


per_seed = []
for seed in seeds:
    sel = [k for k, (s_, _) in enumerate(samples) if s_ == seed]
    per_seed.append({"seed": seed, "genppl": gppl([texts[k] for k in sel]),
                     "entropy": float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))})
g = np.array([p["genppl"] for p in per_seed])
res = {
    "model": a.model, "params_M": n_par / 1e6, "M": M, "temp": a.temp, "min_new": a.min_new,
    "genppl": float(g.mean()), "genppl_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
    "entropy": float(np.mean([p["entropy"] for p in per_seed])), "per_seed": per_seed,
    "boundary_genppl": gppl(bwin), "inner_genppl": gppl(iwin), "n_boundary": len(bwin), "n_inner": len(iwin),
    "mean_tokens": float(np.mean([len(t) for t in toks])),
    "frac_reaching_last_chunk": float(np.mean([len(t) > (M - 1) * C for t in toks])),
    "sequential_steps": max_new,
    "latency_s_per_sample": float(sum(t for t, _ in lat[1:]) / max(1, sum(b for _, b in lat[1:]))) if len(lat) > 1
    else float(lat[0][0] / lat[0][1]),
    "examples": texts[:8],
}
res["boundary_ratio"] = res["boundary_genppl"] / res["inner_genppl"]
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
out = a.out or os.path.expanduser(f"~/ncm/runs/ar_baseline/{a.model.split('/')[-1]}_M{M}_t{a.temp}_min{a.min_new}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
print("->", out)
