"""Best-of-K reranking of unconditional Block-NCM samples with a separate AR scorer (no training).

Samples come from a pool written by sample_eval.py (``..._samples.json``). Within each seed, consecutive samples are
grouped K at a time; the AR scorer (TinyStories-33M, a different model from the GPT-2 Large judge) picks the sample
with the lowest mean token NLL. The picks are then scored by GPT-2 Large exactly as in sample_eval.py and compared
with the first sample of each group (same count, no selection). Entropy, length and distinct 4-grams are reported
too, since likelihood selection can favour short or bland text.

    python experiments/rerank_bestofk.py --samples ~/ncm/runs/oursc_M4_G5_halo4/pool_samples.json --Ks 1,2,4,8
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
ap.add_argument("--samples", required=True)
ap.add_argument("--Ks", default="1,2,4,8")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--scorer", default="roneneldan/TinyStories-33M")
ap.add_argument("--scorer-tok", default="EleutherAI/gpt-neo-125M")
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--out", default="")
a = ap.parse_args()

_, tk, _, _ = load_released(a.ckpt)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)


def full_tokens(chunks):
    flat = [i for c in chunks for i in c]
    if EOS in flat:
        flat = flat[:flat.index(EOS)]
    return [i for i in flat if i != PAD]


d = json.load(open(os.path.expanduser(a.samples)))
seeds = [s for s, _ in d["samples"]]
toks = [full_tokens(ch) for _, ch in d["samples"]]
texts = [dec(t) for t in toks]
print(f"{len(texts)} samples, seeds {sorted(set(seeds))}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# ---- AR scorer: mean token NLL per sample
stok = AutoTokenizer.from_pretrained(a.scorer_tok)
slm = AutoModelForCausalLM.from_pretrained(a.scorer).cuda().eval()
nll = []
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(texts), 32):
        enc = [[stok.eos_token_id] + stok(t)["input_ids"][:600] for t in texts[i:i + 32]]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), stok.eos_token_id)
        mask = torch.zeros((len(enc), L))
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e)
            mask[j, :len(e)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        logits = slm(ids, attention_mask=mask).logits[:, :-1].float()
        lp = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
        m = mask[:, 1:]
        nll += ((lp * m).sum(1) / m.sum(1).clamp(min=1)).tolist()
torch.cuda.synchronize()
score_time = time.time() - t0
del slm
torch.cuda.empty_cache()

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
sys.path.insert(0, NCM_ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).cuda().eval()


def gppl(idx):
    ts = [texts[k] for k in idx if texts[k].strip()]
    return float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs).mean_ppl)


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def distinct4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 1.0


def seam_ratio(idx, W=32):
    """boundary / inner window genPPL, as in sample_eval.py (windows only before the story ends)"""
    bw, iw = [], []
    for k in idx:
        ch = d["samples"][k][1]
        flat, starts = [], []
        for c in ch:
            starts.append(len(flat))
            flat += [i for i in c if i != PAD]
        starts.append(len(flat))
        end = flat.index(EOS) if EOS in flat else len(flat)
        cl = lambda ids: [i for i in ids if i not in (EOS, PAD)]
        for m in range(len(ch) - 1):
            b = starts[m + 1]
            if b - W >= 0 and b + W <= end:
                bw.append(dec(cl(flat[b - W:b + W])))
        for m in range(len(ch)):
            mid = (starts[m] + starts[m + 1]) // 2
            if mid - W >= 0 and mid + W <= end:
                iw.append(dec(cl(flat[mid - W:mid + W])))
    g = lambda ts: float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs).mean_ppl)
    return g(bw) / g(iw)


def stats(idx):
    return {"n": len(idx), "genppl": gppl(idx), "seam_ratio": seam_ratio(idx),
            "entropy": float(np.mean([entropy(toks[k]) for k in idx if toks[k]])),
            "mean_tokens": float(np.mean([len(toks[k]) for k in idx])),
            "distinct4": float(np.mean([distinct4(toks[k]) for k in idx if len(toks[k]) > 3])),
            "scorer_nll": float(np.mean([nll[k] for k in idx]))}


res = {"samples": a.samples, "scorer": a.scorer, "scorer_seconds_total": score_time,
       "scorer_seconds_per_sample": score_time / len(texts), "K": {}}
for K in [int(x) for x in a.Ks.split(",")]:
    first, pick = [], []
    for s in sorted(set(seeds)):
        idx = [k for k, s_ in enumerate(seeds) if s_ == s]
        for g in range(0, len(idx) - K + 1, K):
            grp = idx[g:g + K]
            first.append(grp[0])
            pick.append(min(grp, key=lambda k: nll[k]))
    res["K"][K] = {"first": stats(first), "picked": stats(pick)}
    f, p = res["K"][K]["first"], res["K"][K]["picked"]
    print(f"K={K}: n {p['n']}  genPPL first {f['genppl']:.2f} -> picked {p['genppl']:.2f}  "
          f"entropy {f['entropy']:.3f} -> {p['entropy']:.3f}  tokens {f['mean_tokens']:.0f} -> {p['mean_tokens']:.0f}  "
          f"distinct4 {f['distinct4']:.3f} -> {p['distinct4']:.3f}  seams {f['seam_ratio']:.3f} -> {p['seam_ratio']:.3f}",
          flush=True)
out = a.out or os.path.expanduser(a.samples).replace("_samples.json", "_rerank.json")
json.dump(res, open(out, "w"), indent=2)
print(f"scorer time {score_time:.1f} s for {len(texts)} samples", flush=True)
