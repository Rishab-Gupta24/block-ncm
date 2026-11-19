"""Reference scores for *real* TinyStories text, with the same metrics as sample_eval.py.

Takes the first M*128 BPE tokens of validation documents (prepare_chunks.py's selection), cuts at EOS,
and reports genPPL under GPT-2 Large, unigram entropy, and the boundary/inner window genPPL.

    python experiments/real_text_ppl.py --M 4 --n 512
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--data", default="~/ncm/data/tinystories")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--out", default="~/ncm/runs/real_text")
a = ap.parse_args()

C, M, W = 128, a.M, a.window
data = os.path.expanduser(a.data)
off = np.load(os.path.join(data, "train_bpe_8192_offsets.npy")).astype(np.int64)
tok = np.memmap(os.path.join(data, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
eligible = np.nonzero(np.diff(off) > (M - 1) * C)[0]
vidx = eligible[eligible % a.val_every == 0][: a.n]

_, tk, net, _ = load_released(a.ckpt, device="cpu")  # only the BPE pre-tokenizer is needed
del net
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

texts, toks, bwin, iwin = [], [], [], []
for i in vidx:
    flat = tok[off[i]:off[i + 1]].tolist()[: M * C]
    end = flat.index(EOS) if EOS in flat else len(flat)
    ids = [t for t in flat[:end] if t != PAD]
    texts.append(dec(ids)); toks.append(ids)
    for m in range(M - 1):
        b = (m + 1) * C
        if b + W <= end:
            bwin.append(dec(flat[b - W:b + W]))
    for m in range(M):
        mid = m * C + C // 2
        if mid + W <= end:
            iwin.append(dec(flat[mid - W:mid + W]))

from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).cuda().eval()


def gppl(ts):
    ts = [t for t in ts if t.strip()]
    return float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs).mean_ppl)


def entropy(ids):
    c, n = Counter(ids), len(ids)
    return -sum(v / n * math.log(v / n) for v in c.values()) if n else 0.0


res = {"M": M, "n_docs": len(texts), "genppl": gppl(texts),
       "entropy": float(np.mean([entropy(t) for t in toks if t])),
       "boundary_genppl": gppl(bwin), "inner_genppl": gppl(iwin),
       "n_boundary": len(bwin), "n_inner": len(iwin),
       "mean_tokens": float(np.mean([len(t) for t in toks]))}
res["boundary_ratio"] = res["boundary_genppl"] / res["inner_genppl"]
print(json.dumps(res, indent=2))
out = os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, f"real_M{M}.json"), "w"), indent=2)
