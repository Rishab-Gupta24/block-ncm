"""Score the first N BPE-8192 tokens of saved NCM samples with GPT-2 Large (same metric as sample_eval.py).

Length-matched comparison with the natural-length AR reference (TinyStories-33M stories end at 217 tokens on
average) without blocking AR's end-of-text: truncate the NCM samples instead. Samples shorter than N are kept whole.

    python experiments/score_prefix.py --tokens 217 --device cpu \
        --samples ~/ncm/runs/b0c_M4_G0/eval_M4_G0_t1.0_samples.json ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True, help="*_samples.json files written by sample_eval.py")
ap.add_argument("--tokens", type=int, default=217)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="~/ncm/runs/prefix_scores.json")
a = ap.parse_args()
torch.set_num_threads(os.cpu_count() or 1)

_, tk, net, _ = load_released(a.ckpt, device="cpu")  # only the BPE-8192 pre-tokenizer is needed
del net
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
sys.path.insert(0, NCM_ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).to(a.device).eval()

res = {}
for path in a.samples:
    d = json.load(open(os.path.expanduser(path)))
    by_seed, lens = {}, []
    for seed, chunks in d["samples"]:
        flat = [i for c in chunks for i in c]
        if EOS in flat:
            flat = flat[:flat.index(EOS)]
        ids = [i for i in flat if i != PAD][: a.tokens]
        lens.append(len(ids))
        by_seed.setdefault(seed, []).append(dec(ids))
    g = []
    for seed, ts in by_seed.items():
        ts = [t for t in ts if t.strip()]
        g.append(float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs,
                                             device=a.device).mean_ppl))
    res[path] = {"tokens": a.tokens, "genppl": float(np.mean(g)), "per_seed": g, "mean_tokens": float(np.mean(lens))}
    print(f"{path}: genPPL {np.mean(g):.2f} (seeds {[round(x, 2) for x in g]}), mean tokens {np.mean(lens):.1f}",
          flush=True)
out = os.path.expanduser(a.out)
json.dump(res, open(out, "w"), indent=2)
print("->", out)
