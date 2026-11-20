"""Length-controlled M=6 scoring from saved samples (no regeneration).

sample_eval.py cuts each sample at its first end-of-text token, so a model that ends its stories on time is scored on
shorter text than one that keeps writing. This scores, for each *_samples.json, only the samples whose text runs past
--min-tokens BPE tokens: (a) the full text and (b) the tail beyond --tail-from tokens (the extrapolated part), with
GPT-2 Large, per seed. It also reports the fraction of samples that qualify.

    python experiments/score_tail.py ~/ncm/runs/b2c_M4_G11/eval_M6_G11_t1.0_samples.json ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--min-tokens", type=int, default=576, help="keep samples longer than this (4.5 chunks)")
ap.add_argument("--tail-from", type=int, default=512, help="tail = tokens after this position (beyond M=4)")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--out", default="~/ncm/runs/tail_scores.json")
a = ap.parse_args()

_, tk, _, _ = load_released(a.ckpt)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id  # as in blockncm/model.py
specials = {EOS, PAD}
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
sys.path.insert(0, NCM_ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).cuda().eval()


def gppl(ts):
    ts = [t for t in ts if t.strip()]
    return float(generation_perplexity(rlm, rtok, [""] * len(ts), ts, batch_size=a.score_bs).mean_ppl) if ts else float("nan")


res = []
for f in a.files:
    d = json.load(open(os.path.expanduser(f)))
    by_seed = {}
    for seed, chunks in d["samples"]:
        flat = [i for c in chunks for i in c]
        if EOS in flat:
            flat = flat[:flat.index(EOS)]
        flat = [i for i in flat if i not in specials]
        by_seed.setdefault(seed, []).append(flat)
    row = {"file": f, "per_seed": []}
    for seed, toks in by_seed.items():
        keep = [t for t in toks if len(t) > a.min_tokens]
        row["per_seed"].append({"seed": seed, "n": len(toks), "n_long": len(keep),
                                "full": gppl([dec(t) for t in keep]),
                                "tail": gppl([dec(t[a.tail_from:]) for t in keep])})
    ps = row["per_seed"]
    row.update({k: float(np.nanmean([p[k] for p in ps])) for k in ("full", "tail")})
    row["frac_long"] = sum(p["n_long"] for p in ps) / sum(p["n"] for p in ps)
    res.append(row)
    print(f"{f.split('runs/')[-1]:50s} long {row['frac_long']:.3f}  full {row['full']:6.1f}  tail {row['tail']:6.1f}",
          flush=True)
json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
