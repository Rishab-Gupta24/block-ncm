"""Matched per-chunk scoring: is chunk m written worse when the model generates 6 chunks instead of 4?

Every model ends its story inside chunk 4 in most samples (the training stories end there), and what follows the
end-of-text token is not a story, so forcing 6 chunks is not a meaningful test. This compares the SAME positions
instead: chunk m (m = 1..4) of M=4 samples vs chunk m of M=6 samples, each scored by GPT-2 Large conditioned on the
story text before it (score_only_generated). Text is cut at the first end-of-text token, as in sample_eval.py; a
chunk counts only if it starts before that token.

    python experiments/score_chunks.py --runs b0c_M4_G0:0 oursc_M4_G3_halo4:3 --temp 1.0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name:G")
ap.add_argument("--temp", default="1.0")
ap.add_argument("--Ms", default="4,6")
ap.add_argument("--chunks", type=int, default=4, help="score chunks 1..this")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--out", default="~/ncm/runs/chunk_scores_t{temp}.json")
a = ap.parse_args()

_, tk, _, _ = load_released(a.ckpt)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
sys.path.insert(0, NCM_ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).cuda().eval()


def score(prompts, gens):
    keep = [(p, g) for p, g in zip(prompts, gens) if g.strip()]
    if not keep:
        return float("nan")
    p, g = zip(*keep)
    return float(generation_perplexity(rlm, rtok, list(p), list(g), batch_size=a.score_bs).mean_ppl)


res = {}
for spec in a.runs:
    name, G = spec.split(":")
    for M in [int(x) for x in a.Ms.split(",")]:
        f = os.path.expanduser(f"~/ncm/runs/{name}/eval_M{M}_G{G}_t{a.temp}_samples.json")
        d = json.load(open(f))
        row = {}
        for m in range(a.chunks):
            prompts, gens = [], []
            for _, chunks in d["samples"]:
                pre = [i for c in chunks[:m] for i in c if i != PAD]
                cur = [i for i in chunks[m] if i != PAD]
                if EOS in pre:
                    continue  # story already ended before this chunk
                if EOS in cur:
                    cur = cur[:cur.index(EOS)]
                ptxt, full = dec(pre), dec(pre + cur)
                prompts.append(ptxt)
                gens.append(full[len(ptxt):] if full.startswith(ptxt) else dec(cur))
            row[f"chunk{m + 1}"] = {"n": len(gens), "genppl": score(prompts, gens)}
        res[f"{name}_M{M}"] = row
        print(f"{name:20s} M={M}  " + "  ".join(f"c{k[5:]} {v['genppl']:6.1f} (n {v['n']})" for k, v in row.items()),
              flush=True)
json.dump(res, open(os.path.expanduser(a.out.format(temp=a.temp)), "w"), indent=2)
