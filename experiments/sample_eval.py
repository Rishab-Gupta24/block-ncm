"""Generate long-form samples with a chunked model and score them.

Metrics (scored with the NCM repo's own generation_perplexity and GPT-2 Large, like evaluate.py):
  genPPL     whole sample
  entropy    unigram entropy of the BPE-8192 tokens (repetition check, as in eval_merge.py)
  E2         genPPL of windows straddling a chunk boundary (last W tokens of chunk m + first W of
             chunk m+1) vs. windows of 2W tokens inside a chunk. A large boundary/inner ratio means
             the chunks do not join up.
  latency    wall-clock seconds per generated batch (generator only)

    python experiments/sample_eval.py --run ~/ncm/runs/ours_M4_G3 --M 4 --G 3
    python experiments/sample_eval.py --M 2 --G 0          # zero-shot released weights
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
from blockncm.model import ChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="", help="run dir with model.pt (empty = released weights, zero-shot)")
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--G", type=int, required=True)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--n", type=int, default=256, help="samples per seed")
ap.add_argument("--seeds", default="42,0")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--score-bs", type=int, default=4, help="GPT-2 Large scoring batch (8 GB GPU: 4)")
ap.add_argument("--fast", action="store_true", help="FastChunkedNCM generation (KV-cached plan; same codes)")
ap.add_argument("--halo", type=int, default=None, help="halo width (default: the run's training value)")
ap.add_argument("--n-plans", type=int, default=1, help="plan search: candidates per document (needs 0<G<K)")
ap.add_argument("--out", default="")
a = ap.parse_args()

cfg, tk, net, sd = load_released(a.ckpt)
st = torch.load(os.path.join(os.path.expanduser(a.run), "model.pt"), map_location="cpu", weights_only=False) \
    if a.run else None
halo = a.halo if a.halo is not None else (st["args"].get("halo", 0) if st else 0)
if a.fast or halo or a.n_plans > 1:
    from blockncm.fast import FastChunkedNCM
    model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
else:
    model = ChunkedNCM(net, tk, sd).cuda().eval()
if st:
    model.load_state_dict(st["model"], strict=True)
gen_kw = {"n_plans": a.n_plans} if a.n_plans > 1 else {}
C = model.chunk_len
EOS, PAD = model.eos, model.pad

seeds = [int(s) for s in a.seeds.split(",")]
samples, lat = [], []  # samples: (seed, per-chunk token lists)
for seed in seeds:
    torch.manual_seed(seed)
    for s in range(0, a.n, a.bs):
        B = min(a.bs, a.n - s)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            codes = model.generate(B, a.M, a.G, temperature=a.temp, **gen_kw)
        torch.cuda.synchronize(); lat.append((time.time() - t0, B))
        flat = codes.reshape(B * a.M, -1)
        _, zl = model.vq.idx_to_zhats(flat, return_zlevels=True)
        ids = model.vqvae.decoder(zl[-1].float()).argmax(-1).view(B, a.M, C).tolist()
        samples += [(seed, row) for row in ids]
    print(f"seed {seed}: {a.n} samples", flush=True)

del model, net
torch.cuda.empty_cache()


def clean(ids):
    return [i for i in ids if i not in (EOS, PAD)]


def full_tokens(chunks):
    flat = [i for c in chunks for i in c]
    if EOS in flat:
        flat = flat[:flat.index(EOS)]
    return [i for i in flat if i != PAD]


dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)
texts = [dec(full_tokens(ch)) for _, ch in samples]
toks = [full_tokens(ch) for _, ch in samples]

# boundary vs inner windows (only where the story has not ended before the boundary)
W = a.window
bwin, iwin = [], []
for _, ch in samples:
    # seams are measured on each chunk's real tokens (PAD removed), so padded, sentence-aligned chunks work too;
    # for fixed 128-token chunks this is identical to cutting at multiples of C
    flat, starts = [], []
    for c in ch:
        starts.append(len(flat))
        flat += [i for i in c if i != PAD]
    starts.append(len(flat))
    end = flat.index(EOS) if EOS in flat else len(flat)
    for m in range(a.M - 1):
        b = starts[m + 1]
        if b - W >= 0 and b + W <= end:
            bwin.append(dec(clean(flat[b - W:b + W])))
    for m in range(a.M):
        mid = (starts[m] + starts[m + 1]) // 2
        if mid - W >= 0 and mid + W <= end:
            iwin.append(dec(clean(flat[mid - W:mid + W])))

out = a.out or os.path.join(os.path.expanduser(a.run) if a.run else os.path.expanduser("~/ncm/runs/zeroshot"),
                            f"eval_M{a.M}_G{a.G}_t{a.temp}"
                            + (f"_plans{a.n_plans}" if a.n_plans > 1 else "")
                            + (f"_halo{a.halo}" if a.halo is not None else "") + ".json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"seeds": seeds, "samples": samples, "latency": lat}, open(out.replace(".json", "_samples.json"), "w"))

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rtok = AutoTokenizer.from_pretrained(a.ref_model)
rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).cuda().eval()


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
    "run": a.run or "released(zero-shot)", "M": a.M, "G": a.G, "temp": a.temp,
    "genppl": float(g.mean()), "genppl_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
    "entropy": float(np.mean([p["entropy"] for p in per_seed])),
    "per_seed": per_seed,
    "boundary_genppl": gppl(bwin), "inner_genppl": gppl(iwin),
    "n_boundary": len(bwin), "n_inner": len(iwin),
    "mean_tokens": float(np.mean([len(t) for t in toks])),
    "frac_reaching_last_chunk": float(np.mean([len(t) > (a.M - 1) * C for t in toks])),
    "latency_s_per_sample": float(sum(t for t, _ in lat[1:]) / max(1, sum(b for _, b in lat[1:]))),
    "examples": texts[:8],
}
res["boundary_ratio"] = res["boundary_genppl"] / res["inner_genppl"]
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
json.dump(res, open(out, "w"), indent=2)
print("->", out)
