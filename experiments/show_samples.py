"""Print saved samples chunk by chunk, for the paper's sample figure (CPU only).

For each run, takes the first N samples of the saved M=4 file that reach the last chunk (no selection by quality)
and writes their per-chunk text to a JSON file.

    python experiments/show_samples.py --runs b0c_M4_G0:0 oursc_M4_G3_halo4:3 b2c_M4_G11:11 --n 3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name:G")
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--temp", default="1.0")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--out", default="~/ncm/runs/paper_samples.json")
a = ap.parse_args()

_, tk, _, _ = load_released(a.ckpt, device="cpu")
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
dec = lambda ids: tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

res = {}
for spec in a.runs:
    name, G = spec.split(":")
    d = json.load(open(os.path.expanduser(f"~/ncm/runs/{name}/eval_M{a.M}_G{G}_t{a.temp}_samples.json")))
    picked = []
    for k, (_, chunks) in enumerate(d["samples"]):
        flat = [i for c in chunks for i in c if i != PAD]
        if EOS in flat[:(a.M - 1) * len(chunks[0])]:
            continue  # story ends before the last chunk
        texts, ended = [], False
        for c in chunks:
            c = [i for i in c if i != PAD]
            if EOS in c:
                c, ended = c[:c.index(EOS)], True
            texts.append(dec(c))
            if ended:
                break
        picked.append({"index": k, "chunks": texts})
        if len(picked) == a.n:
            break
    res[name] = picked
    for p in picked:
        print(f"\n===== {name} sample {p['index']}")
        print(" ||| ".join(p["chunks"]))
json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
