"""Can the frozen multiscale VQ tokenizer survive skipped levels? (No generator, no training.)

Cascade-NCM proposes generating only some fine levels (e.g. 6 -> 9 -> 11, skipping 7, 8, 10). The tokenizer is
residual: each level quantizes what the earlier levels left over, so skipping levels changes what the later
codebooks see. This probe gives the *upper bound* for any generator: real text, perfect codes for the kept levels.

Two variants per skip set S (level indices into vq.levels, index 0 = the l=1 level, so "level 7" = l=64 as in
RESULTS.md):
  requant  re-run the residual quantizer with the levels in S contributing nothing (later levels absorb the residual).
           This is the best case: a generator that learned the skipped tokenizer's codes.
  hole     encode normally, then drop the contributions of S at decode time (kept codes are the normal ones).
Reports token accuracy and the share of chunks reconstructed exactly, on real (non-padding) 128-token chunks.

    python experiments/level_skip_probe.py --n 256
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
ap.add_argument("--n", type=int, default=256, help="validation documents (4 real chunks each)")
ap.add_argument("--data", default="~/ncm/data/tinystories")
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="~/ncm/runs/level_skip_probe.json")
a = ap.parse_args()

C, M = 128, 4
data = os.path.expanduser(a.data)
off = np.load(os.path.join(data, "train_bpe_8192_offsets.npy")).astype(np.int64)
tok = np.memmap(os.path.join(data, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
eligible = np.nonzero(np.diff(off) >= M * C)[0]  # every chunk fully real text: no EOS/PAD inside
vidx = eligible[eligible % a.val_every == 0][: a.n]
chunks = np.stack([tok[off[i]:off[i] + M * C].astype(np.int64).reshape(M, C) for i in vidx]).reshape(-1, C)

_, tk, net, _ = load_released(a.ckpt, device=a.device)
del net
if a.device == "cuda":
    torch.cuda.empty_cache()
vae, vq = tk.vqvae, tk.vqvae.vq
levels = list(vq.levels)
K = len(levels)
print(f"{len(chunks)} chunks from {len(vidx)} docs; levels {levels}")


@torch.no_grad()
def reconstruct(z, skip, mode):
    B, L, D = z.shape
    z_rest, z_hat = z.clone(), torch.zeros_like(z)
    for i, lv in enumerate(levels):
        flat = (vq._down_for_quant(z_rest, i, lv) if i != K - 1 else z_rest).reshape(-1, D)
        idx = torch.argmin(vq._distances(flat, vq.embeddings[i].weight), dim=1).view(B, lv)
        q = vq.embeddings[i](idx)
        q = vq._up_to_full(q, i, L) if i != K - 1 else q
        q = vq._apply_quant_resi(q, i)
        if i in skip:
            if mode == "requant":
                continue  # nothing added, residual unchanged: the next level absorbs it
            z_rest = z_rest - q  # hole: encoder proceeds normally, decoder never sees level i
            continue
        z_hat = z_hat + q
        z_rest = z_rest - q
    return vae.decoder(z_hat).argmax(-1)


SKIPS = {
    "none": [],
    "skip 11 (l=128)": [11],
    "skip 10 (l=108)": [10],
    "skip 9 (l=92)": [9],
    "skip 8 (l=76)": [8],
    "skip 7 (l=64)": [7],
    "fallback: skip 7, 9 (6->8->10->11)": [7, 9],
    "cascade: skip 7, 8, 10 (6->9->11)": [7, 8, 10],
    "skip 7-10 (6->11)": [7, 8, 9, 10],
    "skip 5, 7, 9 (alternate mid/fine)": [5, 7, 9],
    "skip 4-10 (3->11)": [4, 5, 6, 7, 8, 9, 10],
}
res = {"n_chunks": int(len(chunks)), "levels": levels, "variants": {}}
print(f"{'variant':38s} {'positions':>9s} | {'requant acc':>11s} {'exact':>6s} | {'hole acc':>8s} {'exact':>6s}")
for name, skip in SKIPS.items():
    pos = sum(lv for i, lv in enumerate(levels) if i not in skip)
    row = {"skip": skip, "positions": pos}
    for mode in ("requant", "hole"):
        correct = exact = 0
        for s in range(0, len(chunks), a.bs):
            x = torch.from_numpy(chunks[s:s + a.bs]).to(a.device)
            z = vae.encode_z(x).float()
            y = reconstruct(z, set(skip), mode)
            correct += (y == x).sum().item()
            exact += (y == x).all(-1).sum().item()
        row[f"{mode}_token_acc"] = correct / chunks.size
        row[f"{mode}_exact_chunks"] = exact / len(chunks)
    res["variants"][name] = row
    print(f"{name:38s} {pos:9d} | {row['requant_token_acc']:11.4f} {row['requant_exact_chunks']:6.3f} | "
          f"{row['hole_token_acc']:8.4f} {row['hole_exact_chunks']:6.3f}", flush=True)

out = os.path.expanduser(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
print("->", out)
