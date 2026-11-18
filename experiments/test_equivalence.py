"""Sanity checks for ChunkedNCM against the released generator.

1. With M=1 and G=0 (independent) or G=K (global), ChunkedNCM must reproduce the released model's
   teacher-forced loss exactly (same mask, positions and embeddings; chunk embedding is zero).
2. Generation runs end to end for several (M, G) and decodes to text.
3. Tokenizer round trip on real chunks (how much text survives encode -> decode).

    python experiments/test_equivalence.py --ckpt ~/ncm/checkpoints/tinystories --data ~/ncm/data/tinystories
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402
from blockncm.model import ChunkedNCM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
a = ap.parse_args()

torch.manual_seed(0)
cfg, tk, net, sd = load_released(a.ckpt)
net.eval()
model = ChunkedNCM(net, tk, sd).cuda().eval()
C = tk.vqvae.context_length
EOS, PAD = model.eos, model.pad

data = os.path.expanduser(a.data)
off = np.load(os.path.join(data, "train_bpe_8192_offsets.npy")).astype(np.int64)
tok = np.memmap(os.path.join(data, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
docs = [tok[off[i]:off[i + 1]].tolist()[:C] for i in range(8)]
x = torch.tensor([d + [PAD] * (C - len(d)) for d in docs], device="cuda")

with torch.no_grad():
    codes = tk.vqvae.encode(x)  # (8, L)
    X, Y = tk.vqvae.get_ncm_inputs(x)
    z_prefix = tk.vqvae.encoder.wte(torch.full((8, net.prefix_len), EOS, device="cuda"))
    logits = net(torch.cat([z_prefix, X], 1)).float()
    ref = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).item()
    for G in (0, model.K):
        got, _ = model.forward_loss(codes.view(8, 1, -1).long(), G)
        print(f"[1] M=1 G={G:2d}: chunked={got.item():.6f} released={ref:.6f} diff={abs(got.item() - ref):.2e}")
        assert abs(got.item() - ref) < 1e-3, "ChunkedNCM does not reproduce the released model"

    # round trip
    rt = model.decode_tokens(codes.view(8, 1, -1))
    match = np.mean([np.mean([a_ == b_ for a_, b_ in zip(r, d)]) for r, d in zip(rt, docs)])
    print(f"[3] tokenizer round-trip token accuracy on 8 chunks: {match:.3f}")

    for M, G in ((1, 0), (2, 0), (2, 3), (2, model.K), (4, 3)):
        out = model.generate(2, M, G, temperature=1.0)
        texts = model.decode_texts(out)
        print(f"[2] M={M} G={G:2d} codes={tuple(out.shape)}  sample: {texts[0][:160]!r}")
print("all checks passed")
