"""Encode TinyStories documents into M chunks of NCM codes with the frozen released tokenizer.

Keeps documents with (M-1)*128 < length; takes the first M*128 BPE tokens, appends EOS when the story
ends inside the window, pads with PAD (as VQVAETokenizer._adjust_length does), and encodes each
128-token chunk independently. Every 100th document is held out for validation.

Output: <out>/train_M{M}.npy and val_M{M}.npy, uint16 arrays of shape (n_docs, M, 1 + sum(levels[1:])).

    python experiments/prepare_chunks.py --M 4 --data ~/ncm/data/tinystories --ckpt ~/ncm/checkpoints/tinystories --out ~/ncm/data/chunked
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-docs", type=int, default=0, help="cap on documents (0 = all eligible)")
ap.add_argument("--bs", type=int, default=512, help="chunks per encoder batch")
ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--inspect", action="store_true", help="print format diagnostics and exit")
ap.add_argument("--lengths-only", action="store_true",
                help="only write {train,val}_M{M}_len.npy (document lengths) for an existing encoding")
a = ap.parse_args()

data = os.path.expanduser(a.data)
off = np.load(os.path.join(data, "train_bpe_8192_offsets.npy")).astype(np.int64)
tok = np.memmap(os.path.join(data, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
assert off[-1] == len(tok), f"offsets end {off[-1]} != token count {len(tok)} (dtype assumption wrong?)"

lengths = np.diff(off)
if a.lengths_only:  # no GPU needed: TinyStories tokenizer context is 128
    C = 128
else:
    cfg, tk, net, _ = load_released(a.ckpt)
    del net
    torch.cuda.empty_cache()
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    C = tk.vqvae.context_length

if a.inspect:
    for i in range(3):
        ids = tok[off[i]:off[i + 1]].tolist()
        print(f"doc {i}: len={len(ids)} first={ids[:5]} last={ids[-5:]} eos_inside={EOS in ids[:-1]} ends_eos={ids[-1] == EOS}")
        print("   ", repr(tk.pre_tokenizer.decode(ids, skip_special_tokens=False)[:300]))
    print(f"EOS={EOS} PAD={PAD} context={C}")
    sys.exit(0)

M = a.M
eligible = np.nonzero(lengths > (M - 1) * C)[0]
if a.max_docs:
    eligible = eligible[: a.max_docs]
print(f"M={M}: {len(eligible):,} eligible documents")
is_val = (eligible % a.val_every) == 0
out = os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
# raw document lengths (tokens, before the window cut), same row order as the code arrays
np.save(os.path.join(out, f"train_M{M}_len.npy"), lengths[eligible[~is_val]].astype(np.int32))
np.save(os.path.join(out, f"val_M{M}_len.npy"), lengths[eligible[is_val]].astype(np.int32))
if a.lengths_only:
    print(f"wrote lengths: train {int((~is_val).sum()):,} / val {int(is_val.sum()):,}")
    sys.exit(0)


def window(i):
    ids = tok[off[i]:off[i + 1]].tolist()
    if len(ids) < M * C and (not ids or ids[-1] != EOS):
        ids = ids + [EOS]
    ids = ids[: M * C]
    return ids + [PAD] * (M * C - len(ids))


L = sum(tk.vqvae.vq.levels)
codes = np.zeros((len(eligible), M, L), dtype=np.uint16)
docs_per_batch = max(1, a.bs // M)
t0 = time.time()
for s in range(0, len(eligible), docs_per_batch):
    idx = eligible[s:s + docs_per_batch]
    x = torch.tensor([window(i) for i in idx], dtype=torch.long).view(-1, C).cuda()
    c = tk.vqvae.encode(x).view(len(idx), M, L)
    assert int(c.max()) < 65536
    codes[s:s + len(idx)] = c.cpu().numpy().astype(np.uint16)
    if (s // docs_per_batch) % 50 == 0:
        print(f"  {s + len(idx):,}/{len(eligible):,}  {time.time() - t0:.0f}s", flush=True)

np.save(os.path.join(out, f"train_M{M}.npy"), codes[~is_val])
np.save(os.path.join(out, f"val_M{M}.npy"), codes[is_val])
print(f"saved train {int((~is_val).sum()):,} / val {int(is_val.sum()):,} docs -> {out}  ({time.time() - t0:.0f}s)")
