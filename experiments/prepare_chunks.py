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
from blockncm.load import load_released, token_docs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-docs", type=int, default=0, help="cap on documents (0 = all eligible)")
ap.add_argument("--bs", type=int, default=512, help="chunks per encoder batch")
ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--source", choices=["train", "valid"], default="train",
                help="valid: encode every eligible story of valid_bpe_8192.bin (never trained on) into heldout_M{M}.npy")
ap.add_argument("--inspect", action="store_true", help="print format diagnostics and exit")
ap.add_argument("--lengths-only", action="store_true",
                help="only write {train,val}_M{M}_len.npy (document lengths) for an existing encoding")
ap.add_argument("--align", choices=["fixed", "sentence"], default="fixed",
                help="fixed: cut every 128 tokens. sentence: pack whole sentences into each chunk and pad the rest "
                     "(files get the suffix _sent)")
ap.add_argument("--dry-run", type=int, default=0, help="print the first N documents' chunks and exit (no encoding)")
a = ap.parse_args()
SUF = "" if a.align == "fixed" else "_sent"

data = os.path.expanduser(a.data)
off, tok = token_docs(data, a.source)
if a.source == "train":
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
VAL = "val"
if a.source == "valid":  # every story is held out: one file, heldout_M{M}
    is_val[:] = True
    VAL = "heldout"
out = os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
# raw document lengths (tokens, before the window cut), same row order as the code arrays
if a.source == "train":
    np.save(os.path.join(out, f"train_M{M}{SUF}_len.npy"), lengths[eligible[~is_val]].astype(np.int32))
np.save(os.path.join(out, f"{VAL}_M{M}{SUF}_len.npy"), lengths[eligible[is_val]].astype(np.int32))
if a.lengths_only:
    print(f"wrote lengths: train {int((~is_val).sum()):,} / val {int(is_val.sum()):,}")
    sys.exit(0)


def window(i):
    ids = tok[off[i]:off[i + 1]].tolist()
    if len(ids) < M * C and (not ids or ids[-1] != EOS):
        ids = ids + [EOS]
    ids = ids[: M * C]
    return ids + [PAD] * (M * C - len(ids))


# sentence-ending BPE tokens: decoded text ends in . ! ? (possibly followed by quotes) or contains a newline
_ends = set()
for _i in range(len(tk.pre_tokenizer.get_vocab()) if hasattr(tk.pre_tokenizer, "get_vocab") else 8192):
    try:
        _s = tk.pre_tokenizer.decode([_i], skip_special_tokens=True)
    except Exception:
        continue
    if "\n" in _s or _s.rstrip(" \"'").endswith((".", "!", "?")):
        _ends.add(_i)
fill = []


def window_sentence(i):
    """Pack whole sentences into chunks of <= C tokens (pad each chunk); a sentence longer than C is cut."""
    ids = tok[off[i]:off[i + 1]].tolist()
    if not ids or ids[-1] != EOS:
        ids = ids + [EOS]
    sents, cur = [], []
    for t_ in ids:
        cur.append(t_)
        if t_ in _ends or t_ == EOS:
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    chunks, cur = [], []
    for s_ in sents:
        while len(s_) > C:                       # over-long sentence: hard cut
            if cur:
                chunks.append(cur); cur = []
            chunks.append(s_[:C]); s_ = s_[C:]
        if len(cur) + len(s_) > C:
            chunks.append(cur); cur = []
        cur = cur + s_
        if len(chunks) >= M:
            break
    if cur and len(chunks) < M:
        chunks.append(cur)
    chunks = (chunks + [[]] * M)[:M]
    fill.append(sum(len(c) for c in chunks) / (M * C))
    return [t_ for c in chunks for t_ in c + [PAD] * (C - len(c))]


make_window = window if a.align == "fixed" else window_sentence
print(f"alignment: {a.align}" + (f" ({len(_ends)} sentence-ending tokens)" if a.align == "sentence" else ""))
if a.dry_run:
    for i in eligible[: a.dry_run]:
        w_ = make_window(i)
        print(f"\n=== doc {i} ({lengths[i]} tokens)")
        for m in range(M):
            c = [t_ for t_ in w_[m * C:(m + 1) * C] if t_ != PAD]
            print(f"--- chunk {m + 1}: {len(c)} tokens | ...{tk.pre_tokenizer.decode(c[-12:], skip_special_tokens=True)!r}")
    big = [make_window(i) for i in eligible[: 2000]]
    print(f"\nmean share of real tokens over {len(big)} docs: {np.mean(fill[-len(big):]):.3f}" if fill else "")
    sys.exit(0)
L = sum(tk.vqvae.vq.levels)
codes = np.zeros((len(eligible), M, L), dtype=np.uint16)
docs_per_batch = max(1, a.bs // M)
t0 = time.time()
for s in range(0, len(eligible), docs_per_batch):
    idx = eligible[s:s + docs_per_batch]
    x = torch.tensor([make_window(i) for i in idx], dtype=torch.long).view(-1, C).cuda()
    c = tk.vqvae.encode(x).view(len(idx), M, L)
    assert int(c.max()) < 65536
    codes[s:s + len(idx)] = c.cpu().numpy().astype(np.uint16)
    if (s // docs_per_batch) % 50 == 0:
        print(f"  {s + len(idx):,}/{len(eligible):,}  {time.time() - t0:.0f}s", flush=True)

if a.source == "train":
    np.save(os.path.join(out, f"train_M{M}{SUF}.npy"), codes[~is_val])
np.save(os.path.join(out, f"{VAL}_M{M}{SUF}.npy"), codes[is_val])
if fill:
    print(f"sentence packing: mean share of real tokens in the M*C window {np.mean(fill):.3f}")
print(f"saved train {int((~is_val).sum()):,} / val {int(is_val.sum()):,} docs -> {out}  ({time.time() - t0:.0f}s)")
