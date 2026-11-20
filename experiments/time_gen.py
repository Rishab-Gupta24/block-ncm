"""Same-GPU latency: Block-NCM (G=5, halo 4, M=4, 512 tokens) vs an AR model generating the same length.

Latency does not depend on the weights, so Block-NCM uses the released generator in the FastChunkedNCM layout (no
checkpoint needed). Reported per batch and per sample, median of --reps timed runs after one warm-up:
  NCM codes   the 11 generator passes (what sample_eval.py reports)
  NCM total   plus decoding codes to tokens (the tokenizer's residual sum and decoder), comparable with AR output
  AR          Hugging Face generate with KV cache, fp16/bf16, exactly --ar-tokens new tokens (end-of-text blocked;
              517 GPT-Neo tokens is about 512 BPE-8192 tokens on TinyStories)

    python experiments/time_gen.py --batches 1,32
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import amp_dtype, load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--G", type=int, default=5)
ap.add_argument("--halo", type=int, default=4)
ap.add_argument("--batches", default="1,32")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--ar-model", default="roneneldan/TinyStories-33M")
ap.add_argument("--ar-tokens", type=int, default=517)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
ap.add_argument("--out", default="~/ncm/runs/time_gen.json")
a = ap.parse_args()
AMP = amp_dtype(a.amp)
bs = [int(x) for x in a.batches.split(",")]


def timed(fn, reps, warmup=3):
    for _ in range(warmup):  # one warm-up was not enough: the first-timed config read slow (clock ramp, allocator)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        fn()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return float(np.median(ts))


res = {"gpu": torch.cuda.get_device_name(0), "amp": str(AMP), "M": a.M, "G": a.G, "halo": a.halo, "rows": []}
_, tk, net, sd = load_released(a.ckpt)
ncm = FastChunkedNCM(net, tk, sd, halo=a.halo).cuda().eval()
C = ncm.chunk_len


def ncm_codes(B):
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP or torch.float32, enabled=AMP is not None):
        return ncm.generate(B, a.M, a.G)


def decode(codes, B, amp=True):
    # The decoder in plain fp32 hits a slow path on this GPU (4.2 s at batch 32; experiments/decode_precision.py).
    # Code-to-latent lookup stays fp32 (running it under bf16 changed 9% of tokens); only the decoder runs under fp16
    # autocast (0.011 s, 99.95% token agreement with fp32). Agreement is checked once below.
    with torch.no_grad():
        _, zl = ncm.vq.idx_to_zhats(codes.reshape(B * a.M, -1), return_zlevels=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return ncm.vqvae.decoder(zl[-1].float()).argmax(-1)


def ncm_total(B):
    return decode(ncm_codes(B), B)


c8 = ncm_codes(32)
agree = (decode(c8, 32) == decode(c8, 32, amp=False)).float().mean().item()
res["decode_amp_token_agreement"] = agree
print(f"decoder token agreement, autocast vs fp32: {agree:.5f}", flush=True)

for B in bs:
    tc, tt = timed(lambda: ncm_codes(B), a.reps), timed(lambda: ncm_total(B), a.reps)
    res["rows"].append({"model": f"Block-NCM G={a.G} halo {a.halo}", "batch": B, "tokens": a.M * C,
                        "codes_s": tc, "total_s": tt, "total_s_per_sample": tt / B})
    print(res["rows"][-1], flush=True)
del ncm, net
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM  # noqa: E402

dt = torch.float16 if AMP in (None, torch.float16) else torch.bfloat16
ar = AutoModelForCausalLM.from_pretrained(a.ar_model, torch_dtype=dt).cuda().eval()
eos = ar.config.eos_token_id


def ar_gen(B):
    x = torch.full((B, 1), eos, dtype=torch.long, device="cuda")
    with torch.no_grad():
        return ar.generate(x, attention_mask=torch.ones_like(x), do_sample=True, top_k=0, top_p=1.0, use_cache=True,
                           max_new_tokens=a.ar_tokens, min_new_tokens=a.ar_tokens, pad_token_id=eos)


for B in bs:
    t = timed(lambda: ar_gen(B), a.reps)
    res["rows"].append({"model": f"AR {a.ar_model.split('/')[-1]} ({dt})", "batch": B, "tokens": a.ar_tokens,
                        "total_s": t, "total_s_per_sample": t / B})
    print(res["rows"][-1], flush=True)

print("\n| model | batch | codes s | s per batch | s per sample |\n|---|---|---|---|---|")
for r in res["rows"]:
    c = f"{r['codes_s']:.3f}" if "codes_s" in r else "—"
    print(f"| {r['model']} | {r['batch']} | {c} | {r['total_s']:.3f} | {r['total_s_per_sample']:.4f} |")
json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
