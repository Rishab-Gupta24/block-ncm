"""Tokenizer-decoder precision: speed and token agreement with the plain fp32 decode (the one sample_eval.py scores).

    python experiments/decode_precision.py --B 32
Variants: fp32 (TF32 off), fp32 with TF32 matmuls, fp16 autocast, bf16 autocast.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--G", type=int, default=5)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
a = ap.parse_args()
_, tk, net, sd = load_released(a.ckpt)
ncm = FastChunkedNCM(net, tk, sd, halo=4).cuda().eval()
print("tf32 default: matmul", torch.backends.cuda.matmul.allow_tf32, "cudnn", torch.backends.cudnn.allow_tf32,
      "| sdpa flash", torch.backends.cuda.flash_sdp_enabled(), "mem-eff", torch.backends.cuda.mem_efficient_sdp_enabled())

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    codes = ncm.generate(a.B, a.M, a.G)
with torch.no_grad():
    _, zl = ncm.vq.idx_to_zhats(codes.reshape(a.B * a.M, -1), return_zlevels=True)
z = zl[-1].float()


def run(tf32, amp):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    with torch.no_grad(), torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
        return ncm.vqvae.decoder(z).argmax(-1)


def timed(*args):
    run(*args); torch.cuda.synchronize()
    ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize(); t0 = time.time(); out = run(*args); torch.cuda.synchronize(); ts.append(time.time() - t0)
    return sorted(ts)[len(ts) // 2], out


t_ref, ref = timed(False, None)
print(f"{'fp32':16s} {t_ref:8.4f} s  agreement 1.00000")
for name, args in [("fp32 + tf32", (True, None)), ("fp16 autocast", (False, torch.float16)),
                   ("bf16 autocast", (False, torch.bfloat16))]:
    t, out = timed(*args)
    print(f"{name:16s} {t:8.4f} s  agreement {(out == ref).float().mean().item():.5f}")

# where does fp32 time go? (decoder pieces)
torch.backends.cuda.matmul.allow_tf32 = False
dec = ncm.vqvae.decoder
with torch.no_grad():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        dec(z).argmax(-1); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
