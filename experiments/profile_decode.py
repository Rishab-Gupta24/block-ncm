"""Split Block-NCM generation time into generator passes, code-to-latent lookup and the tokenizer decoder.

    python experiments/profile_decode.py --B 32
Run on an idle GPU for absolute numbers; under contention only the split between stages is meaningful.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import amp_dtype, load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--G", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ckpt", default="~/ncm/checkpoints/tinystories")
a = ap.parse_args()
AMP = amp_dtype("auto")
_, tk, net, sd = load_released(a.ckpt)
ncm = FastChunkedNCM(net, tk, sd, halo=4).cuda().eval()
print("vqvae training mode:", ncm.vqvae.training, " decoder:", type(ncm.vqvae.decoder).__name__)


def t(fn, reps=a.reps):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); out = fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    return min(ts), out


with torch.no_grad():
    tg, codes = t(lambda: ncm.generate(a.B, a.M, a.G) if AMP is None else
                  torch.autocast("cuda", dtype=AMP)(ncm.generate)(a.B, a.M, a.G))
    flat = codes.reshape(a.B * a.M, -1)
    tz, (_, zl) = t(lambda: ncm.vq.idx_to_zhats(flat, return_zlevels=True))
    z = zl[-1].float()
    print("latent", tuple(z.shape))
    td32, _ = t(lambda: ncm.vqvae.decoder(z).argmax(-1))
    with torch.autocast("cuda", dtype=AMP or torch.bfloat16):
        td16, _ = t(lambda: ncm.vqvae.decoder(z).argmax(-1))
print(f"B={a.B}: generator {tg:.3f} s | idx_to_zhats {tz:.3f} s | decoder fp32 {td32:.3f} s | decoder amp {td16:.3f} s")
