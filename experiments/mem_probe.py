"""Why is Block-NCM slow at batch 32 on an 8 GB GPU? Split generate vs decode time and log peak GPU memory.

    python experiments/mem_probe.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockncm.fast import FastChunkedNCM
from blockncm.load import amp_dtype, load_released

AMP = amp_dtype("auto")
_, tk, net, sd = load_released("~/ncm/checkpoints/tinystories")
ncm = FastChunkedNCM(net, tk, sd, halo=4).cuda().eval()
free, total = torch.cuda.mem_get_info()
print("gpu", torch.cuda.get_device_name(0), "free/total GiB", free / 2**30, total / 2**30, "m_max", ncm.m_max)


def run(B, M):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        codes = ncm.generate(B, M, 5)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        _, zl = ncm.vq.idx_to_zhats(codes.reshape(B * M, -1), return_zlevels=True)
        with torch.autocast("cuda", dtype=torch.float16):
            ncm.vqvae.decoder(zl[-1].float()).argmax(-1)
    torch.cuda.synchronize(); t2 = time.time()
    return t1 - t0, t2 - t1, torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


for B, M in [(32, 4), (32, 6), (32, 8), (16, 8), (8, 8)]:
    for i in range(3):
        g, d, pa, pr = run(B, M)
    print(f"B={B} M={M} generate {g:.3f}s decode {d:.3f}s peak alloc {pa:.2f} GiB reserved {pr:.2f} GiB", flush=True)
