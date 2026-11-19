"""Load the released NCM tokenizer + generator exactly as scripts/eval_shard.py does.

The NCM repository must be importable: set NCM_ROOT (default ~/ncm).
"""
import os
import sys

import torch
from omegaconf import OmegaConf

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
if NCM_ROOT not in sys.path:
    sys.path.insert(0, NCM_ROOT)

from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402


def load_released(ckpt_dir, device="cuda"):
    """Returns (cfg, tokenizer, generator, generator_state_dict)."""
    ckpt_dir = os.path.expanduser(ckpt_dir)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ckpt_dir, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    return cfg, tk, net, sd


def amp_dtype(name="auto"):
    """Autocast dtype: 'bf16', 'fp16', 'fp32' (None = no autocast), or 'auto' = bf16 on GPUs with native bf16
    (compute capability >= 8.0, e.g. RTX 30/40/50, A100, H100), else fp16 (T4, P100, V100: Kaggle/Colab)."""
    if name == "auto":
        if not torch.cuda.is_available():
            return None
        name = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]
