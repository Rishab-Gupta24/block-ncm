"""PlanAdapterNCM: plan-only adapters on top of FastChunkedNCM (docs/plan_verify_refine.md, Sec. 2).

* The released generator weights are frozen.
* **Global passes** (prefix, levels 1..G) run with LoRA adapters on every block (attention qkv/out, MLP in/out).
* **Local passes** (levels G+1..K) run the frozen released weights. They see the plan only through its
  cached keys/values, which the adapted global pass produces; local-level loss trains the adapters through
  those cached K/V (they are never detached).
* **Plan logit bias:** a learnable scalar per layer and head is added to local queries' attention logits on
  the plan keys, initialised negative (default -8) so local levels start by ignoring the plan (at step 0 the
  plan raises local NLL from 2.471 to 2.875 with full weights) and open it as it becomes useful. Implemented as
  one extra key dimension (q' = [q, 1], k' = [k, b*sqrt(hd)] on plan keys, scale 1/sqrt(hd)): gradients through
  a float attention mask fail on the GPU's fused kernel ("LSE is not correctly aligned"); this form is exact
  (checked against manual attention: relative gradient error 5e-7 fp32, 2e-3 bf16).

With no plan in context the local levels compute exactly the released model, so fine-tuning cannot cost
single-chunk fluency on the 592 of 640 positions per chunk that are local (G=3).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from blockncm.fast import FastChunkedNCM
from models.common import apply_rotary_emb  # noqa: E402

PAD_DIM = 8  # extra key dims (1 used); keeps head_dim a multiple of 8 for the fused kernels


class LoRA(nn.Module):
    """Low-rank deltas for the four linear layers of one block."""

    def __init__(self, blk, r, alpha):
        super().__init__()
        self.scale = alpha / r
        dims = {"qkv": blk.attn.c_attn, "out": blk.attn.c_proj, "fc": blk.mlp.c_fc, "proj": blk.mlp.c_proj}
        self.A = nn.ParameterDict({k: nn.Parameter(torch.randn(r, m.in_features) / math.sqrt(m.in_features))
                                   for k, m in dims.items()})
        self.B = nn.ParameterDict({k: nn.Parameter(torch.zeros(m.out_features, r)) for k, m in dims.items()})

    def __call__(self, name, lin, x, skip=0):
        """skip: leading positions (the prefix) that get no adapter delta, so the prefix stays released."""
        d = (x @ self.A[name].t().to(x.dtype)) @ self.B[name].t().to(x.dtype) * self.scale
        if skip:
            keep = torch.ones(x.shape[1], 1, device=x.device, dtype=x.dtype)
            keep[:skip] = 0
            d = d * keep
        return lin(x) + d


def block_fwd_ad(blk, lora, x, past_k, past_v, mask, cos, sin, kbias, skip=0):
    """Released pre-norm Block with optional LoRA and an optional per-head logit bias on past keys.

    kbias: None, or (H, Tp) additive logit bias for the Tp past keys (zeros where no bias applies).
    skip: leading positions of x without the LoRA delta (the prefix).
    """
    a = blk.attn
    lin = (lambda n, m, z: lora(n, m, z, skip)) if lora is not None else (lambda n, m, z: m(z))
    B, T, C = x.shape
    h = blk.ln_1(x)
    q, k, v = lin("qkv", a.c_attn, h).split([a.n_embd, a._kv_dim, a._kv_dim], dim=2)
    hd = C // a.n_head
    q = q.view(B, T, a.n_head, hd).transpose(1, 2)
    k = k.view(B, T, a.n_kv_head, hd).transpose(1, 2)
    v = v.view(B, T, a.n_kv_head, hd).transpose(1, 2)
    if a.n_kv_head != a.n_head:
        rep = a.n_head // a.n_kv_head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    if a.use_qk_norm:
        q, k = a.q_norm(q), a.k_norm(k)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    K = k if past_k is None else torch.cat([past_k.to(k.dtype), k], dim=2)
    V = v if past_v is None else torch.cat([past_v.to(v.dtype), v], dim=2)
    drop = a.dropout if a.training else 0.0
    if kbias is None:
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, dropout_p=drop)
    else:
        Tp = kbias.shape[1]
        kb = torch.cat([kbias.to(K.dtype) * math.sqrt(hd), kbias.new_zeros(kbias.shape[0], T).to(K.dtype)], 1)
        qa = torch.cat([q, q.new_ones(B, a.n_head, T, 1), q.new_zeros(B, a.n_head, T, PAD_DIM - 1)], -1)
        ka = torch.cat([K, kb[None, :, :, None].expand(B, -1, -1, 1), K.new_zeros(B, a.n_head, Tp + T, PAD_DIM - 1)], -1)
        y = F.scaled_dot_product_attention(qa, ka, V, attn_mask=mask, dropout_p=drop, scale=1 / math.sqrt(hd))
    y = a.resid_dropout(lin("out", a.c_proj, y.transpose(1, 2).contiguous().view(B, T, C)))
    x = x + y
    m = blk.mlp
    x = x + m.dropout(lin("proj", m.c_proj, m.gelu(lin("fc", m.c_fc, blk.ln_2(x)))))
    return x, k, v


class PlanAdapterNCM(FastChunkedNCM):
    def __init__(self, net, tk, sd=None, m_max=8, halo=0, r=16, alpha=16, plan_bias=-8.0):
        super().__init__(net, tk, sd, m_max, halo)
        for blk in net.transformer.blocks:
            assert not getattr(blk.mlp, "gated_ffn", False) and not getattr(blk.mlp, "use_relu2", False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        blocks = self.net.transformer.blocks
        self.lora = nn.ModuleList([LoRA(b, r, alpha) for b in blocks])
        n_head = blocks[0].attn.n_head
        self.plan_bias = nn.Parameter(torch.full((len(blocks), n_head), float(plan_bias)))

    # Chunk embeddings go on global-level inputs only, so local passes see exactly the released inputs
    # (with the plan bias closed, a local level is then the released model, whatever was trained).
    _cur_G = 0

    def _embed(self, x_lat, t, chunk_ids):
        tr = self.net.transformer
        e = tr.wte(x_lat) + tr.wle.weight[t]
        return e + self.chunk_emb(chunk_ids) if t < self._cur_G else e

    def forward_loss(self, codes, G, dense=False, per_code=False):
        assert not dense, "the dense path does not implement adapters"
        self._cur_G = G
        return super().forward_loss(codes, G, per_code=per_code)

    def generate(self, B, M, G, **kw):
        self._cur_G = G
        return super().generate(B, M, G, **kw)

    def _run_cached(self, h, mask, rope, past=None, want_kv=False, lora=True, n_plan=0):
        """lora: apply adapters (global passes). n_plan: number of plan keys after the prefix in `past`
        (local passes): they get the per-layer, per-head logit bias."""
        cos, sin = rope
        kvs = []
        for i, blk in enumerate(self.net.transformer.blocks):
            pk, pv = past[i] if past is not None else (None, None)
            kbias = None
            if n_plan and pk is not None:
                Tp = pk.shape[2]
                cols = torch.zeros(Tp, device=pk.device)
                cols[self.P:self.P + n_plan] = 1.0
                kbias = self.plan_bias[i][:, None] * cols[None, :]
            ad = self.lora[i] if lora else None
            skip = self.P if (lora and past is None) else 0  # global passes start with the prefix: keep it released
            if self.grad_ckpt and self.training:
                h, k, v = checkpoint(block_fwd_ad, blk, ad, h, pk, pv, mask, cos, sin, kbias, skip, use_reentrant=False)
            else:
                h, k, v = block_fwd_ad(blk, ad, h, pk, pv, mask, cos, sin, kbias, skip)
            if want_kv:
                kvs.append((k, v))
        return self.net.transformer.ln_f(h), kvs

    def _local_pass(self, emb_t, t, M, past, dev):
        l = self.lt[t]
        x, pos, valid = self._local_inputs(emb_t, M, l, dev)
        BM, T = x.shape[:2]
        past_rep = [(k.repeat_interleave(M, 0), v.repeat_interleave(M, 0)) for k, v in past]
        Tp = past_rep[0][0].shape[2]
        if valid is None:
            mask = torch.ones(1, 1, T, Tp + T, dtype=torch.bool, device=dev)
        else:
            keys = torch.cat([torch.ones(BM, Tp, dtype=torch.bool, device=dev), valid], 1)
            mask = keys[:, None, None, :].expand(BM, 1, T, Tp + T)
        hs, _ = self._run_cached(x, mask, self._rope(pos, dev), past=past_rep, lora=False, n_plan=Tp - self.P)
        h = self._halo_h(l)
        return hs[:, h:h + l]
