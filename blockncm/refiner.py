"""RefinerNCM: a shallower network for the local levels (layer-drop refiner), on top of FastChunkedNCM.

* Global passes (and local levels below `start_level`) run the full frozen generator.
* Local levels >= `start_level` run a **separate copy** of a subset S of the generator's blocks (NCM shares
  weights across levels, so a copy is needed to train the refiner without touching the global passes),
  followed by its own copy of the final LayerNorm and the shared (frozen) codebook heads.
* Local block j attends to the cached prefix/plan keys/values of global layer S[j], so the plan sits in the
  representation space that layer was trained for.
* Training (optional distillation): cross-entropy on the true codes plus `distill` × KL(teacher || refiner) on
  the refiner's levels, where the teacher is the full-depth model on the same (corrupted) inputs and plan.

With S = all layers the model equals FastChunkedNCM exactly (checked in experiments/test_refiner.py).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from blockncm.fast import FastChunkedNCM, block_fwd

PATTERNS = {  # layer subsets for the probe
    6: {"prefix": [0, 1, 2, 3, 4, 5], "uniform": [0, 2, 4, 6, 8, 11], "symmetric": [0, 1, 2, 9, 10, 11]},
    3: {"prefix": [0, 1, 2], "uniform": [0, 5, 11], "symmetric": [0, 1, 11]},
}


def refiner_from_run(net, tk, sd, run_state, **kw):
    """Load a trained FastChunkedNCM/ChunkedNCM state into `net`, then build a RefinerNCM whose refiner blocks
    are copies of the *trained* blocks (order matters: load first, copy second)."""
    tmp = FastChunkedNCM(net, tk, sd)
    tmp.load_state_dict(run_state, strict=True)
    r = RefinerNCM(net, tk, sd, **kw)
    r.chunk_emb.load_state_dict(tmp.chunk_emb.state_dict())
    return r


class RefinerNCM(FastChunkedNCM):
    def __init__(self, net, tk, sd=None, m_max=8, halo=0, layers=None, start_level=4, distill=0.0):
        super().__init__(net, tk, sd, m_max, halo)
        n = len(net.transformer.blocks)
        self.layers = list(range(n)) if layers is None else list(layers)
        assert all(0 <= i < n for i in self.layers)
        self.start_level = start_level  # 1-based target level from which the refiner is used
        self.distill = distill
        self.ref_blocks = nn.ModuleList([copy.deepcopy(net.transformer.blocks[i]) for i in self.layers])
        self.ref_ln_f = copy.deepcopy(net.transformer.ln_f)
        self._kl = None

    def freeze_base(self):
        """Freeze everything except the refiner copy (global passes = fixed teacher weights)."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in list(self.ref_blocks.parameters()) + list(self.ref_ln_f.parameters()):
            p.requires_grad_(True)

    def _uses_refiner(self, t):
        return t + 1 >= self.start_level

    def _run_ref(self, h, mask, rope, past):
        cos, sin = rope
        for j, blk in enumerate(self.ref_blocks):
            pk, pv = past[self.layers[j]]
            if self.grad_ckpt and self.training:
                h, _, _ = checkpoint(block_fwd, blk, h, pk, pv, mask, cos, sin, use_reentrant=False)
            else:
                h, _, _ = block_fwd(blk, h, pk, pv, mask, cos, sin)
        return self.ref_ln_f(h)

    def _local_pass(self, emb_t, t, M, past, dev):
        if not self._uses_refiner(t):
            return super()._local_pass(emb_t, t, M, past, dev)
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
        rope = self._rope(pos, dev)
        h = self._halo_h(l)
        hs = self._run_ref(x, mask, rope, past_rep)[:, h:h + l]
        if self.training and self.distill > 0:  # KL to the full-depth teacher on the same inputs and plan
            with torch.no_grad():
                ht, _ = self._run_cached(x, mask, rope, past=past_rep)
                pt = F.log_softmax(self.net.lm_heads[t](ht[:, h:h + l]).float(), -1)
            ps = F.log_softmax(self.net.lm_heads[t](hs).float(), -1)
            kl = (pt.exp() * (pt - ps)).sum(-1)  # (BM, l)
            self._kl = (self._kl[0] + kl.sum(), self._kl[1] + kl.numel())
        return hs

    def forward_loss(self, codes, G, dense=False, per_code=False):
        assert not dense, "the dense path does not implement the refiner"
        z = torch.zeros((), device=codes.device)
        self._kl = (z, 0)
        loss, st = super().forward_loss(codes, G, per_code=per_code)
        if self.training and self.distill > 0 and self._kl[1]:
            kl = self._kl[0] / self._kl[1]
            st["kl"] = kl.item()
            loss = loss + self.distill * kl
        self._kl = None
        return loss, st
