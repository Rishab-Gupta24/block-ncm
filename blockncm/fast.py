"""FastChunkedNCM: Block-NCM with a KV-cached plan, optional halo attention and plan search.

Same model as ChunkedNCM (same weights, same state_dict), different execution:

* **KV-cached plan.** Global tokens attend only to the prefix and their own level (`build_mask`), so the
  prefix's and the plan's keys/values are identical for every local level. They are computed once (in the
  global pass of level G, or from the prefix alone when G = 0) and reused by every local level, in training
  and in generation. This is exact: forward_loss equals ChunkedNCM.forward_loss when halo = 0.
* **Halo attention (halo = h > 0).** A local level of chunk m also sees the level's inputs at the last h
  positions of chunk m-1 and the first h positions of chunk m+1, h capped at l_t // 4. Those inputs come from
  coarser levels that are already generated, so this adds no sequential passes. Positions are -h..-1 and
  l..l+h-1 (RoPE is relative, and NCM computes its angles from the positions directly), so the chunk's own
  positions stay 0..l-1. Halo positions at the document edges are masked out. Halo tokens are processed as
  queries, too (their outputs are discarded): about 20% extra compute at h=8.
* **Plan search (n_plans > 1, generation only).** Sample n candidate plans (global levels 1..G) per document,
  keep the one with the highest model log-probability, and run the local levels once.

Transformer blocks run through `block_fwd`, a functional copy of the released pre-norm Block that takes past
keys/values explicitly (the released block caches by mutating state, which breaks gradient checkpointing).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from blockncm.model import ChunkedNCM, GLOBAL, LOCAL, PREFIX, build_mask, vq_ctx  # noqa: F401
from models.common import apply_rotary_emb  # noqa: E402


def block_fwd(blk, x, past_k, past_v, mask, cos, sin):
    """Released Block.forward (pre-norm, no parallel/normformer/layerscale/rezero) with explicit past K/V.

    x (B, T, C); past_k/v (B, H, Tp, hd) or None, already rotated; mask bool broadcastable to (B, H, T, Tp+T);
    cos/sin (T, hd) for the T new positions. Returns (x_out, k, v) with k/v the rotated keys/values of x.
    """
    a = blk.attn
    B, T, C = x.shape
    h = blk.ln_1(x)
    q, k, v = a.c_attn(h).split([a.n_embd, a._kv_dim, a._kv_dim], dim=2)
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
    y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, dropout_p=a.dropout if a.training else 0.0)
    y = a.resid_dropout(a.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))
    x = x + y
    x = x + blk.mlp(blk.ln_2(x))
    return x, k, v


class FastChunkedNCM(ChunkedNCM):
    def __init__(self, net, tk, sd=None, m_max=8, halo=0):
        super().__init__(net, tk, sd, m_max)
        for blk in net.transformer.blocks:
            assert not (blk.parallel_block or blk.normformer or blk.use_layerscale or blk.use_rezero), \
                "block_fwd only replicates the plain pre-norm block"
        self.halo = halo

    # ------------------------------------------------------------------ helpers
    def _run_cached(self, h, mask, rope, past=None, want_kv=False):
        """Run all blocks. past: list of (k, v) per layer or None. Returns (h_out, [(k, v)] if want_kv)."""
        cos, sin = rope
        kvs = []
        for i, blk in enumerate(self.net.transformer.blocks):
            pk, pv = past[i] if past is not None else (None, None)
            if self.grad_ckpt and self.training:
                h, k, v = checkpoint(block_fwd, blk, h, pk, pv, mask, cos, sin, use_reentrant=False)
            else:
                h, k, v = block_fwd(blk, h, pk, pv, mask, cos, sin)
            if want_kv:
                kvs.append((k, v))
        return self.net.transformer.ln_f(h), kvs

    def _prefix_past(self, pre, dev):
        """K/V of the prefix alone (the past for local levels when there is no plan, G = 0)."""
        P = self.P
        mask = torch.ones(1, 1, P, P, dtype=torch.bool, device=dev)
        _, kvs = self._run_cached(pre, mask, self._rope(torch.arange(P), dev), want_kv=True)
        return kvs

    def _halo_h(self, l):
        return min(self.halo, l // 4)

    def _local_inputs(self, emb_t, M, l, dev):
        """emb_t (B, M*l, D) -> local queries (B*M, T, D), RoPE, and key mask over [past ; T] (past added later).

        Rows are b-major, chunk-minor. With a halo, each row is [left halo ; own l ; right halo].
        """
        B = emb_t.shape[0]
        x = emb_t.view(B, M, l, -1)
        h = self._halo_h(l)
        if h == 0:
            return x.reshape(B * M, l, -1), torch.arange(l), None
        zeros = x.new_zeros(B, 1, h, x.shape[-1])
        left = torch.cat([zeros, x[:, :-1, -h:]], 1)   # chunk m-1's last h (zeros for m = 0)
        right = torch.cat([x[:, 1:, :h], zeros], 1)    # chunk m+1's first h (zeros for m = M-1)
        seq = torch.cat([left, x, right], 2).reshape(B * M, l + 2 * h, -1)
        pos = torch.cat([torch.arange(-h, 0), torch.arange(l), torch.arange(l, l + h)])
        valid = torch.ones(M, l + 2 * h, dtype=torch.bool, device=dev)
        valid[0, :h] = False
        valid[M - 1, h + l:] = False
        return seq, pos, valid.repeat(B, 1)            # (B*M, T) validity of the T local keys

    def _local_pass(self, emb_t, t, M, past, dev):
        """Hidden states (B*M, l, C) of local level t given past K/V (batch B, repeated over chunks)."""
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
        hs, _ = self._run_cached(x, mask, self._rope(pos, dev), past=past_rep)
        h = self._halo_h(l)
        return hs[:, h:h + l]

    # ------------------------------------------------------------------ training
    def forward_loss(self, codes, G, dense=False, per_code=False):
        if dense:
            return self._forward_loss_dense(codes, G)
        B, M, L = codes.shape
        dev = codes.device
        flat = codes.reshape(B * M, L)
        with torch.no_grad():
            with vq_ctx():
                X = self.vq.idx_to_ncm_input(self._corrupt_inputs(flat))
        D = X.shape[-1]
        X = X.view(B, M, -1, D)
        Y = flat[:, self.l0:].view(B, M, -1)
        pre = self._prefix_emb(B, dev)
        emb = []
        for t, l in enumerate(self.lt):
            o = self.offsets[t]
            cid = torch.arange(M, device=dev).repeat_interleave(l)
            emb.append(self._embed(X[:, :, o:o + l].reshape(B, M * l, D).to(pre.dtype), t, cid))

        past = self._prefix_past(pre, dev) if G == 0 else None
        # level weighting (NCM paper Eq. 10): w_k = l_k^-alpha / sum_j l_j^(1-alpha), per code; alpha=0 = code average
        alpha = getattr(self, "level_alpha", 0.0)
        z = sum(lj ** (1 - alpha) for lj in self.lt)
        w = [lj ** -alpha / z / (B * M) for lj in self.lt]
        ref = self.__dict__.get("ref")
        anchored = ref is not None and self.training and getattr(self, "kl_weight", 0.0) > 0
        ref_past, kl_sum, kl_n, obj = None, 0.0, 0, 0.0
        nll_sum, n_tok, per_level, codes_nll = 0.0, 0, [], []
        g_sum = g_n = l_sum = l_n = 0.0
        for t, l in enumerate(self.lt):
            o = self.offsets[t]
            if t < G:
                mask, rope = self._gen_layout(M, G, t, dev)
                hs, kvs = self._run_cached(torch.cat([pre, emb[t]], 1), mask, rope, want_kv=(t == G - 1 and G < self.K))
                if t == G - 1 and G < self.K:
                    past = kvs                                  # prefix + plan K/V, reused by all local levels
                hs = hs[:, self.P:]
                tgt = Y[:, :, o:o + l].reshape(B, M * l)
            else:
                hs = self._local_pass(emb[t], t, M, past, dev)
                tgt = Y[:, :, o:o + l].reshape(B * M, l)
            logits = self.net.lm_heads[t](hs).float()
            pc = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
            if per_code:
                codes_nll.append(pc.view(B, M, l))
            nll = pc.sum()
            if anchored and t >= G:
                # forward KL(released || model) on this local level, the released model seeing each chunk alone
                with torch.no_grad():
                    if ref_past is None:
                        ref_pre = ref._prefix_emb(B, dev)
                        ref_past = ref._prefix_past(ref_pre, dev)
                    cid = torch.arange(M, device=dev).repeat_interleave(l)
                    r_emb = ref._embed(X[:, :, o:o + l].reshape(B, M * l, D).to(ref_pre.dtype), t, cid)
                    p_ref = F.softmax(ref.net.lm_heads[t](ref._local_pass(r_emb, t, M, ref_past, dev)).float(), -1)
                kl = (p_ref * (torch.log(p_ref.clamp_min(1e-30)) - F.log_softmax(logits, -1))).sum()
                kl_sum += kl.item(); kl_n += B * M * l
                obj = obj + w[t] * self.kl_weight * kl
            obj = obj + w[t] * nll
            nll_sum = nll_sum + nll
            n = B * M * l
            n_tok += n
            v = nll.item()
            per_level.append(v / n)
            if t < G:
                g_sum += v; g_n += n
            else:
                l_sum += v; l_n += n
        stats = {"nll": nll_sum.item() / n_tok, "nll_global": g_sum / g_n if g_n else float("nan"),
                 "nll_local": l_sum / l_n if l_n else float("nan"), "per_level": per_level}
        if per_code:
            stats["codes_nll"] = codes_nll
        if anchored:
            stats["kl"] = kl_sum / kl_n
        if self.training and (alpha or anchored):
            return obj, stats          # weighted / anchored training objective; stats stay the plain code average
        return nll_sum / n_tok, stats

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, B, M, G, temperature=1.0, top_k=0, top_p=1.0, l0=None, device="cuda", n_plans=1):
        """Returns codes (B, M, L). K sequential passes; the plan K/V are computed once.

        n_plans > 1: sample n plans per document (levels 1..G, batched), keep the most likely, then run
        the local levels once. Requires 0 < G < K.
        """
        assert M <= self.m_max
        n = n_plans if 0 < G < self.K else 1
        Bn = B * n
        D = self.vqvae.embed_dim
        idx = l0.repeat_interleave(n, 0) if l0 is not None else self.sample_l0(Bn * M, device)
        z_hat = torch.zeros(Bn * M, self.chunk_len, D, device=device)
        out = [idx]
        pre = self._prefix_emb(Bn, device)
        past = self._prefix_past(pre, device) if G == 0 else None
        logp = torch.zeros(Bn, device=device)
        for t, l in enumerate(self.lt):
            if t == G and n > 1:  # choose the most likely plan per document, drop the other candidates
                best = logp.view(B, n).argmax(1) + torch.arange(B, device=device) * n
                rows = (best[:, None] * M + torch.arange(M, device=device)).reshape(-1)  # (B*M) chunk rows
                out = [o[rows] for o in out]
                idx, z_hat = out[-1], z_hat[rows]
                past = [(k[best], v[best]) for k, v in past]
                pre, Bn = pre[best], B
            with vq_ctx():
                zq = self.vq.embeddings[t](idx)
            with vq_ctx():
                z_hat, x = self.vq.get_next_autoregressive_input(t, z_hat, zq)
            cid = torch.arange(M, device=device).repeat_interleave(l)
            h = self._embed(x.reshape(Bn, M * l, D).to(pre.dtype), t, cid)
            if t < G:
                mask, rope = self._gen_layout(M, G, t, device)
                keep = t == G - 1 and G < self.K
                hs, kvs = self._run_cached(torch.cat([pre, h], 1), mask, rope, want_kv=keep)
                if keep:
                    past = kvs
                logits = self.net.lm_heads[t](hs[:, self.P:]).float()
            else:
                logits = self.net.lm_heads[t](self._local_pass(h, t, M, past, device)).float()
            idx = self.net._sample_level_raw(logits, temperature, top_k, top_p).reshape(Bn * M, l)
            if t < G and n > 1:
                lp = torch.log_softmax(logits / max(temperature, 1e-6), -1)
                logp += lp.gather(-1, idx.view(Bn, M * l, 1)).view(Bn, -1).sum(1)
            out.append(idx)
        return torch.cat(out, 1).view(B, M, -1)
