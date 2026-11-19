"""Chunked Block-NCM: global coarse levels across chunks, local fine levels per chunk.

A document is split into M chunks of the tokenizer's context length (128 for TinyStories). Each chunk
is encoded independently by the frozen released VQ-VAE into K+1 levels of codes (level 0 plus K
target levels). The released generator is reused with one change to its attention pattern:

    G = number of *global* target levels (levels 1..G).

  * prefix tokens   attend to the prefix only (as in the released model);
  * global level t  attends to the prefix and to level t of *all* chunks;
  * local level t   attends to the prefix, to the *plan* (global level G, all chunks) and to level t
                    of its own chunk only.

G = 0 gives independent chunks (baseline B0), G = K gives a fully global model (baseline B2), and
0 < G < K is Block-NCM. Training and generation build their masks from the same rule
(`build_mask`), so the factorisation used at inference is exactly the one trained.

Within a sequence, target level t always occupies M * l_t consecutive positions, chunk-major.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from blockncm.load import NCM_ROOT  # noqa: F401  (puts the NCM repo on sys.path)
from models.common import precompute_rope_cos_sin  # noqa: E402

PREFIX, GLOBAL, LOCAL = 0, 1, 2


def build_mask(kind, lvl, chunk, G, K):
    """Boolean (1, 1, S, S) mask, True = may attend. Attributes are 1-D tensors of length S."""
    plan_lvl = G - 1 if 0 < G < K else -10**9
    qk, ql, qc = kind[:, None], lvl[:, None], chunk[:, None]
    kk, kl, kc = kind[None, :], lvl[None, :], chunk[None, :]
    k_prefix = kk == PREFIX
    allow_global = k_prefix | ((kk == GLOBAL) & (kl == ql))
    allow_local = k_prefix | ((kk == GLOBAL) & (kl == plan_lvl)) | ((kk == LOCAL) & (kl == ql) & (kc == qc))
    allow = torch.where(qk == PREFIX, k_prefix, torch.where(qk == GLOBAL, allow_global, allow_local))
    return allow[None, None]


class ChunkedNCM(nn.Module):
    def __init__(self, net, tk, sd=None, m_max=8):
        super().__init__()
        assert net.use_rope and not net.use_ctx_encoder and not net.use_post_upsample_conv
        assert not net.query_level0 and not net.predict_level0 and net.config.use_level_encoding
        assert bool(getattr(net.config, "raw_token_prefix", False))
        self.net = net
        self._vqvae = (tk.vqvae,)  # tuple keeps the frozen tokenizer out of the module tree
        self.pre_tokenizer = tk.pre_tokenizer
        self.vq = tk.vqvae.vq
        self.levels = list(self.vq.levels)
        self.l0 = self.levels[0]
        self.lt = self.levels[1:]  # target level lengths
        self.K = len(self.lt)
        self.chunk_len = self.levels[-1]
        self.offsets = [sum(self.lt[:t]) for t in range(self.K)]
        self.P = net.prefix_len
        n_embd = net.config.n_embd
        self.head_dim = n_embd // net.config.n_head
        self.rope_base = net.rope_base
        self.m_max = m_max
        self.chunk_emb = nn.Embedding(m_max, n_embd)
        nn.init.zeros_(self.chunk_emb.weight)
        self.eos = tk.pre_tokenizer.eos_token_id
        self.pad = tk.pre_tokenizer.pad_token_id
        uni = None
        if sd is not None and isinstance(sd.get("inference"), dict):
            uni = sd["inference"].get("level0_unigram")
        self.register_buffer("l0_unigram", uni.float() if uni is not None else None, persistent=False)
        self.grad_ckpt = False
        self._cache = {}

    @property
    def vqvae(self):
        return self._vqvae[0]

    # ------------------------------------------------------------------ helpers
    def _rope(self, pos, device):
        key = ("rope", tuple(pos.tolist()), str(device))
        if key not in self._cache:
            cos, sin = precompute_rope_cos_sin(pos, self.head_dim, self.rope_base)
            self._cache[key] = (cos.to(device), sin.to(device))
        return self._cache[key]

    def _train_layout(self, M, G, device):
        key = ("train", M, G, str(device))
        if key not in self._cache:
            kind, lvl, chunk, pos = [], [], [], []
            P = self.P
            kind.append(torch.full((P,), PREFIX)); lvl.append(torch.full((P,), -1))
            chunk.append(torch.full((P,), -1)); pos.append(torch.arange(P))
            for t, l in enumerate(self.lt):
                g = t < G
                kind.append(torch.full((M * l,), GLOBAL if g else LOCAL))
                lvl.append(torch.full((M * l,), t))
                chunk.append(torch.arange(M).repeat_interleave(l))
                pos.append(torch.arange(M * l) if g else torch.arange(l).repeat(M))
            kind, lvl, chunk, pos = (torch.cat(a) for a in (kind, lvl, chunk, pos))
            mask = build_mask(kind, lvl, chunk, G, self.K).to(device)
            self._cache[key] = (mask, self._rope(pos, device))
        return self._cache[key]

    def _gen_layout(self, M, G, t, device):
        """Mask/RoPE for one generation step at target level t (sequence: prefix [; plan] ; level t)."""
        key = ("gen", M, G, t, str(device))
        if key not in self._cache:
            P, l = self.P, self.lt[t]
            kind, lvl, chunk, pos = [torch.full((P,), PREFIX)], [torch.full((P,), -1)], [torch.full((P,), -1)], [torch.arange(P)]
            if t < G:  # global step: level t of all chunks
                kind.append(torch.full((M * l,), GLOBAL)); lvl.append(torch.full((M * l,), t))
                chunk.append(torch.arange(M).repeat_interleave(l)); pos.append(torch.arange(M * l))
            else:  # local step, chunks folded into the batch: prefix ; plan ; own level t
                if 0 < G < self.K:
                    lp = self.lt[G - 1]
                    kind.append(torch.full((M * lp,), GLOBAL)); lvl.append(torch.full((M * lp,), G - 1))
                    chunk.append(torch.arange(M).repeat_interleave(lp)); pos.append(torch.arange(M * lp))
                kind.append(torch.full((l,), LOCAL)); lvl.append(torch.full((l,), t))
                chunk.append(torch.zeros(l, dtype=torch.long)); pos.append(torch.arange(l))
            kind, lvl, chunk, pos = (torch.cat(a) for a in (kind, lvl, chunk, pos))
            mask = build_mask(kind, lvl, chunk, G, self.K).to(device)
            self._cache[key] = (mask, self._rope(pos, device))
        return self._cache[key]

    def _prefix_emb(self, B, device):
        ids = torch.full((B, self.P), self.eos, dtype=torch.long, device=device)
        z = self.vqvae.encoder.wte(ids)
        return self.net.transformer.wte(z)

    def _embed(self, x_lat, t, chunk_ids):
        tr = self.net.transformer
        return tr.wte(x_lat) + tr.wle.weight[t] + self.chunk_emb(chunk_ids)

    def _run(self, h, mask, rope):
        for blk in self.net.transformer.blocks:
            if self.grad_ckpt and self.training:
                h = checkpoint(blk, h, mask, rope, use_reentrant=False)
            else:
                h = blk(h, attn_mask=mask, rope_cos_sin=rope)
        return self.net.transformer.ln_f(h)

    # ------------------------------------------------------------------ training
    def forward_loss(self, codes, G, dense=False, per_code=False):
        """codes: (B, M, 1 + sum(lt)) long. Returns (mean NLL over all target codes, stats dict).

        Default path: one short sequence per level, built with the *generation* layouts
        (`_gen_layout`), i.e. teacher-forced generation. Because the mask is block-structured
        (a level sees only the prefix, the plan and itself, and the plan sees only the prefix and
        itself), this is exactly equal to the single dense-mask sequence (`dense=True`), but costs
        sum_t (P + M*l_t)^2 attention instead of (P + M*sum l_t)^2.
        """
        if dense:
            return self._forward_loss_dense(codes, G)
        B, M, L = codes.shape
        dev = codes.device
        flat = codes.reshape(B * M, L)
        with torch.no_grad():
            X = self.vq.idx_to_ncm_input(flat)
        D = X.shape[-1]
        X = X.view(B, M, -1, D)
        Y = flat[:, self.l0:].view(B, M, -1)
        pre = self._prefix_emb(B, dev)
        emb = []
        for t, l in enumerate(self.lt):
            o = self.offsets[t]
            cid = torch.arange(M, device=dev).repeat_interleave(l)
            emb.append(self._embed(X[:, :, o:o + l].reshape(B, M * l, D).to(pre.dtype), t, cid))
        plan = emb[G - 1] if 0 < G < self.K else None

        nll_sum, n_tok, per_level, codes_nll = 0.0, 0, [], []
        g_sum = g_n = l_sum = l_n = 0.0
        for t, l in enumerate(self.lt):
            mask, rope = self._gen_layout(M, G, t, dev)
            if t < G:
                hs = self._run(torch.cat([pre, emb[t]], 1), mask, rope)[:, self.P:]  # (B, M*l)
                tgt = Y[:, :, self.offsets[t]:self.offsets[t] + l].reshape(B, M * l)
            else:
                seq = [pre.repeat_interleave(M, 0)]
                if plan is not None:
                    seq.append(plan.repeat_interleave(M, 0))
                seq.append(emb[t].view(B, M, l, -1).reshape(B * M, l, -1))
                hs = self._run(torch.cat(seq, 1), mask, rope)[:, -l:]  # (B*M, l)
                tgt = Y[:, :, self.offsets[t]:self.offsets[t] + l].reshape(B * M, l)
            logits = self.net.lm_heads[t](hs).float()
            if per_code:  # (B, M, l) NLL per code, for diagnostics
                pc = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
                codes_nll.append(pc.view(B, M, l))
                nll = pc.sum()
            else:
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
            nll_sum = nll_sum + nll
            n = B * M * l
            n_tok += n
            v = nll.item()
            per_level.append(v / n)
            if t < G:
                g_sum += v; g_n += n
            else:
                l_sum += v; l_n += n
        stats = {
            "nll": nll_sum.item() / n_tok,
            "nll_global": g_sum / g_n if g_n else float("nan"),
            "nll_local": l_sum / l_n if l_n else float("nan"),
            "per_level": per_level,
        }
        if per_code:
            stats["codes_nll"] = codes_nll
        return nll_sum / n_tok, stats

    def _forward_loss_dense(self, codes, G):
        B, M, L = codes.shape
        dev = codes.device
        flat = codes.reshape(B * M, L)
        with torch.no_grad():
            X = self.vq.idx_to_ncm_input(flat)  # (B*M, T, D) next-level inputs
        D = X.shape[-1]
        X = X.view(B, M, -1, D)
        Y = flat[:, self.l0:].view(B, M, -1)

        parts, tgts = [self._prefix_emb(B, dev)], []
        for t, l in enumerate(self.lt):
            o = self.offsets[t]
            x = X[:, :, o:o + l].reshape(B, M * l, D)
            cid = torch.arange(M, device=dev).repeat_interleave(l)
            parts.append(self._embed(x.to(parts[0].dtype), t, cid))
            tgts.append(Y[:, :, o:o + l].reshape(B, M * l))
        mask, rope = self._train_layout(M, G, dev)
        h = self._run(torch.cat(parts, 1), mask, rope)

        s = self.P
        nll_sum, n_tok, per_level = 0.0, 0, []
        g_sum = g_n = l_sum = l_n = 0.0
        for t, l in enumerate(self.lt):
            n = M * l
            logits = self.net.lm_heads[t](h[:, s:s + n]).float()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgts[t].reshape(-1), reduction="sum")
            nll_sum = nll_sum + nll
            n_tok += B * n
            per_level.append(nll.item() / (B * n))
            if t < G:
                g_sum += nll.item(); g_n += B * n
            else:
                l_sum += nll.item(); l_n += B * n
            s += n
        stats = {
            "nll": nll_sum.item() / n_tok,
            "nll_global": g_sum / g_n if g_n else float("nan"),
            "nll_local": l_sum / l_n if l_n else float("nan"),
            "per_level": per_level,
        }
        return nll_sum / n_tok, stats

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def sample_l0(self, n, device):
        if self.l0_unigram is None:
            raise ValueError("checkpoint has no level-0 unigram; pass l0 explicitly")
        return torch.multinomial(self.l0_unigram.to(device), n * self.l0, replacement=True).view(n, self.l0)

    @torch.no_grad()
    def generate(self, B, M, G, temperature=1.0, top_k=0, top_p=1.0, l0=None, device="cuda"):
        """Returns codes (B, M, L). Sequential forward passes: one per target level (K)."""
        assert M <= self.m_max
        D = self.vqvae.embed_dim
        idx = l0 if l0 is not None else self.sample_l0(B * M, device)
        z_hat = torch.zeros(B * M, self.chunk_len, D, device=device)
        out = [idx]
        pre = self._prefix_emb(B, device)
        plan_h = None
        for t, l in enumerate(self.lt):
            zq = self.vq.embeddings[t](idx)
            z_hat, x = self.vq.get_next_autoregressive_input(t, z_hat, zq)  # (B*M, l, D)
            cid = torch.arange(M, device=device).repeat_interleave(l)
            h = self._embed(x.reshape(B, M * l, D).to(pre.dtype), t, cid)  # rows are b-major, chunk-minor
            mask, rope = self._gen_layout(M, G, t, device)
            if t < G:
                hs = self._run(torch.cat([pre, h], 1), mask, rope)
                logits = self.net.lm_heads[t](hs[:, self.P:])
                if t == G - 1:
                    plan_h = h
            else:
                hl = h.view(B, M, l, -1).reshape(B * M, l, -1)
                seq = [pre.repeat_interleave(M, 0)]
                if plan_h is not None:
                    seq.append(plan_h.repeat_interleave(M, 0))
                seq.append(hl)
                hs = self._run(torch.cat(seq, 1), mask, rope)
                logits = self.net.lm_heads[t](hs[:, -l:])
            idx = self.net._sample_level_raw(logits.float(), temperature, top_k, top_p).reshape(B * M, l)
            out.append(idx)
        return torch.cat(out, 1).view(B, M, -1)

    @torch.no_grad()
    def decode_tokens(self, codes):
        """codes (B, M, L) -> list of BPE id lists, chunks concatenated, cut at first EOS, PAD removed."""
        B, M, L = codes.shape
        _, zl = self.vq.idx_to_zhats(codes.reshape(B * M, L), return_zlevels=True)
        ids = self.vqvae.decoder(zl[-1]).argmax(-1).view(B, M * self.chunk_len).tolist()
        res = []
        for row in ids:
            if self.eos in row:
                row = row[:row.index(self.eos)]
            res.append([i for i in row if i != self.pad])
        return res

    @torch.no_grad()
    def decode_texts(self, codes):
        return [self.pre_tokenizer.decode(r, skip_special_tokens=True) for r in self.decode_tokens(codes)]
