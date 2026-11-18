# Long-form chunked Block-NCM: design (paper 1)

Status: draft, 2026-11-19. Facts marked **[repo]** were read from `HazyResearch/ncm@main`.
Numbers marked **[calc]** come from `paper/scripts/complexity_constants.py`.

## 1. What the released NCM can and cannot do

| | TinyStories | OWT |
|---|---|---|
| Generator **[repo]** | 12 layers, d=768, 12 heads | 16 layers, d=1024, 16 heads |
| Text per sample **[repo]** | 128 BPE-8192 tokens | 1024 GPT-2 tokens |
| Levels **[repo]** | 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128 (12) | 1, 4, 9, ..., 1024 (k², 32 levels) |
| Latent positions per sample | 641 | 11,440 |
| Attention pattern **[repo]** | block-diagonal (each level attends to itself + prefix) | same |

- Generation is one forward pass per level, at most `context_length` = 129 / 1025 tokens, with no cross-level KV cache.
- `forward()` is teacher-forced over all levels in one pass and returns logits only. There is **no loss and no training script**; we write both.
- The prefix is a set of latent vectors embedded by `wte` (OWT uses `raw_token_prefix`). Prefix positions attend only to the prefix, and targets attend to the prefix. **This is the hook we reuse for the plan.**
- Known bug: issue #11 (`idx_to_zhats` in-place `add_`). Patch it before using per-level reconstructions.

## 2. Honest framing of the speed claim

Attention is a small share of generator FLOPs at these widths, because block-diagonal levels are short:

| Setup **[calc]** | Attention share φ | Attention cut | End-to-end FLOPs |
|---|---|---|---|
| TinyStories, 1 chunk | 1.8% | 4.3× | ≈1.00× |
| TinyStories, 4 chunks (512 tok) | 6.7% | 1.8× | ≈1.00× |
| TinyStories, 8 chunks (1024 tok) | 12.5% | 2.4× | 1.04× |
| OWT, 1024 tok split M=8 | 9.4% | 2.3× | 1.04× |

**So the paper must not claim a FLOP reduction.** The claims that survive are:

1. **Sequential depth.** Generating M chunks one after another costs M·K forward passes. Block-NCM costs K (K₀ global + K−K₀ batched), which is M× fewer. At small batch, decoding is memory-bound, so wall-clock follows depth.
2. **Short-context reuse.** Long outputs come from a tokenizer and generator trained on 128-token windows, with no long-context tokenizer retraining.
3. **Length extrapolation (hypothesis).** Fine levels never attend beyond their own chunk, so a model trained at M=4 should still work at M=8. A globally attending model sees out-of-distribution sequence lengths there.

The complexity section stays as the formal accounting. Its table already shows that the attention saving only pays off end to end at very long N.

## 3. Method

A document Y of M×128 tokens is split into chunks c₁..c_M. Each chunk is encoded **independently** by the frozen released tokenizer into codes r⁽ᵐ⁾₁..r⁽ᵐ⁾_K.

- **Plan P** = the coarse codes r⁽ᵐ⁾₁..r⁽ᵐ⁾_{K₀} of all chunks. It is a deterministic function of Y, so Theorem 1 applies as written.
- **Stage A (global coarse), levels 1..K₀.** Concatenate the chunks' level-k inputs into one level of length M·ℓ_k and run the generator once per level. Level-k tokens of all chunks attend to each other (block-diagonal across levels, global across chunks). Cost is tiny: for TinyStories with K₀=3 and M=8, the longest coarse level is 192 tokens.
- **Plan tokens.** Pool each chunk's reconstruction ẑ^{(K₀)} to length ℓ_{K₀+1} and concatenate across chunks, giving M·ℓ_{K₀+1} latent vectors. Add a learned **chunk embedding** (zero-initialised).
- **Stage B (local fine), levels K₀+1..K.** Fold chunks into the batch (B·M). Each chunk runs the released per-level generation with **prefix = [text prefix ; plan tokens]**. Since prefix positions attend only to the prefix, plan K/V are computed once and cached for all fine levels (the existing `use_kv_cache` prefix path).
- **Decode.** The tokenizer decodes each chunk on its own; concatenate the text.

Training (fine-tuning from the released checkpoint) uses two kinds of teacher-forced sequences, both from the existing `forward()`:
- coarse-global: levels [M·ℓ₁, ..., M·ℓ_{K₀}], with the mask rebuilt for those lengths;
- fine-local: per chunk, prefix = plan tokens, levels ℓ_{K₀+1}..ℓ_K (the released mask, longer prefix).

The loss is cross-entropy on the codes, summed over levels; we write this ourselves.

### Open design choices (decide with a quick ablation)

- **RoPE positions for plan tokens.** Option (a): span-centre positions in a shared full-resolution coordinate (0..M·128), so query–plan offsets mean something. Option (b): plain indices 0..M·ℓ_{K₀+1}−1. (a) is more principled; (b) is closer to the released model's position distribution.
- **K₀** ∈ {1, 2, 3, 4}: a richer plan versus a longer Stage A.
- **Full fine-tune vs. LoRA.** The TinyStories model (~85M) can be fully fine-tuned on 8 GB.

## 4. Baselines (same data, same fine-tuning budget)

| ID | Name | Cross-chunk information | Sequential passes |
|---|---|---|---|
| B0 | Independent chunks | none | K |
| B1 | Sequential chunk-AR (BD3-LM style) | prefix = previous chunk | M·K |
| B2 | Global (no split) | all levels attend across chunks | K |
| **Ours** | Global coarse, local fine | plan P only | K |

Expected ordering on quality: B0 < Ours ≤ B2, with B1 as the sequential reference. The NLL gap between Ours and B2 on the fine levels is the latent-space estimate of the factorization cost (Δ̂). Caveat: B2 is itself parallel within a level, so it is not an exact joint model.

## 5. Experiments

- **E1 (quality vs. M, K₀):** generative perplexity (GPT-2 Large) on the full text, unigram entropy, MAUVE if cheap. Grid: M ∈ {2, 4} × K₀ ∈ {1, 2, 3, 4}.
- **E2 (boundary coherence):** generative perplexity on windows that straddle chunk boundaries vs. windows inside a chunk. This is the failure mode reviewers will look for first.
- **E3 (systems):** sequential passes, wall-clock latency, peak VRAM, and FLOPs measured with `torch.utils.flop_counter` against the analytical table, for B1, B2 and Ours.
- **E4 (length extrapolation):** train at M ≤ 4, generate at M ∈ {4, 6}, for Ours vs. B2.
- **E5 (factorization cost):** Δ̂ in bits per latent position vs. K₀ and M.

## 6. Data

**Update 2026-11-19 (measured on the laptop):**

- **The OWT checkpoint is not released.** `owt/` in `hazyresearch/ncm-checkpoints` contains only `TODO.md`. The OWT scale-up is off unless it ships. LM1B is sentence-level and doesn't help, so **TinyStories is the only long-form dataset for paper 1.**
- **The TinyStories model was trained on story *openings*** (checkpoint README). Its generator has only seen the first 128 tokens of each story. Chunks 2..M are continuations, which are out of distribution, so fine-tuning is required for every method, including baselines.
- **Story lengths** (BPE-8192, 2,717,700 training stories; `experiments/story_lengths.py`): median 178, 95th percentile 375, max 1574 tokens.

| M | min tokens | stories | share |
|---|---|---|---|
| 2 | 256 | 318,265 | 11.7% |
| 3 | 384 | 127,829 | 4.7% |
| 4 | 512 | 46,367 | 1.7% |
| 6 | 768 | 5,190 | 0.2% |
| 8 | 1024 | 415 | ~0% |

**Plan:** train at M ∈ {2, 3, 4} (46k stories at M=4 is about 24M tokens, enough for fine-tuning). Use M=6 (5,190 stories) only as the held-out length-extrapolation test (E4). Drop M=8. Stories longer than M·128 are truncated to M·128; shorter ones are excluded from that M's split.

## 7. Environment: two corrections to the earlier checklist

- **The RTX 5050 is a Blackwell GPU (sm_120).** It needs a PyTorch build with **CUDA 12.8 or newer** (the `cu128` wheels), not just `torch>=2.4`. Check with `torch.cuda.get_arch_list()`; it must include `sm_120`.
- **FlexAttention needs Triton, which has no official Windows build.** Use **WSL2 (Ubuntu)** for training, or the eager/SDPA path (slower, higher memory). The repo's `generate` already uses the eager path, so evaluation works natively on Windows.

## 8. Milestones

0. Environment (WSL2, Python 3.12, cu128 PyTorch); clone the repo; patch #11; download the TinyStories checkpoint and dataset.
1. Reproduce the TinyStories generative perplexity reported in the paper with `evaluate.py`.
2. Run the data-length stats and fix M for TinyStories.
3. Zero-shot sanity check: B0 (independent chunks) and a training-free "plan" variant, to confirm the pipeline produces text end to end.
4. Write the training loop (loss, the two sequence types, chunk embedding); fine-tune Ours, B0, B1, B2.
5. E1–E5 on TinyStories (OWT only if its checkpoint is released).
6. Revise the paper's framing: abstract and introduction lead with sequential depth and length extrapolation, not FLOPs.
