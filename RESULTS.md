# Results log

Hardware: RTX 5050 Laptop GPU (8 GB), WSL2 Ubuntu 24.04, PyTorch 2.11.0+cu128.

## 2026-11-19: Milestone 1, reproduce the released TinyStories model

`python evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir checkpoints/tinystories`
(5 seeds × 256 unconditional samples, generative perplexity under GPT-2 Large)

| seed | genPPL | entropy |
|---|---|---|
| 42 | 16.57 | 3.9784 |
| 0 | 16.66 | 3.9721 |
| 1 | 16.24 | 3.9745 |
| 2 | 16.25 | 3.9782 |
| 3 | 16.18 | 3.9700 |
| **mean** | **16.38 ± 0.22** | **3.9746** |

The README gives no TinyStories number to compare against; check the paper's table.

## 2026-11-19: Milestone 2, story lengths

See `docs/long_form_design.md` §6. Chunked training is feasible for M ∈ {2, 3, 4}; M = 6 is used for evaluation only.

## 2026-11-19: Implementation checks (`experiments/test_equivalence.py`)

- `ChunkedNCM` with M=1 reproduces the released teacher-forced loss exactly for G=0 and G=K
  (1.290056 vs 1.290056, diff 1.2e-7).
- Tokenizer round trip on real 128-token chunks: 100% token accuracy.
- Generation runs for (M, G) ∈ {(1,0), (2,0), (2,3), (2,11), (4,3)}. Zero-shot M=4 output is garbled, as expected:
  the released generator was trained only on story *openings*, so fine-tuning is required.
- Per-level training path (one short sequence per level, built with the generation layouts) equals the
  single dense-mask sequence for M=4, G ∈ {0, 1, 3, 6, 11} (max per-level difference < 1e-6). The dense path
  does not fit in 8 GB at M=4, so training uses the per-level path.

## 2026-11-19: Milestone 3, first M=4 fine-tuning runs (one seed)

Setup: M=4 (512 tokens, documents with more than 384 tokens; 125,723 train / 1,228 val), fine-tune all generator
weights from the released TinyStories checkpoint, 1,500 steps × 8 documents (about 10% of one epoch), AdamW lr 1e-4
(chunk embedding 1e-3), warmup 200, cosine to 0.1×, bf16, gradient checkpointing. RTX 5050: 53–81 min per run.

| | B0: independent chunks (G=0) | **Ours (G=3)** | B2: fully global (G=11) |
|---|---|---|---|
| sequential forward passes to generate | 11 | **11** | 11 (one pass sees all 4 chunks) |
| val NLL, 512 docs, step 0 | 2.471 | 3.107 | 3.101 |
| val NLL, step 500 | 1.643 | 1.634 | 1.643 |
| val NLL, step 1000 | 1.532 | 1.521 | 1.520 |
| val NLL, step 1500 | 1.4853 | **1.4727** | 1.4703 |
| train time | 53 min | 81 min | 62 min |

Ours closes 84% of the B0 → B2 gap. The whole gap is small (0.015 nats) and this is one seed.

### NLL breakdown, step 1500 (`experiments/nll_breakdown.py`, all 1,228 val docs)

Non-padding codes only. 37% of chunk-4 codes are padding (stories that end before token 512); padding is
equally easy for every model and dilutes the averages.

| | B0 | **Ours** | B2 | Ours: share of B0→B2 gap closed |
|---|---|---|---|---|
| total | 1.3309 | **1.3154** | 1.3108 | 77% |
| coarse, levels 1–3 (48 of 640 codes) | 4.9528 | **4.8339** | 4.8169 | 87% |
| fine, levels 4–11 | 1.0360 | **1.0289** | 1.0253 | 66% |
| level 1 (l=8) | 6.520 | 6.527 | 6.516 | none |
| level 2 (l=16) | 5.437 | 5.284 | 5.280 | 97% |
| level 3 (l=24) | 4.103 | 3.965 | 3.937 | 83% |
| level 4 (l=32) | 3.066 | 2.975 | 2.853 | 43% |
| level 5 (l=40) | 2.033 | 1.999 | 1.915 | 29% |
| level 6 (l=52) | 1.283 | 1.280 | 1.264 | |
| levels 7–11 | ≈ equal | ≈ equal | **worse** (e.g. level 9: 1.205 vs 1.177) | |
| chunk 1 | 1.179 | 1.175 | 1.174 | |
| chunk 2 | 1.385 | 1.365 | 1.356 | |
| chunk 3 | 1.410 | 1.391 | 1.385 | |
| chunk 4 | 1.361 | 1.339 | 1.340 | 100% |

What this shows:
- Masking padding raises the gap from 0.015 to 0.020 nats, and Ours still closes 77% of it.
- The gain is concentrated where the method predicts. On levels 2–3 (part of the plan), Ours is
  0.14–0.15 nats better than B0. Chunks 2–4 gain about 4× more than chunk 1.
- **The local fine decoder uses the plan.** Levels 4–5 are local in Ours, yet still beat B0 by 0.03–0.09 nats.
- Level 1 (8 codes per chunk) gains nothing from global attention in either model.
- **B2 is worse than both on levels 7–11.** Global attention at the finest levels hurts at this budget,
  plausibly because it is under-trained and sees RoPE positions up to 4×128 that the released model never saw.
  B2 wins on levels 4–5, where Ours captures only 29–43% of the gap. So the best G is probably between 3 and 6.
  The G ∈ {4, 5, 6} ablation is the obvious next run.
### Generation, step 1500 (`experiments/sample_eval.py`, 2 seeds × 256 samples, temperature 1.0)

| | B0 | **Ours (G=3)** | B2 | Ours: share of B0→B2 gap closed |
|---|---|---|---|---|
| genPPL, GPT-2 Large (mean ± sd over seeds) | 53.5 ± 0.4 | **51.7 ± 0.8** | 47.8 ± 0.8 | 31% |
| boundary-window genPPL (32 + 32 tokens across a chunk seam) | 96.6 | 94.8 | 85.8 | |
| inner-window genPPL (64 tokens inside a chunk) | 76.4 | 80.1 | 80.2 | |
| **boundary / inner ratio** | 1.264 | **1.184** | 1.071 | 41% |
| unigram entropy | 4.72 | 4.65 | 4.61 | |
| mean tokens, share reaching chunk 4 | 448, 98% | 433, 97% | 431, 96% | |
| latency, s/sample (batch 64, 11 passes each) | 0.094 | 0.091 | 0.098 | |

What this shows:
- **The ranking matches NLL, but Ours closes much less of the gap here: 31–41% vs 77–84%.**
  B2 is clearly best on every generation metric at M=4.
- **The seams are where Ours falls short.** The boundary ratio falls from 1.26 (B0) to 1.18 (Ours), against
  1.07 for B2. A level-3 plan carries topic and characters, but the seam needs token-level continuity: the
  sentence ending in chunk m has to run on into chunk m+1. In Ours that is decided by the local fine levels,
  which cannot see the neighbouring chunk.
- **Latency is identical at M=4.** All three take 11 passes; B2's global attention over 4×l tokens costs
  nothing noticeable here. Ours' cost advantage over B2 only matters at larger M, or measured in memory and
  FLOPs. At M=4 there is no speed case for Ours over B2.
- **Overall quality is low for every model:** genPPL around 50, against 16.4 for the released model on single
  128-token samples. The two are not directly comparable (4× longer texts, and windows scored without
  context). Still, the samples show local garbling ("six, six, five bugs"). A real-text reference is missing:
  genPPL of real 512-token TinyStories and of the zero-shot model (running).

Candidate fix for the seams: **halo attention.** Let local level t also attend to the level-t *inputs* of the
nearest w positions of the neighbouring chunks. Level-t inputs come from levels < t, which are already
generated, so this adds no sequential passes. Its cost is O(l·(l + 2w)) instead of B2's O((M·l)²).
It targets exactly the deficit measured here.

- Caveat: the padding mask comes from token windows rebuilt with `prepare_chunks.py`'s selection.
  Re-encoding the first 4 docs reproduces 91% of the stored codes, not 100%. That could be encoder numerics
  (different batch size), or a document mismatch. To check: compare per document; a mismatched document would
  agree on almost no codes.
