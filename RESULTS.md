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

## 2026-11-19: Serving-cost roofline model (`paper/scripts/serving_cost.py`)

Analytic, not measured. Assumptions: 7B parameters for both AR and NCM (equal quality at equal size is
assumed, not shown), H100 specs (989 TFLOP/s, 3.35 TB/s, 80 GB), 40% FLOP and 80% bandwidth utilisation,
$2.50/GPU-hour, 2,048-token outputs (M=16), Block-NCM G=3 with a KV-cached plan. Counts dense and attention
FLOPs, weight loads and AR KV-cache reads; ignores tokenizer decode, launch overheads and low utilisation on
small passes.

| | AR (GQA-8) | B1 (chunks in sequence) | B2 (fully global) | **Block-NCM** |
|---|---|---|---|---|
| batch-1 latency | 10.8 s | 0.92 s | 0.38 s | **0.37 s** |
| batch-1 cost, $ per 1M tokens | 3.65 | 0.31 | 0.13 | **0.125** |
| batch-32 cost | 0.131 | 0.123 | 0.129 | 0.125 |
| batch-256 cost | **0.033** | 0.123 | 0.129 | 0.125 |

- **Batch 1:** Block-NCM is about 29× faster and cheaper than AR, and 2.5× faster than B1 for the same FLOPs.
  B1's small per-chunk passes are limited by weight loading.
- **Large batch:** AR is cheaper from about batch 34 (predicted B*/ρ = 148/5 = 30). At batch 256, Block-NCM
  costs about 3.8× more per token: NCM runs ρ = 5.0 positions per output token.
- **Latency also converges:** NCM is compute-bound from batch 1, so its batch latency grows linearly. At
  batch 32, Block-NCM (11.8 s) is no faster than AR (12.3 s).
- **Full multi-head attention (no GQA):** AR's KV cache (1.07 GB per sequence) caps it below batch 64 on
  80 GB, and Block-NCM is cheaper at every feasible batch (about 2× at batch 32).
- **Block-NCM vs B2 is small at this model size:** 3% faster at 2k tokens, 13% at 8k. Attention is a small
  share of FLOPs at d=4096. The local-attention saving only matters at much longer outputs.
- **The economic case is low-batch, latency-bound generation** (interactive, on-device, single-user), not
  high-throughput serving.

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

> **Warning: these generation numbers measure damage from fine-tuning, not Block-NCM.** Fine-tuned B0 also
> degrades *single-chunk* generation from genPPL 16.5 (released) to 47.7 (table below), with no chunking
> involved. Sampling settings match the released `evaluate.py` (temperature 1.0, top-k 0, top-p 1.0). The NLL
> results above are unaffected. Cause under investigation; see "Fine-tuning damages free generation" below.

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

### Fine-tuning damages free generation (diagnosis in progress)

| | genPPL | inner-window genPPL | boundary / inner |
|---|---|---|---|
| real TinyStories, first 512 tokens, 512 val docs (`experiments/real_text_ppl.py`) | **6.1** | 15.4 | **1.008** |
| released model, single 128-token samples (seed 42) | 16.5 | | |
| fine-tuned B0, single 128-token samples (seed 42) | **47.7** | | |
| released model zero-shot, M=4 (four independent openings) | 14.4 | 25.3 | 1.25 |
| fine-tuned B0 / Ours / B2, M=4 | 53.5 / 51.7 / 47.8 | 76–80 | 1.26 / 1.18 / 1.07 |

- Ruled out: the training inputs (`idx_to_ncm_input`) equal the generation inputs
  (`get_next_autoregressive_input`) bit for bit at every level, in train and eval mode. Fine-tuning leaves the
  tokenizer's quantizer unchanged (0 of 88 tensors differ). The sampling settings match `evaluate.py`.
- Symptom: fluent sentences with wrong words ("there was a that threat"). That is the signature of parallel
  sampling within a level when per-position predictions get flatter.
- The released generator's training code is not public (the repo has evaluation only), so any training-time
  trick it relied on, such as input noise, is unknown.
- Real text sets the target boundary ratio at 1.008: seams are invisible in real stories.
- **Likely cause: missing conditioning corruption.** The NCM paper (arXiv:2610.14162, Sec. 3.4, App. F.3)
  trains the generator with "a fraction of conditioning codes replaced by other codes from the same level's
  codebook" at "a linearly decreasing corruption rate across levels". The selected recipe has mean rate 0.50
  and gives TinyStories genPPL 15.41; targets stay clean. Our fine-tuning used clean inputs. The paper's
  generator recipe also differs in scale: batch 2,048, peak lr 3e-4 (floor 3e-5), weight decay 0.1,
  50k updates. Ours: batch 32 chunks, lr 1e-4, weight decay 0.01.
- **Evidence:** on story openings (chunk 1, clean teacher forcing), fine-tuned B0 beats the released model
  by a wide margin, 1.179 vs 1.558 NLL (paired over 1,228 docs, B0 better on 100%), yet generates 3× worse.
  A generator trained on corrupted inputs leans less on its conditioning: worse on clean inputs, robust to
  its own sampling errors. Clean fine-tuning reverses both.
- **Consequence:** NLL from our recipe cannot be compared with the released model, only between runs that
  share a recipe. The M=4 NLL comparison (B0 / Ours / B2) is internally consistent, but measures clean
  teacher-forced prediction, not generation.
- Fix being tested: `train.py --corrupt-mean 0.5` (uniform code replacement, linear 0.75 → 0.25 over
  conditioning levels 0..10, finest level never an input). The exact per-level rates are not in the paper's
  text, so this schedule is our reconstruction of "shifted linear, mean 0.50". The gate before more M=4 runs:
  300 openings-only steps must keep single-chunk genPPL near 16.5.

#### Gate: 300 openings-only steps (M=1, batch 32), single-chunk generation (seed 42, 256 samples)

| recipe | openings NLL, step 150 / 300 | genPPL | entropy |
|---|---|---|---|
| released, no training | 1.551 | **16.5** | 3.98 |
| lr 1e-4, clean | 1.327 / 1.272 | 41.0 | 3.99 |
| lr 1e-5, clean | 1.397 / 1.381 | 34.1 | 3.98 |
| lr 1e-4, corruption mean 0.5 (0.75 → 0.25) | 1.579 / 1.554 | **22.2** | 3.92 |
| lr 1e-4, corruption mean 0.25 (0.50 → 0.00) | 1.487 / 1.441 | 24.4 | 3.94 |
| lr 1e-4, corruption flat 0.5 | 1.574 / 1.550 | 21.8 | 3.92 |
| lr 3e-5, corruption mean 0.5 (0.75 → 0.25) | 1.536 / 1.533 | **20.7** | 3.92 |

None passes the gate. More corruption is monotonically better (0 → 41.0, 0.25 → 24.4, 0.5 → 22.2), the schedule
shape barely matters (flat 21.8 vs linear 22.2), and a lower learning rate helps slightly (20.7). All corrupted
runs land at 20.7–22.2 with entropy 3.92, even when NLL matches the released calibration.

| control and further variants | openings NLL, step end | genPPL | entropy |
|---|---|---|---|
| **control: lr 1e-12, 40 steps (train → save → load → generate)** | 1.548 | **16.6** | 3.98 |
| lr 1e-5, corruption 0.5, 300 steps | 1.532 | 20.8 | 3.91 |
| lr 3e-5, corruption 0.5, effective batch 128 (accum 4), 150 steps | 1.531 | 20.3 | 3.91 |

- The pipeline costs nothing (16.6 vs 16.5), so the gap comes from the weight updates.
- **The fine-tuned models plateau at about 20.5.** Learning rates from 1e-5 to 3e-5 give the same result, and a
  4× larger batch gains only 0.4. So neither step size nor gradient noise explains it. Most likely the model
  adapts to *our* corruption distribution, which differs in some detail from the unpublished original
  (entropy drops 3.98 → 3.91 in every corrupted run).
- **Decision:** proceed with lr 3e-5 and corruption 0.5 for the M=4 runs. The recipe costs about 4 genPPL on
  single chunks for every variant; it is measured per run and reported, so comparisons between variants stay
  fair. Matching the released recipe exactly (batch 2,048, 50k updates) is out of reach on 8 GB.

- **The damage is in the recipe, not the data:** clean training on the released model's own kind of data
  (openings) breaks generation, and a 10× lower learning rate only slows it.
- **Corruption fixes most of it:** it keeps NLL at the released calibration (1.554 vs 1.551) and recovers
  most of the generation quality, but has not yet passed the gate. The remaining samples are fluent with some
  repetition. Testing next: a lighter schedule (mean 0.25), lr 3e-5, and a flat 0.5 schedule.
- Footnote: released zero-shot at M=4 (14.4) scores *below* single-chunk (16.5). That isn't a bug: four fluent
  openings give GPT-2 Large longer context, which lowers perplexity more than the seams raise it.

- Caveat: the padding mask comes from token windows rebuilt with `prepare_chunks.py`'s selection.
  Re-encoding the first 4 docs reproduces 91% of the stored codes, not 100%. That could be encoder numerics
  (different batch size), or a document mismatch. To check: compare per document; a mismatched document would
  agree on almost no codes.
