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

## 2026-11-19: Tokenizer level-skip probe (`experiments/level_skip_probe.py`, no generator, no training)

Tests the "skip fine levels" part of the Cascade-NCM proposal: can the frozen residual VQ tokenizer reconstruct
text when some levels are never produced? 1,024 real 128-token chunks (256 val docs, no padding). Each variant
encodes, drops the levels, decodes, and compares tokens.
- **requant:** re-run the residual quantizer without those levels, so later levels absorb the residual. This is
  the best case, for a generator retrained on that tokenization.
- **hole:** normal codes; the skipped levels are simply left out at decode time. This is what a generator trained
  on the normal codes would face.

Level numbers as in RESULTS.md (level 7 = l=64).

| skipped levels | positions / chunk | requant: token acc | requant: exact chunks | hole: token acc | hole: exact chunks |
|---|---|---|---|---|---|
| none | 641 | 99.97% | 96.7% | 99.97% | 96.7% |
| 11 (l=128) | 513 | 98.26% | 17.9% | 98.26% | 17.9% |
| 10 | 533 | 99.95% | 93.8% | 99.93% | 92.5% |
| 9 | 549 | 99.94% | 93.2% | 99.91% | 89.1% |
| 8 | 565 | 99.91% | 90.7% | 99.86% | 85.1% |
| 7 | 577 | 99.90% | 88.7% | 99.61% | 62.7% |
| 7, 9 (6→8→10→11) | 485 | 99.78% | 77.5% | 99.14% | 36.7% |
| **7, 8, 10 (Cascade: 6→9→11)** | **393** | **99.52%** | 57.0% | 98.41% | 14.4% |
| 5, 7, 9 | 445 | 99.53% | 57.6% | 98.00% | 9.9% |
| 7–10 (6→11) | 301 | 98.71% | 23.5% | 97.10% | 3.6% |
| 4–10 (3→11) | 177 | 91.89% | 0% | 84.81% | 0% |

What this shows:
- **The tokenizer tolerates skipped intermediate levels; the last level is the one it can't lose.** Cascade's
  6→9→11 keeps 99.5% of tokens (re-quantized) with 39% fewer positions. Skipping level 11 alone already costs
  1.7%.
- Re-quantizing matters: with the normal codes ("hole"), the same skip keeps 98.4%. That's still high per token,
  but only 14% of chunks come back exactly. So a level-skipping generator would need codes from the skipped
  tokenization, which means re-encoding the data.
- These are upper bounds with perfect codes. The generator still has to predict levels 9 and 11 across a bigger
  jump. Measured separately, cutting *depth* on these levels failed (Milestone 4). Level-skipping at full depth
  remains open and is the cheaper idea to test next.

## 2026-11-20: Second seed of the main models (Kaggle Version 7, `kaggle/run_seed.sh`, `--seed 1`)

Same recipe; all four runs share one Kaggle encoding, so their NLL is comparable with each other (and with the
Milestone 5b Kaggle runs), not with local NLL. Seed-0 values: B0, Ours G=3 + halo 4 and B2 from the local runs
(Milestone 5), G=5 + halo 4 from Milestone 5b.

| | B0 | Ours G=3 + halo 4 | **Ours G=5 + halo 4** | B2 |
|---|---|---|---|---|
| NLL, seed 1 (Kaggle encoding) | 1.6156 | **1.5999** | **1.6002** | 1.6353 |
| Δ NLL vs B0, seed 1, 95% CI | — | −0.0157 [−0.0164, −0.0150] ‡ | −0.0157 ‡ | +0.0197 |
| genPPL M=4, seed 0 / seed 1 / **mean** | 24.1 / 24.0 / **24.0** | 21.0 / 20.9 / **21.0** | 19.1 / 19.0 / **19.05** | 19.0 / 18.4 / **18.7** |
| boundary ratio M=4, seed 0 / 1 / **mean** | 1.52 / 1.48 / **1.50** | 1.19 / 1.21 / **1.20** | 1.14 / 1.13 / **1.14** | 1.17 / 1.09 / **1.13** |
| **genPPL M=6**, seed 0 / 1 / **mean** | 23.7 / 23.7 / **23.7** | 21.5 / 20.6 / **21.1** | 20.0 / 21.0 / **20.5** | 33.5 / 31.0 / **32.3** |
| boundary ratio M=6, seed 1 | 1.50 | 1.19 | 1.15 | 1.13 |

‡ From the paired bootstrap against B2 in the seed-1 breakdown (−0.0353 for both Ours runs vs B0's −0.0195).

What replicates and what does not:
- **Replicates: B2 collapses beyond the training length** (19.0 → 33.5 and 18.4 → 31.0), while Ours G=5 + halo 4
  (20.0 / 21.0) and G=3 + halo 4 (21.5 / 20.6) hold. The main claim of the paper survives a second seed.
- **Replicates: NLL ranking.** Both halo models beat B0 and B2 on NLL; B2 is worst again (levels 6–11, chunk 4).
- **Replicates: halo fixes the seams** (B0 1.50 → 1.14–1.20 averaged).
- **Corrected: at M=4, B2 is slightly better than G=5 + halo 4** (18.7 vs 19.05 averaged; seed 1: 18.4 vs 19.0),
  and the seams are a tie (1.13 vs 1.14). Seed 0's "G=5 + halo 4 ties B2 with better seams" was within noise. The
  accurate claim: B2 leads by about 0.4 genPPL at the training length and loses by about 12 at 1.5× the length.
- G=3 and G=5 with halo tie on NLL in seed 1 (1.5999 vs 1.6002); G=5 generates better at M=4 (19.05 vs 21.0).

## 2026-11-19: AR reference vs base NCM vs Block-NCM (`experiments/ar_baseline.py`)

AR model: `roneneldan/TinyStories-33M` (GPT-Neo, 68.5M parameters with embeddings, trained on TinyStories), the
closest public stand-in for the NCM paper's 88.6M AR baseline (no public checkpoint). Sampled at temperature 1.0,
top-k 0, top-p 1.0 from end-of-text, cut to M×128 BPE-8192 tokens, scored exactly like `sample_eval.py` (GPT-2 Large,
2 seeds × 256 samples). AR stories end early (217 tokens on average at M=4), so a length-matched run blocks
end-of-text for 400 AR tokens (457 tokens on average, our samples: about 425).

| | sequential passes | genPPL ↓ | entropy | boundary / inner ratio | latency, s/sample |
|---|---|---|---|---|---|
| **128 tokens** | | | | | |
| real text (NCM paper) | — | 9.67 | 3.96 | — | — |
| AR 88.6M (NCM paper, Table 8) | 128 | 8.89 | 4.01 | — | — |
| AR TinyStories-33M (ours) | ≈ 129 | 9.68 | 3.98 | — | 0.039 |
| base NCM (our rerun) | 11 † | 16.38 | 3.97 | — | — |
| **512 tokens** | | | | | |
| real text | — | 6.1 | 4.46 | 1.01 | — |
| AR TinyStories-33M, natural length (217 tokens) | ≈ 219 | 9.17 | 4.21 | 1.12 | 0.17 |
| AR TinyStories-33M, length-matched (457 tokens) | ≈ 462 | 9.93 | 4.61 | 1.04 | 0.24 |
| base NCM, 4 independent chunks | 11 | 14.4 | — | 1.25 | — |
| B2 (all global) | 11 | 19.0 | 4.36 | 1.17 | — |
| **Ours G=5 + halo 4** | **11** | **19.1** | 4.35 | **1.14** | 0.08 (T4) |
| **768 tokens:** Ours G=5 + halo 4 / B2 | 11 | 20.0 / 33.5 | | 1.15 / 1.15 | |

AR sequential steps are the AR model's own tokens: 1.011 GPT-Neo tokens per BPE-8192 token on TinyStories (500 real
stories, sd 0.023), so 457 BPE tokens ≈ 462 steps. The script's `max_new_tokens` cap (784) only bounds generation;
samples are then cut to 512 BPE tokens (≈ 517 AR tokens). TinyStories-33M's config lists 2,048 positions, but it is
reported to have been trained on 512-token contexts (local attention window 256); our samples stay within that
(mean 462, at most about 517 AR tokens), so AR is evaluated at 512 tokens only, not at 768.
† The NCM paper counts 12 function evaluations at 128 tokens. Level 0 (1 code) is drawn from a stored unigram table,
not a transformer pass, so it is 11 transformer passes, as in the 512-token rows.

What this shows:
- **TinyStories-33M is a fair AR stand-in:** at 128 tokens it scores 9.68 against 8.89 for the paper's 88.6M AR and
  9.67 for real text.
- **AR writes about 2× better text than our best model at 512 tokens** (9.9 vs 19.1, length-matched). Base NCM is
  already 1.7× behind AR at 128 tokens (16.4 vs 9.7); our fine-tuning recipe costs about 4 more on single chunks
  (the gap the gates are testing). Block-NCM's contribution is making NCM work at length, not matching AR quality.
- **The seam metric has a floor above 1.0 on generated text:** AR has no seams at all and scores 1.04
  (length-matched) and 1.12 (natural length). So our 1.14 is close to that floor, and B0's 1.52 is clearly not.
- **Length-matched without blocking AR's end-of-text** (first 217 tokens of our M=4 samples vs natural-length AR,
  9.17): B0 23.8, Ours G=3 21.9, Ours G=3 + halo 4 21.5, Ours G=5 20.4, B2 19.7. Same ranking and the same ~2× gap
  as the full-length comparison, so the gap is not a length artefact.
- **Style caveat:** TinyStories-33M was trained on all stories and writes mostly "Once upon a time" openings, while
  our M=4 models write in the long-story style (see the recipe-gap section), which GPT-2 Large scores about 1.1×
  higher on real text. Part of the AR gap is therefore style, not quality.
- **Speed:** 11 sequential passes vs about 462 for AR at matched length (one per token). Measured latency favours Block-NCM (0.08 s/sample on a T4 at
  batch 32 vs 0.24 s for AR on the RTX 5050 at batch 64) even though the AR model is smaller, but the AR side uses
  unoptimised Hugging Face `generate`, so this is indicative only. The 7B cost model (26.5× faster at batch 1, AR
  cheaper from batch 32) remains the cost estimate.

## 2026-11-19: Sentence-aligned chunks and plan-only adapters (local, one seed)

**Sentence-aligned chunks (negative result).** Chunks end at the last sentence end before 128 tokens, and the rest
of the chunk is padding (`prepare_chunks.py --align sentence`). Same recipe as Milestone 3b. The idea was that seams
at sentence ends would be easier to join.

| | B0, fixed | B0, sentence-aligned | Ours G=3, fixed | Ours G=3, sentence-aligned |
|---|---|---|---|---|
| genPPL, M=4 | 24.1 | 27.0 | 22.0 | 24.3 |
| boundary / inner ratio | 1.52 | 1.76 | 1.39 | 1.54 |

Both get worse, and the seams do not get easier. A likely reason (not tested): padded chunks hold fewer real tokens,
and the model must also learn where each chunk's padding starts. Fixed-length chunks plus halo stay the design.

**Plan-only adapters (frozen generator, LoRA rank on the global passes + plan bias; 2.4M trainable parameters).**
NLL on the same 1,228 val docs (non-padding):

| | released | B0 | Ours G=3 | **plan-only adapters** | B2 |
|---|---|---|---|---|---|
| total | 2.0964 | 1.7486 | **1.7438** | 1.8343 | 1.7780 |
| fine, levels 4–11 | 1.7468 | 1.4616 | 1.4556 | 1.5508 | 1.5022 |
| chunk 4 | 2.7750 | 1.8040 | **1.7933** | 2.2235 | 1.8718 |

The adapters are the worst fine-tuned model on NLL, worst of all on chunk 4. Their local levels are the released
model's, which never saw chunks 2–4. The plan bias barely opened (−6.0 → −5.19). Generation (GPT-2 Large, 2 seeds × 256):

| | B0 | Ours G=3 | **plan-only adapters** |
|---|---|---|---|
| genPPL, M=4 | 24.1 | 22.0 | 28.5 |
| boundary / inner ratio, M=4 | 1.52 | 1.39 | 1.24 |
| genPPL, M=6 | 23.7 | 21.8 | 24.0 |
| entropy, M=4 / M=6 | 4.50 / — | 4.39 / — | 4.49 / 4.59 |

**Negative result: freezing the generator did not keep the released model's fluency.** The adapters generate worst
of all at M=4 (28.5). Their seams are fairly good (1.24), plausibly because the frozen local levels still write each
chunk as a fluent opening. Their high entropy and the lower M=6 score (24.0, longer samples give GPT-2 Large more
context) fit the same picture: locally fluent text that does not follow one story. Full fine-tuning stays the method.

## 2026-11-19: Milestone 5b, halo widths and halo + larger plans (Kaggle, 2× T4, fp16, one seed)

Same recipe as Milestone 5 (corruption 0.5, lr 3e-5, 1,500 steps, `--fast`), trained on Kaggle (`kaggle/run_halo.sh`,
Version 6, all stages passed). genPPL: GPT-2 Large, 2 seeds × 256 samples. The local columns are from Milestone 5.

| | Ours G=3 + halo 2 | Ours G=3 + halo 4 (local) | Ours G=3 + halo 8 | **Ours G=5 + halo 4** | Ours G=6 + halo 4 | B2 (local) |
|---|---|---|---|---|---|---|
| genPPL, M=4 | 21.4 | 21.0 | 20.4 | **19.1** | 18.7 | 19.0 |
| boundary / inner ratio, M=4 | 1.22 | 1.19 | 1.17 | **1.14** | 1.12 | 1.17 |
| entropy, M=4 | 4.39 | 4.41 | 4.39 | 4.35 | 4.31 | 4.36 |
| **genPPL, M=6** (trained on M=4) | 21.1 | 21.5 | 20.4 | **20.0** | 23.6 | 33.5 |
| boundary / inner ratio, M=6 | 1.19 | 1.19 | 1.21 | **1.15** | 1.14 | 1.15 |
| genPPL sd over 2 seeds, M=6 | 0.03 | 0.3 | 0.36 | 0.17 | 0.32 | 2.3 |
| NLL, Kaggle encoding (non-padding) | 1.6002 | — | 1.5997 | **1.5976** | 1.6114 | — |

What this shows:
- **G=5 + halo 4 is the best model overall.** At the training length it ties B2 (19.1 vs 19.0) with better seams
  (1.14 vs 1.17), and at M=6 it is the best of all runs (20.0 vs B2's 33.5). *(Seed 1 corrects the M=4 part: B2
  leads by about 0.4 and seams tie; see the 2026-11-20 second-seed section.)* Halo also fixes G=5's extrapolation:
  without halo G=5 went 20.0 → 23.8 at M=6; with halo it goes 19.1 → 20.0.
- **G=6 + halo 4 generates best at M=4 (18.7, below B2) but still degrades at M=6 (23.6).** Six global levels
  learn the training length even with halo. Level 6 NLL is again worse when it is global (1.79 vs 1.59–1.61).
- **Wider halos help a little:** h = 2 / 4 / 8 give genPPL 21.4 / 21.0 / 20.4 and boundary ratio 1.22 / 1.19 / 1.17
  at G=3. h=8 costs about 2× the halo compute of h=4 (about 20% of local-level positions instead of 10%).
- **NLL:** among the Kaggle runs G=5 + halo 4 is best (−0.0138 vs G=6 + halo 4, 95% CI [−0.0144, −0.0133], better on
  93% of docs); halo 8 vs halo 2 is a tie (1.5997 vs 1.6002).
- **Kaggle NLL is not comparable with local NLL.** The released model itself scores 1.9558 on the Kaggle encoding and
  2.0966 on the local one (non-padding), so the two machines' code files differ, most likely through quantizer
  precision during encoding (F4: the bf16 quantizer loses precision). genPPL is measured on decoded text, so it is
  comparable across machines; NLL is compared only within a machine. **To check:** encode the local data with an
  fp32 quantizer and compare.
- One seed throughout. The G=5 + halo 4 vs B2 tie at M=4 is within noise; the M=6 gap (20.0 vs 33.5) is not.

## 2026-11-19: Milestone 5, halo attention and the G sweep (same recipe and data as Milestone 3b, one seed)

All runs: corruption 0.5, lr 3e-5, 1,500 steps, KV-cached plan (`--fast`). NLL from `nll_breakdown.py` on all
1,228 val docs, non-padding codes; the B0/Ours G=3/B2 columns are the Milestone 3b runs.

| | B0 (G=0) | Ours G=3 | **Ours G=3 + halo 4** | Ours G=5 | Ours G=6 | B2 (G=11) |
|---|---|---|---|---|---|---|
| total NLL | 1.7486 | 1.7438 | **1.7361** | 1.7377 | 1.7471 | 1.7780 |
| coarse, levels 1–3 | 5.2730 | 5.2823 | 5.2689 | 5.1710 | 5.1660 | **5.1647** |
| fine, levels 4–11 | 1.4616 | 1.4556 | **1.4484** | 1.4581 | 1.4686 | 1.5022 |
| chunk 4 | 1.8040 | 1.7933 | **1.7843** | 1.7949 | 1.8146 | 1.8718 |
| Δ vs B0, 95% CI | — | −0.0049 [−0.0056, −0.0042] | **−0.0127 [−0.0134, −0.0120]**, better on 85% of docs | −0.0111 [−0.0120, −0.0103] | −0.0018 [−0.0027, −0.0008] | +0.0292 |
| genPPL, M=4 | 24.1 | 22.0 | 21.0 | 20.0 | 19.6 | **19.0** |
| boundary / inner ratio, M=4 | 1.52 | 1.39 | 1.19 | 1.26 | 1.22 | **1.17** |
| entropy, M=4 | 4.50 | 4.39 | 4.41 | 4.35 | 4.34 | 4.36 |
| **genPPL, M=6** (trained on M=4) | 23.7 | 21.8 | **21.5** | 23.8 | 30.7 | 33.5 |
| boundary / inner ratio, M=6 | 1.49 | 1.34 | 1.19 | 1.23 | 1.20 | **1.15** |
| genPPL sd over 2 seeds, M=6 | 0.1 | 0.3 | 0.3 | 1.3 | 2.1 | 2.3 |

What this shows:
- **Halo attention nearly closes the seam gap.** Letting each chunk's local levels see 4 positions of each neighbour cuts
  the boundary ratio from 1.39 to 1.19 (B2: 1.17); that's 95% of the B0→B2 gap. It also gives the best NLL of any
  run. genPPL improves 22.0 → 21.0 (61% of the B0→B2 gap) but stays behind B2's 19.0. So seams were a large part of Ours's generation
  gap, but not all of it.
- **At M=4, generation improves steadily with G** (22.0 → 20.0 → 19.6 → B2's 19.0), even though NLL is best at G=5
  and G=6 starts losing on fine levels (level 6: 1.86 vs 1.68).
- **At M=6, the ranking flips: the more chunks a model attends across, the worse it extrapolates.** All models were
  trained on 4 chunks. B2 collapses from 19.0 to 33.5, and G=6 goes from 19.6 to 30.7, both with high seed
  variance. G=5 degrades a little (20.0 → 23.8). Ours G=3 (21.8) and **G=3 + halo 4 (21.5) hold their M=4
  quality and become the best models.** Halo keeps its seam benefit at M=6 (1.19).
  Interpretation: global attention over all chunks learns the training length. Block-NCM's few global levels plus
  local + halo levels are close to length-agnostic. This is the strongest argument for the design: **B2 wins only
  at the training length.**
- Zero-shot (the released model, independent chunks) scores 13.7 at M=6. Its entropy is much higher (4.80), and
  each chunk is a fresh fluent opening, which GPT-2 Large scores well. The same effect appeared at M=4 (14.4); it
  isn't comparable to the trained models.

**Plan search** (sample 8 plans per story, keep the one the model itself scores highest; `--n-plans 8`, M=4):

| | genPPL | boundary ratio | latency, s/sample (KV-cached path, batch 8–16) |
|---|---|---|---|
| Ours G=3 | 22.0 → **20.5** | 1.39 → 1.36 | — → 0.079 |
| Ours G=3 + halo 4 | 21.0 → **20.0** | 1.19 → 1.21 | 0.059 → 0.079 |

Plan search gains 1–1.5 genPPL for about 1.3× latency. It mostly improves the content rather than the seams, so it
stacks with halo.

- Running on Kaggle (two T4s): G=5 + halo 4, G=6 + halo 4, G=3 + halo 8, and G=3 + halo 2.

## 2026-11-19: Milestone 4, layer-drop refiner on Kaggle (T4, fp16, one seed)

`kaggle/run_refiner.sh` in full mode: Kaggle notebook version 4, free T4, 2 h 38 min, all stages passed. It used a
30,000-doc training subset and 285 val docs, so its absolute numbers **can't be compared with Milestone 3b**; compare
only within this table.
- **Teacher:** full-depth Ours G=3, lr 3e-5, corruption 0.5, 1,500 steps.
- **Refiners:** 6 of the 12 blocks, layers [0, 1, 2, 9, 10, 11]. That's the best 6/12 pattern in the depth probe,
  which reproduced the local result (symmetric +4.00 vs uniform +4.08 vs prefix +4.81 nats on levels 4–11 before
  training).
- **Refiner training:** each refiner is initialised from the teacher, trained 1,500 steps at lr 1e-4, and distilled
  toward the teacher (KL weight 1.0).
- **Variants:** `s4` replaces the teacher on local levels 4–11 (592 of 641 codes per chunk); `s7` on levels 7–11
  (468 codes).

| | Teacher (12/12) | Refiner s4 (6/12, levels 4–11) | Refiner s7 (6/12, levels 7–11) |
|---|---|---|---|
| NLL, non-padding | **1.6043** | 2.1064 | 1.9037 |
| Δ vs teacher, 95% CI | — | +0.502 [+0.496, +0.508] | +0.299 [+0.294, +0.305] |
| fine levels 4–11 | **1.307** | 1.850 | 1.631 |
| genPPL (256 samples, M=4) | **21.6 ± 0.7** | 41.6 ± 0.3 | 32.0 ± 0.8 |
| boundary / inner ratio | 1.35 | 1.24 | 1.29 |
| entropy | 4.40 | 4.36 | 4.37 |
| latency, s/sample (T4) | 0.083 | **0.053** (1.55× faster) | 0.060 (1.38× faster) |

What this shows:
- **The refiner doesn't pay off at this budget.** It's 1.4–1.55× faster per sample, but genPPL rises by 10–20 and NLL by
  0.3–0.5 nats, with every document worse. Distillation recovered most of the untrained gap: +4.0 nats became +0.5.
  But that falls far short of "near-free".
- The lower seam ratio isn't a gain: the refiner degrades the inner windows too, so the ratio falls while every
  absolute number gets worse.
- **Consequence for the cost model:** the 0.25× and 0.15× refiner scenarios in `serving_cost.py` assumed no quality loss.
  These results don't support that assumption, so treat those scenarios as hypothetical.
- Next options, cheapest first:
  - drop fewer layers (9/12);
  - refine only the finest levels (≥ 9);
  - distil much longer on all 125k docs.
- The fp16 path on T4 was stable throughout: no non-finite losses, and the quantizer ran in fp32.

## 2026-11-19: Milestone 3b, M=4 runs on the corrected recipe (conditioning corruption 0.5, lr 3e-5, one seed)

Same setup as Milestone 3 (1,500 steps × 8 documents, 125,723 train docs), now with corruption. These are the
first valid generation numbers.

**NLL** (`nll_breakdown.py`, all 1,228 val docs, non-padding codes; paired per-document bootstrap, 1,000 resamples):

| | B0 (G=0) | **Ours (G=3)** | B2 (G=11) |
|---|---|---|---|
| total | 1.7486 | **1.7438** | 1.7780 |
| coarse, levels 1–3 | 5.2730 | 5.2823 | **5.1647** |
| fine, levels 4–11 | 1.4616 | **1.4556** | 1.5022 |
| levels 9 / 10 | 1.821 / 1.979 | **1.817 / 1.968** | 1.941 / 2.043 |
| chunk 4 | 1.804 | **1.793** | 1.872 |
| Ours − run, 95% CI | **−0.0049 [−0.0056, −0.0042]**, Ours better on 66% of docs | — | **−0.0342 [−0.0354, −0.0330]**, 96% of docs |

**Generation** (`sample_eval.py`, 2 seeds × 256 samples, M=4):

| | B0 | **Ours** | B2 | Ours: share of B0→B2 gap |
|---|---|---|---|---|
| genPPL | 24.1 ± 0.2 | **22.0 ± 0.3** | 19.0 ± 0.1 | 42% |
| boundary / inner ratio | 1.52 | **1.39** | 1.17 | 38% |
| inner-window genPPL | 35.3 | 35.8 | 35.4 | |
| entropy | 4.50 | 4.39 | 4.36 | |
| latency, s/sample (batch 64, RTX 5050) | 0.045 | 0.090* | 0.051 | |

\* ChunkedNCM path, which recomputes the plan for every local level; the KV-cached path halves training time and is
measured separately.

References: B0 single-chunk (M=1) genPPL **21.6** (released 16.5: the recipe's cost, as in the gate runs); real
TinyStories at 768 tokens (M=6, 145 docs): genPPL 5.3, boundary ratio 1.015.

What this shows:
- **Ours has the best NLL,** significantly better than both baselines (CIs exclude 0). With corruption on, fully global
  attention (B2) has the *worst* NLL: better on the coarse levels, clearly worse on levels 6–11 and on chunk 4.
- **Generation ranks B2 > Ours > B0,** the opposite of NLL for B2. Ours closes about 40% of the B0→B2 gap on genPPL and
  on seams. Teacher-forced NLL (clean inputs) and free generation (the model's own errors) measure different things.
  Within a level, B2's attention across all chunks seems to help most where sampling errors and seams arise.
- So **"Ours ≈ B2" holds on NLL but not on generation;** closing the seam gap is the open problem. That is what the
  queued halo and G=5/6 runs target.
- M=6 evaluations ran out of memory in sampling at batch 64; queued again at batch 16.

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

## 2026-11-19: Simulations of next methods (projections, not measurements)

### Quality: projected NLL for other G (`experiments/project_G.py`)

Levels ≤ G take the fully global run's (B2) per-level NLL; levels > G take the measured Block-NCM local levels
(G=3 plan). Inputs are the clean-recipe M=4 runs, non-padding codes. Level-size weighting reproduces the
measured totals to within 0.002. Per level, global attention wins on levels 1–7 (most on level 4: −0.12, and
level 5: −0.08); local-with-plan wins on levels 8–11 (level 9: +0.027).

| G | projected NLL | vs B2 | share of B0→B2 gap |
|---|---|---|---|
| 3 (measured: 1.3154) | 1.3124 | +0.003 | 84% |
| 4 | 1.3063 | −0.003 | 114% |
| 5 | 1.3011 | −0.008 | 140% |
| **6–7** | **1.2998** | **−0.009** | **147%** |
| 9 | 1.3048 | −0.004 | 122% |
| 11 (= B2) | 1.3091 | 0 | 100% |

Projection: **G = 6–7 would beat fully global attention**, by taking B2's gains on levels 4–7 while keeping
levels 8–11 local. Caveats: built from clean-recipe runs (to be redone on the corrupted retrain); one seed;
B2's fine-level deficit may shrink with longer training; a richer plan could help local levels beyond what is
assumed.

### Cost: method variants (`paper/scripts/method_scenarios.py`; 7B on an H100, as in the serving-cost model)

| method | positions per token | batch-1 latency, 2k tokens | vs AR | $/1M at batch 256 | AR cheaper from batch |
|---|---|---|---|---|---|
| Block-NCM G=3 (current) | 5.00 | 0.369 s | 29× | 0.125 | 34 |
| Block-NCM G=6 | 5.00 | 0.374 s | 29× | 0.127 | 34 |
| G=3 + halo h=8 | 6.00 | 0.443 s | 24× | 0.150 | 28 |
| G=3 + half-depth levels 7–11 | 3.17 | 0.234 s | 46× | 0.079 | 59 |
| G=2 + 6 levels (8, 16, 32, 64, 96, 128) | 2.69 | 0.198 s | 54× | 0.067 | 74 |
| 6 levels + half-depth last 3 | 1.56 | 0.115 s | 93× | 0.039 | 171 |
| B2 fully global | 5.00 | 0.381 s | 28× | 0.129 | 33 |
| B1 chunks in sequence | 5.00 | 0.919 s | 12× | 0.123 | 35 |

These assume each variant keeps quality, which only experiments can show. Raising G is nearly free (+1%).
A halo of h=8 costs about 20% because halo positions are processed as queries; h=4 or halo on only a few levels
would cost less. Reducing positions per token (shallower fine levels, fewer levels) is the only lever that moves
the cost crossover with AR.

More scenarios (same model): a **0.25× refiner on the local levels** gives 1.53 positions per token,
$0.038 per 1M tokens at batch 256 against AR's $0.033, and moves the crossover from batch 34 to 177.

**Batch-1 streaming, 2,048 tokens**

| method | time to first token | time to last token |
|---|---|---|
| AR (GQA) | 5 ms | 10.75 s |
| AR + speculative decoding (0.5B draft, 4 drafted tokens, acceptance 0.8) | 5 ms | 4.11 s |
| B1 chunks in sequence | 57 ms | 0.92 s |
| Block-NCM G=3 | 369 ms | 0.37 s |
| **Super-block streaming** (Block-NCM over 4-chunk blocks, blocks in sequence) | **101 ms** | **0.40 s** |

Super-block streaming removes most of the no-streaming drawback: first output in 0.1 s, only 9% slower in
total. AR with speculative decoding is still about 11× slower than Block-NCM.

### Recipe gap (16.5 → 20.5): second round of hypotheses, gates queued

Checked against the code and the NCM paper (App. F.2.3, F.3, Table 6) before running anything:

| hypothesis | check | status |
|---|---|---|
| Loss averages over codes, not levels (level 11 gets 16× level 1's weight) | True: `forward_loss` is the code average (α=0 in the paper's Eq. 10). But the paper's own α ablation gives genPPL 18.9–19.0 for every α ≤ 1 | Unlikely cause; gate `gate_a1` (α=1) |
| The gate trained on openings of long stories only | True: `train_M1.npy` was cut from the filtered M=4 set (stories > 384 tokens, 6% of TinyStories) | Gate `gate_unf`: 60k unfiltered openings |
| Full-weight drift away from the released model | Plausible; untested | Gate `gate_kl1`: KL(frozen released ‖ model) on local levels, weight 1 |
| The layer-drop refiner failed because it cut levels 4–6 | Already tested: levels 7–11 only, symmetric layers [0,1,2,9,10,11], still +0.30 nats, genPPL 32.0 | Refuted (Milestone 4) |

Also noted: the released recipe uses weight decay 0.1 and an lr floor of 3e-5 over 50k updates at batch 2,048; we
fine-tune with weight decay 0.01 and peak lr 3e-5.

Gates (M=1, 300 steps, batch 32, lr 3e-5, corruption 0.5, same as the 20.7 gate): `gate_base` (repeat, noise
control), `gate_unf`, `gate_a1`, `gate_kl1`, `gate_all` (unfiltered + α=1 + KL 1). Code: `--level-alpha`,
`--kl-anchor` in `experiments/train.py`; checks in `experiments/test_anchor.py` (7/7 pass).

| gate (300 steps, single chunk, seed-42 samples) | openings NLL, step 300 | genPPL | entropy |
|---|---|---|---|
| released, no training | 1.551 | 16.5 | 3.98 |
| `gate_base` (repeat of the 20.7 recipe) | 1.533 | 21.0 | 3.91 |
| `gate_a1` (equal weight per level, α=1) | 1.533 | 20.9 | 3.91 |
| **`gate_kl1` (KL anchor to the released model, weight 1)** | 1.537 | **19.5** | **3.94** |
| **`gate_unf` (unfiltered openings, 60k stories of any length)** | 1.473 * | **17.0** | **3.98** |
| **`gate_all` (unfiltered + α=1 + KL anchor 1)** | 1.484 * | **16.1** | **3.97** |

\* NLL on the unfiltered validation openings, not comparable with the rows above.

- **Level weighting does nothing** (20.9 vs 21.0), as the paper's α ablation predicted.
- **The KL anchor helps: 21.0 → 19.5, and entropy recovers 3.91 → 3.94**, at almost no NLL cost (1.537 vs 1.533).
  It closes about 1.5 of the 4.5 points. With the style shift (about 1.6 points, below), about 1.4 points remain
  unexplained. Repeat-to-repeat noise is about 0.3 (20.7 vs 21.0).
- **Training on unfiltered openings closes most of the gap: 21.0 → 17.0, entropy back to 3.98** (released: 16.5,
  3.98). With the KL anchor added, 16.1, the released model's level within noise. So the "fine-tuning damage" at
  single chunks was mostly the long-story subset, not the recipe: conditioning corruption plus the right data keeps
  the released fluency. Part of this is the style effect on the metric itself (real long-story openings score
  1.10× higher than all openings).
- Consequence for the M=4 models: they must train on long stories (512 tokens of text), so their genPPL carries
  this style offset; real long-story text at M=4 scores 6.1. The KL anchor is the part that transfers; it is being
  tested at M=4 next (anchor 1 on all chunks, and with chunks 2–4 at 0.3×, since the reference model only knows
  openings).
- Queued: weight 3 (`gate_kl3`, `gate_unf_kl3`).

**The long-story subset is a different style of story.** Stories over 384 tokens (our M=4 set, 4.7% of
TinyStories) never open with "Once upon a time" (0.0% of 2,000 sampled, vs 57.8% of all stories); they open with
character names ("Lily and Tom…"). The released model's samples open with "Once upon a time" 70–72% of the time,
fine-tuned B0's 0.2% (1/512). So every fine-tuned model, and the single-chunk gate, learned the long-story style.

How much of the gap that explains (real text, first 128 tokens, 512 stories each, GPT-2 Large, CPU):

| real openings | genPPL | entropy |
|---|---|---|
| all stories | 8.50 | 3.95 |
| long stories (> 384 tokens) | 9.33 | 3.94 |

Long-story openings score 1.10× higher. Scaled the same way, the released model's 16.5 would be about 18.1 on the
long-story style, so the style shift accounts for about 1.6 of the 4.0 points, and about 2.4 points (13%) remain as
recipe cost. (Our all-stories figure, 8.50, is below the paper's 9.67; the selection differs: training split, every
100th story, short stories kept whole.) `gate_unf` tests this directly by training on unfiltered openings.
The earlier rejection below ("length filter, no EOS") tested only end-of-story tokens, not the style of the text.

### Checked and rejected

- **"The 20.5 plateau comes from the length filter (no EOS in openings of long stories)."** Refuted: only 5.0%
  of all stories end within 128 tokens, and the released model's own samples almost never end (≤ 0.4% contain
  EOS, mean length 128), the same as every fine-tuned run.
- **"Train local levels on self-sampled plans (on-policy corruption)."** Not adopted: with residual
  quantization, the fine codes are residuals of the *true* coarse codes. A coherent but different sampled
  plan makes the clean fine targets unpredictable, which teaches the local levels to ignore the plan, the
  opposite of the goal. Uniform corruption is visibly noise and does not have this problem.

### Measured: plan KV cache speeds up training 2.0× (Kaggle T4, `experiments/test_fast.py`)

Forward + backward, G=3, M=4, 2 documents: **ChunkedNCM 2.43 s, FastChunkedNCM 1.21 s**, with identical
results (loss difference ≤ 8e-7 for G ∈ {0, 1, 3, 6, 11}, identical generated codes).

### Built: `blockncm/fast.py` (`FastChunkedNCM`, same weights and state_dict as ChunkedNCM)

- **KV-cached plan**, exact: the prefix and plan keys/values are computed once and reused by every local
  level, in training and generation. Uses a functional copy of the released block (`block_fwd`) that takes
  past K/V explicitly, so it works with gradient checkpointing.
- **Halo attention** (`--halo h`): local levels also see h inputs of each neighbouring chunk (positions
  −h…−1 and l…l+h−1; edges masked).
- **Plan search** (`sample_eval.py --n-plans n`): sample n plans, keep the most likely, run local levels once.
  Likelihood selection can lower diversity, so entropy must be reported alongside genPPL.
- Checks (`experiments/test_fast.py`, CPU): loss equal to ChunkedNCM for G ∈ {0, 1, 3, 6, 11} (max
  difference 1.2e-6); halo with M=1 changes nothing; generation gives identical codes under the same seed
  (G = 0, 3, 11); plan search and halo generation run. **The untrained halo already lowers the released
  model's M=4 loss** (2.456 vs 2.508, G=3).
- Flags: `train.py --fast / --halo`, `sample_eval.py --fast / --halo / --n-plans`; `nll_breakdown.py` now
  scores with FastChunkedNCM (exact at halo 0, and applies each run's halo).

### RTX 5050 time budget for the next runs (from measured s/step)

| run | uncached plan | KV-cached plan |
|---|---|---|
| Ours G=5 / G=6 | 88 / 91 min | 56 / 56 min |
| Ours G=3 + halo h=8 | 97 min | 67 min |
| 2nd seed: B0 + Ours + B2 | 196 min | 170 min |
| long runs, 5000 steps: B0 + best Ours + B2 | 676 min | 568 min |
| **total** | **19.1 h** | **15.3 h** |

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
