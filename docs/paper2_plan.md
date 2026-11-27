# Paper 2 plan: better text from Block-NCM without losing the speed

Status: plan only, updated 2026-11-27 (after the error analysis, deletion test and chunk-embedding test). Nothing here is built. Paper 1 goes to the professor and arXiv first.
**Measured** numbers come from RESULTS.md. Everything marked *est.* is an educated guess, not a prediction: past
estimates in this project were often off once measured. Treat the estimates as "worth trying", never as promises.

## 1. Goal

Paper 1 showed Block-NCM is fast (16.5× faster than AR at batch 1) but writes about 2× worse text (genPPL 19.2 vs
9.9 at 512 tokens). Paper 2 asks: **how much of that gap can we close while staying much faster than AR?**

| | target | why |
|---|---|---|
| Main target | genPPL **14–16** at 512 tokens, temperature 1.0, entropy within 0.05 of today's 4.35 | only reachable by combining training fixes, the new method and test-time search (Sec. 5) |
| Speed condition | still **≥ 5× faster than AR at batch 1** (today 16.5×), i.e. under about 1.2 s per story | leaves room for about 3× more time per story |
| Consistency target | clearly fewer character-name switches across chunks than today | the new method's main benefit; genPPL barely sees it |
| Not a target | matching AR (9.9) | the NCM generator itself is 1.7× behind AR on single chunks (16.4 vs 9.7) |

Headline we want: **"better text than X in the same wall-clock time"**, as a quality-vs-seconds plot.

## 2. Every major model so far (measured, 512 tokens, batch 1, RTX 5050)

One pass costs about **34 ms** at batch 1 (0.376 s / 11 passes, decoding included), almost regardless of how much
work the pass does, because an 89M-parameter model does not fill the GPU. So **latency ≈ number of passes × 34 ms**.

| model | passes | genPPL, 4 chunks | seams | genPPL, 6 chunks | time per story | vs AR |
|---|---|---|---|---|---|---|
| real long stories | — | 6.1 | 1.01 | — | — | — |
| AR, TinyStories-33M | ~512 | **9.9** | 1.04 | (not run) | 6.20 s | 1× |
| released NCM, 4 independent chunks (not one story) | 11 | 14.4 | 1.25 | — | ~0.38 s | ~16× |
| B1, released NCM chunk after chunk | 44 | (never trained) | — | — | 1.62 s | 3.8× |
| B0, no sharing (G=0) | 11 | 24.2 | 1.49 | 23.8 | ~0.38 s | ~16× |
| Block-NCM G=3 + halo 4 | 11 | 21.1 | 1.20 | **21.2** | ~0.38 s | ~16× |
| **Block-NCM G=5 + halo 4** (Paper 1 main) | 11 | 19.2 | 1.14 | 20.6 † | **0.376 s** | **16.5×** |
| Block-NCM G=6 + halo 4 (one seed) | 11 | 18.7 | 1.12 | 23.6 | ~0.38 s | ~16× |
| B2, share everything (G=11) | 11 | **18.7** | 1.12 | 31.3 | ~0.37 s *est.* | ~16× |
| G=5 + halo 4 + plan search, 8 plans (local run) | 11 + plan re-runs | 17.1 (from 19.4) | 1.16 | — | ~0.5 s *est.* (1.3× measured for G=3) | ~12× |
| G=5 + halo 4 + best-of-8, AR scorer (local run) | 11, batch of 8 | 15.7 (from 19.55) | — | — | ~0.65 s *est.* | ~9.5× |
| **New (2026-11-27):** B2, chunks 5–6 given chunk 4's embedding | 11 | — | 1.17 | **26.7** (was 33.5) | ~0.37 s | ~16× |
| **New:** G=5 + halo 4 (local run), chunks 5–6 given chunk 4's embedding | 11 | — | 1.20 | **22.7** (was 25.2) | 0.38 s | 16.5× |
| **New:** G=5 + halo 4 (local run), doubled words deleted by a rule | 11 | 18.8 (from 19.4) | — | 24.0 (from 25.2) | 0.38 s | 16.5× |

3-seed means except where noted. † Kaggle seeds; a fourth, local run reached 25.2.

What this table says:
- The gap to AR is mostly **fluency inside each chunk** (chunk 1 scores about 21 for every model vs 16.4 for the
  released NCM), not seams (1.14 vs a floor of 1.04) and not sharing (B2, which shares everything, is only 0.5 better).
- Selection already works: the model *can* write 15.7-quality stories; it just doesn't always.
- About half of the collapse at 6 chunks comes from **chunk embeddings 5–6 never being trained** (they stay zero).
  Filling them in, without any training, moved B2 33.5 → 26.7 and G=5 + halo 4 25.2 → 22.7; chunk 1 alone improves
  (B2 35.3 → 28.1, G=5 28.6 → 24.9), so this is not stories ending earlier. The rest is probably unseen positions.

### What the mistakes are (measured 2026-11-27, `experiments/error_types.py`, `experiments/dedup_test.py`)

| mistake (G=5 + halo 4, 4 chunks) | how often | real stories | what it costs |
|---|---|---|---|
| doubled word ("and and") | 8.2 per 1,000 words, 93% of stories | 0.13 | deleting all of them: genPPL 19.38 → 18.79 (−0.6; a lower bound, since a correct replacement would gain more) |
| missing or wrong word ("It white and white") | 20 of 20 stories read by hand | — | not measurable by rule; likely most of the gap |
| meaning contradictions ("She sees the flames. She is happy") | ~17 of 20 | — | — |
| who-is-who: wrong speaker, talking to oneself | ~15 of 20; self-address 25% of stories for **every** model, even B2 | 2% | — |
| character switches or appears from nowhere | ~13 of 20; one-off names B0 50% → G=5 24% → B2 16% | 1% | — |

- Mistakes are spread through each chunk, not bunched at seams (seams: 1.2–1.6× more, except sentences cut at a
  chunk end). The released NCM already writes 4.9 doubled words per 1,000 on single chunks, so the stutter comes
  from parallel decoding, not from Block-NCM's sharing.
- **Doubled words are the most visible mistake but a small part of genPPL.** Deleting them gains 0.6; replacing
  them correctly would gain somewhat more (deletion turns an overwritten word into a missing one). The main quality
  target is still missing or wrong words and meaning inside chunks.
- **Sharing fixes characters appearing from nowhere, but not self-addressed speech** (same rate in B0 and B2). That
  mistake looks local, inside one chunk.

## 3. What to add: ranked, with estimates

Estimates are changes in genPPL at 4 chunks relative to Paper 1's 19.2 (negative = better), plus the effect at 6
chunks and the cost in time. "Evidence" says what the estimate rests on.

### 3a. Add: training fixes (no extra time at all)

| # | addition | evidence | genPPL, 4 chunks | 6 chunks | time |
|---|---|---|---|---|---|
| T1 | **Mixed-length training** (stories of 2–6 chunks), so chunk embeddings 5–6 get trained | filling the embeddings alone, no training, cut the collapse by about half for both B2 (33.5 → 26.7) and G=5 (25.2 → 22.7) | 0 to −0.5 *est.* | −2 to −3 *est.* | +0 |
| T2 | **Recipe fix:** mix stories of all lengths into training | single-chunk test: 21.0 → 17.0 | −1.5 to −3 *est.* | similar | +0 |
| T3 | Halo 8 instead of 4 for G=5 | at G=3: 21.0 → 20.4 | −0.3 to −0.6 *est.* | small | about +10% compute, ~+0 latency |
| T4 | Position fix for cross-chunk levels (position interpolation) | addresses the other half of the collapse; inference-only test first | ~0 | −1 to −3 *est.* for B2 / G=5 | +0 |

### 3b. Add: test-time search (no training)

| # | addition | evidence | genPPL, 4 chunks | time |
|---|---|---|---|---|
| S1 | **Plan search + best-of-K together**, and MBR using NCM's own likelihood (no outside judge in the loop) | measured separately: −2.3 and −3.6 | −3 to −4.5 *est.* combined | 1.3–1.7× (0.5–0.65 s) |

Search is also **the bar every new method must clear**: at batch 1 the GPU is mostly idle, so 8 parallel samples
cost only about 1.5× the time. Any method that adds sequential passes must beat best-of-K at equal seconds.

### 3c. Add: the new method, "plan-first graph Block-NCM" (your sketches, adapted to the trained model)

Built on top of the trained Block-NCM (fine-tuning only; no from-scratch training). Three parts, tested one at a time:

| # | part | what it does | genPPL, 4 chunks | 6 chunks | extra passes | time |
|---|---|---|---|---|---|---|
| N1 | **Vertical sharing via graph slots** | each chunk exposes a few fixed "character / event card" vectors; any chunk can read any card, not just neighbours (halo = sideways, this = any-to-any) | 0 to −1 *est.* | −0.5 to −1.5 *est.* | **0** | ~0.39 s |
| N2 | **Plan slots** | passes 1–2 write explicit plan vectors: characters, number of chunks (story length), story gist; all chunks read them | −0.5 to −2 *est.* | −1 to −2 *est.* (length-aware ending) | +2 | ~0.45 s |
| N3 | **Correction step** (after levels 3 and 5, or at the end) | reads the whole draft of all chunks and fixes it; the stutter cannot be fixed inside the pass that makes it | −1 to −3 *est.* | −1 to −2 *est.* | +2 | ~0.51 s |
| | **N1 + N2 + N3 together** | overlapping gains, so not additive | **−1.5 to −3.5** *est.* (≈ 15.7–17.7) | | **+4 (15 passes)** | **~0.51 s** |

**Expected main benefit: character consistency** (fewer new names mid-story, like "Sara" in Paper 1's Figure 3).
genPPL may move only a little even if consistency improves a lot, so the consistency metric (Phase 1) is required.

Design rules (from Paper 1's evidence):
- **Keep cross-chunk sharing small and fixed.** B2 shared everything and collapsed beyond the trained length; G=3,
  which shares a few short levels, held (+6%). Each chunk gets a fixed number of slots, never "all positions".
- **Share mainly at coarse and middle levels.** Full sharing at the finest levels hurt B2's NLL.
- **The graph is learned vectors, not parsed text.** Codes are not words until the last level, so a text parser
  cannot run mid-generation. Parse text only at the end, to *check* the graph against the story.
- **Train only what is new at first, behind zero-initialised gates,** so step 0 equals Paper 1's model. But note
  that freezing the whole generator failed before (plan-only adapters: 28.5); allow full fine-tuning if needed.
- **Train on the model's own drafts** for the correction step, so it learns to fix its own mistakes.
- **Keep conditioning corruption on** (without it, fine-tuning broke generation: 16.5 → 47.7).

How to train the correction step (N3 / M1):
- **The correction step works in code space, fixed length.** It is the NCM re-predicting its fine-level codes, not a
  separate word editor. No insert/delete word editor (Levenshtein-Transformer style) in Paper 2: it would be a second
  model on words, more work, and it brings back the credit question.
- **Coarse-conditioned rollouts (main data source).** Keep a real story's coarse codes (levels 1–G) and let the
  trained model generate only the fine levels. The draft follows the real story but contains the model's real
  mistakes. In code space the target is simply the real story's fine codes at the same positions, so it lines up by
  construction. A free-running story cannot be paired with a real one, because the plots differ.
- **Rollout alignment check (run before choosing the training loss; inference only, no training).** Decode the
  rollouts and compare them with the real stories in two ways: (a) exact position-by-position token match, and
  (b) edit-distance (Levenshtein) match. If (b) is much higher than (a), the decoder shifts words by a position or
  two. That decides two things:
  - Supervise **fine codes**, not token positions. With shifts, a position-by-position token loss would mark every
    word after a shift as an error. If tokens must be supervised anyway, use only spans that line up.
  - Whether re-predicting fine levels can **move words**. If it can, the fixed-length step can still repair a
    missing word, with no insert operation needed.
- **Fake errors on clean stories, in place only:** doubled words, a word overwritten by a neighbour's copy or a vague
  filler (this is how a fixed-slot parallel decoder "drops" a word), and swapped character names in later chunks
  (restore them from chunks 1–2). Never delete a word and shift the rest left: shifted text encodes to shifted codes
  and breaks the targets. Revisit shifted errors only if the alignment check shows the decoder absorbs shifts. Plus
  some clean stories left unchanged, so it learns not to over-edit.
- **Sparse edits:** an "edit here?" output per position plus a replacement, applied only where flagged. Rule:
  never replace two neighbours in the same pass, and a replacement must differ from both neighbours (otherwise it
  can create a new doubled word).
- **What it must beat:** the free deletion rule (−0.6 genPPL, 0 passes; a lower bound on fixing doubled words), and
  best-of-K at equal seconds.
- No outside AI model to fix drafts: gains would partly be its work (same credit problem as an AR fixer).

For character slots (N1), add a **slot-swap loss**: swap two characters' slots during training (labels from the
name tagger) so the model has to read the slots to get names right, instead of guessing from nearby words.
Build N1 only if the correction step leaves name mix-ups high at 6 chunks. Measure self-addressed speech
separately; slots may not fix it.

Training labels we can get cheaply: characters (rule-based name tagger; TinyStories names are simple), number of
chunks (known from length), story gist (check whether the TinyStories "Instruct" release has per-story summaries or
required words; not verified yet).

### 3d. Maybe: build only as a comparison

| # | addition | genPPL, 4 chunks | extra passes | time | why only "maybe" |
|---|---|---|---|---|---|
| M1 | **11 + 1 + 4** (read back finished chunks, then revise fine levels) | −1 to −3 *est.* | +5 (16) | ~0.55 s | overlaps with N3 (both correct mistakes); worth one comparison run: "fix early (N3) vs fix at the end (M1)" |
| M2 | Split the finest level into two half-passes | −0.5 to −1.5 *est.* | +1 (12) | ~0.41 s | backup if correction shows no room to help |

### 3e. Measurement additions (no quality gain, but needed)

- **Entity-consistency metric** (names and objects per chunk; count contradictions). Validate on real text first.
  A first rule-based version exists (`experiments/error_types.py`: doubled words, unclosed quotes, self-addressed
  speech, one-off names, wrong pronouns), split into inside-chunk and near-seam. It does not catch characters who
  switch names many times, or missing words; hand-check 30–50 stories against it.
- **Second judge model** plus a **blind side-by-side** on about 100 story pairs, so gains are not one AR model
  pleasing another.
- **Plot-duplicate rate** (already built: `experiments/plot_dup.py`): same plot ≥ 0.5 similarity is 1.2% for real
  stories and 0.4% for every model, so search must not raise it (picking "safe" stories would).

## 4. What not to add, and why

| idea | why not (evidence) |
|---|---|
| Odd/even power or precision switching | no quality gain; at 89M parameters and batch 1 each pass is overhead-bound, so cheaper math saves little time; quantization speeds up AR just as much. Later, reframed as **per-level precision** in an efficiency paper |
| Correction after **every** level (22 passes) | ~0.75 s (2× today) for a gain bounded like N3's; best-of-K is cheaper at equal time |
| New NCM from scratch with ~6 levels (the 2 + 3K sketch as drawn) | needs pretraining (original recipe: batch 2,048 × 50k updates), loses the released checkpoint, breaks the comparison. Later, with the department GPU |
| KL anchor to the released model | measured: hurt the planned models (G=5 + halo: 19.1 → 25.7) |
| Half-depth / layer-drop refiner | measured: 21.6 → 41.6 |
| Frozen generator + plan-only adapters | measured: 28.5 vs 22.0 |
| Sentence-aligned chunks | measured: 22.0 → 24.3, seams worse |
| More global levels (G ≥ 6) | measured: 23.6–30.7 at 6 chunks |
| Token-space denoiser distilled from AR | gains would be credited to the AR teacher; adds a second model |
| Length router "bot" | product engineering, not research; build in the code phase |
| Lower temperature to report better genPPL | lowers entropy too; not a fair quality gain |
| Experiments at 8–12 chunks on TinyStories | data doesn't exist (415 stories ≥ 1,024 tokens, none near 1,536) |
| Parsing a Graphify tree from mid-generation codes | codes aren't text until the last level |
| A paper aimed mainly at doubled words | measured: deleting every doubled word gains only 0.6 genPPL |
| "Develop two chunks, refine them together" to stop doubled words | Block-NCM already generates chunks together (plan + halo); most doubled words are inside chunks, so a seam fixer removes at most ~7% of them |
| Inference-only "re-predict one word from its finished neighbour" | NCM predicts all positions of a level in one pass from the coarser level; no step sees a finished neighbour without training |
| Code-Block-NCM (Python) inside Paper 2 | needs a new codec and pretraining from scratch; HumanEval/MBPP would score near 0% at this size. Later, as Paper 3 |

## 5. The recommended package, step by step, against every model

| configuration | passes | genPPL, 4 chunks | 6 chunks | time per story | vs AR | vs B1 (1.62 s) |
|---|---|---|---|---|---|---|
| AR (reference) | ~512 | 9.9 | — | 6.20 s | 1× | 3.8× slower |
| B2 (Paper 1 baseline) | 11 | 18.7 | 31.3 | ~0.37 s | ~16× | 4.4× faster |
| **Paper 1: G=5 + halo 4** | 11 | 19.2 | 20.6 | 0.38 s | 16.5× | 4.3× faster |
| + training fixes (T1, T2, T3) | 11 | ~17–18 *est.* | ~18.5–19.5 *est.* | 0.38 s | 16.5× | 4.3× faster |
| + plan-first graph (N1–N3) | 15 | ~15.5–17 *est.* | ~17–18.5 *est.* | ~0.51 s | ~12× | ~3.2× faster |
| + search (best-of-4, batched) | 15, batch 4 | **~14–16** *est.* | ~16–17.5 *est.* | ~0.65–0.7 s | **~9×** | ~2.4× faster |

Even the full package would stay about 9× faster than AR and well inside the ≥ 5× condition. It would also clearly
beat every Paper 1 model at 6 chunks if the length-aware parts work.

If only one thing could be built: **training fixes + search** (no new architecture) is the safest path to ~15–16,
and gives a solid workshop paper. The plan-first graph method is what makes Paper 2 *novel*.

## 6. Phases, with go / no-go gates

**Phase 0 (now):** send Paper 1 to the professor (Paper 1 text stays as is). Done: chunk-embedding test, error
analysis, deletion test. No Paper 2 code.

**Phase 1 (~3 days, mostly CPU): measuring stick.** One eval script for every run (genPPL with T=1.0, entropy,
distinct-4, seams, plot-duplicate rate, latency at batch 1/8/32); the entity-consistency metric; a second judge.
Gate: the consistency metric separates real text from B0. If not, fix it before anything else.

**Phase 2 (~1 week, no training): search + cheap inference tests.** Plan search × best-of-K grid and MBR, on all
three seeds; the position-interpolation test (T4) at 6 chunks; the story checker with chunk-only repair (find a
broken chunk by rule, regenerate only its local levels, keep the plan and other chunks). Gate: none (always useful);
this sets the bar. Also the rollout alignment check (Sec. 3c): exact vs edit-distance match of coarse-conditioned
rollouts against the real stories, which fixes the correction step's training loss before Phase 4.

**Phase 3 (~1 week, Kaggle): new base model.** Train G=5 + halo (4 and 8) with mixed lengths and all-length data,
3 seeds. Gate: better than 19.2 at 4 chunks by more than the seed spread (~0.3), and better at 6 chunks.

**Phase 4 (~2–3 weeks, Kaggle or department GPU): one part at a time.** Order changed by the error analysis:
the correction step comes first, because inside-chunk mistakes are the biggest problem.
1. + correction step, trained on coarse-conditioned rollouts and fake errors. Gate: beats the deletion rule
   (−0.6) and best-of-K at equal seconds per story; doubled words and missing words go down.
2. + plan slots. Gate: better ending on time at 6 chunks, or genPPL gain > 0.3.
3. + character slots with the slot-swap loss, only if name mix-ups stay high. Gate: one-off names and wrong
   pronouns go down, genPPL not worse.
4. One comparison: fix early (after levels 3 and 5) vs fix at the end (11 + 1 + 4).
If a part fails its gate, report it as a negative result and keep the parts that passed.

**Phase 5: write-up.** Main figure: genPPL and consistency vs seconds per story, for every model in Sec. 2 and 5.

## 7. Do's and don'ts

Do: report entropy and seconds per story next to every genPPL; three seeds for every abstract claim; write the
prediction before each run; compare every method against best-of-K at equal time; add new parts behind
zero-initialised gates.

Don't: trust projected numbers (ours or outside ones) until measured; cite papers you haven't opened; plan
experiments beyond 6 chunks on TinyStories; lower the temperature for a better genPPL; start Phase 1 before Paper 1
has been sent.

## 8. Corrections to outside suggestions (check before reusing any of it)

| it said | actually |
|---|---|
| baseline 19.1 / 20.0 | one seed; three-seed means are 19.2 / 20.6 |
| "halo 8 lost to halo 4" | halo 8 was better (20.4 vs 21.0 at G=3) |
| a story takes 18.7 ms, or a pass 1.5–2 ms | measured 376 ms per story, about 34 ms per pass |
| 14 passes cost the same as 11 ("iso-latency") | at batch 1 each pass costs about the same: 14 passes ≈ 0.48 s |
| NCM has a "~15.5 VQ-codec ceiling" | the tokenizer reconstructs real text 99.97% correctly; the limit is the generator |
| projected genPPL 12.5–14.8, energy −38.8% | invented constants (α, β, "equivalent passes"); nothing measured |
| half-depth refinement passes | measured to fail (21.6 → 41.6) |
| KL anchor to a teacher | hurt the planned models; rejected in Paper 1 |
| evaluate at 8–12 chunks | no data that long in TinyStories |
| test power switching on the lab's Jetson Nano | unverified that the lab has one; the original Jetson Nano lacks bf16/fp8 |
| cites HaloBlock, dLoop, FLARE, RouterWise, CoMem, DVD-Quant | none verified; open each before citing |
| "stutters ~4.9 per 1k" for G=5 + halo 4 | 8.2 for G=5 + halo 4; 4.9 is the released NCM on single chunks |
| fixing stutters drives the seam ratio to 1.0 | doubled words are mostly inside chunks (22% near seams vs 15% of text) |
| sparse edits "mathematically cannot create new stutters" | only if the replacement must differ from its neighbours; add that rule |
| who-is-who errors come from "names blurred in compressed codes" | a guess; self-addressed speech is equally common with full sharing (B2) |

## 9. Open questions

- Compute: is the department GPU available to students? Phase 4 is much easier with it.
- Does the TinyStories "Instruct" release include per-story summaries or required words (free plan labels)?
- Venue: one paper (training fixes + method + search), or search + training fixes as a short workshop paper first?

## 10. Idea credits (for the AI-use statement)

Yours: Block-NCM itself; 11 + 1 + 4; odd/even passes; plan-first passes with a graph, characters and paragraph
count; connector slots; vertical sharing; a story checker ("code story"); refining chunks together. Suggested by
Claude: the split-level backup, the fixed-size-slot rule for vertical sharing, mixed-length training and the
position fix (from the chunk-embedding test), chunk-only repair, the error analysis and deletion test, and
reframing odd/even as per-level precision. From outside AI suggestions (checked): coarse-conditioned rollouts,
sparse edits, fake-error training, the slot-swap loss.
