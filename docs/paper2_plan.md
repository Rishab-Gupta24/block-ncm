# Paper 2 plan: better text from Block-NCM without losing the speed

Status: plan only, 2026-11-25. Nothing here is built. Paper 1 goes to the professor and arXiv first.
Every "measured" number below is from RESULTS.md; anything else is labelled as a guess or a target.

## 1. Goal

Paper 1 showed Block-NCM is fast (16.5× faster than AR at batch 1) but writes about 2× worse text (genPPL 19.2 vs
9.9 at 512 tokens). Paper 2 asks: **how much of that gap can we close while staying much faster than AR?**

| | target | why this number |
|---|---|---|
| Main target | genPPL **14–15** at 512 tokens, temperature 1.0, entropy within 0.05 of today's 4.35 | best-of-8 already reached 15.7 (measured); combining methods should go lower |
| Stretch | 12–13 | needs a method that fixes per-chunk fluency, not just selection (Sec. 4) |
| Speed condition | still **≥ 5× faster than AR at batch 1** (today 16.5×) | leaves room for about 3× more work per story |
| Not a target | "matching AR (9.9)" | base NCM itself is 1.7× behind AR on one chunk (16.4 vs 9.7); closing that needs NCM's original training budget, which does not fit our compute |

The headline we want: **"better text than X in the same wall-clock time"**, shown as a quality-vs-seconds curve.

## 2. Where we stand (measured)

| | genPPL ↓ | notes |
|---|---|---|
| AR TinyStories-33M, 512 tokens (length-matched) | 9.9 | 6.20 s per story at batch 1 |
| Released NCM, one 128-token chunk | 16.4 | 11 passes |
| Block-NCM G=5 + halo 4, 512 tokens, 3-seed mean | 19.2 | 0.376 s at batch 1; seams 1.14 |
| same, M=6 (trained on M=4), 3-seed mean | 20.6 | single runs ranged 20.0–21.0 (local run 25.2) |
| G=6 + halo 4, M=4 (one seed) | 18.7 | but 23.6 at M=6 |
| G=3 + halo 8, M=4 (one seed) | 20.4 | halo 2/4/8 at G=3: 21.4 / 21.0 / 20.4 |
| + plan search, 8 plans (local G=5 run) | 19.4 → 17.1 | no second model; global levels only, batched |
| + best-of-8 with TinyStories-33M scorer (local G=5 run) | 19.55 → 15.7 | 8× generation + scorer; entropy −0.04 |
| temperature 0.8 | × 0.77 for every model | lowers entropy too, so not a fair quality gain |

**Where the gap comes from** (measured pieces, rough):
- **Inherited from NCM:** 16.4 vs 9.7 on single chunks. Not ours to fix cheaply.
- **Long-story style:** GPT-2 Large scores real long-story text ~1.10× higher than all stories. About 1.6 points.
- **Our fine-tuning recipe:** ~2.4 points. Single-chunk tests: training on stories of all lengths gives 17.0 (vs
  21.0 on long stories only); adding the KL anchor gives 16.1, the released level. But the KL anchor **hurt** the
  M=4 planned models, so it does not transfer as is.
- **Chunk 1 is already ~21** for every model (per-chunk test), well above the released 16.5. So the biggest
  Block-NCM-specific loss is per-chunk fluency, not the seams.
- **Seams are small:** 1.14 against a floor of 1.04 for AR text, which has no seams at all. Even perfect seams would
  move genPPL by well under 1 point.

## 3. Corrections to the Gemini plan (check before reusing any of it)

| Gemini said | Actually |
|---|---|
| Baseline 19.1 (4 chunks), 20.0 (6 chunks) | Those are one seed. Three-seed means: 19.2 and 20.6 |
| "halo 8 lost to halo 4", so draft noise hurts | Wrong: halo 8 was better (20.4 vs 21.0 at G=3). The "draft-noise lever" rests on a false premise |
| 4-chunk story takes 18.7 ms; model ~50M params | Measured 376 ms at batch 1 (~34 ms per pass incl. decoding); the generator is ~85M |
| 5 extra passes cost "nearly the same time" | At batch 1 each pass is a fixed cost: 16 passes ≈ 1.45× time (~0.55 s), still ~11× faster than AR |
| Real TinyStories genPPL 8.53 | Our reference: 9.67 (128 tokens, NCM paper), 6.1 at 512 tokens |
| Projected 18.02, 13.8–15.2, 12–14 with equations | The equations use invented constants (e.g. "seam fix 68%", "+0.045 nats"). They are guesses, not predictions |
| Evaluate at 8 and 12 chunks with Graphify | TinyStories has 415 stories ≥ 1,024 tokens and 0 near 1,536. Nothing to train or check against at 8–12 chunks |
| Use a KL anchor to a teacher | Our KL anchor hurt M=4 planned models (NLL and genPPL); rejected in Paper 1 |
| Block-NCM has "planner-refiner distillation" and "start/context/conclusion" blocks | Not in Block-NCM. The layer-drop refiner was tried and failed (genPPL 32–42) |
| Nothing is public yet | The GitHub repo is public, and Paper 1 has a full draft |
| Cites HaloBlock, dLoop, FLARE, RouterWise, CoMem (2026) | Not verified. Find and read each one before citing; some may not exist |

The useful ideas from it: the 11 + 1 + 4 revision idea (yours), an entity-consistency metric, zero-initialised
gates with a frozen base, training on the model's own drafts, and reporting entropy next to genPPL.

## 4. Candidate methods, ranked by evidence and cost

| # | method | evidence so far | training? | extra time per story | expected effect (guess) |
|---|---|---|---|---|---|
| A | **Test-time search:** plan search + best-of-K together, and MBR using NCM's own likelihood (no outside scorer) | plan search −2.3, best-of-8 −3.6 (measured, separately) | none | K× (batchable) | 15–16 |
| B | **Draft-aware revision (your 11 + 1 + 4)** | none yet: new | yes | +5 passes ≈ +45% | unknown; this is the novel part |
| C | **Recipe fix:** mix stories of all lengths into M=4 training, tune regularisation | single-chunk gates: 21.0 → 17.0 with all-length data | yes | none | 1–3 points |
| D | Cheap ablations: G=5 + halo 8, G=6 + halo 8 | halo 8 helped at G=3; G=6 best at M=4 | yes, short | small | 0.5–1 point |
| E | Entity-consistency metric (the useful part of "Graphify") | none | none | none | a new measurement, not a gain |
| F | Token-space denoiser distilled from AR | none | large | +4 passes of a 2nd model | park it: gains would be credited to the AR teacher, and it is a different system |
| G | Length router "bot" | — | — | — | product engineering, not research; build it in the code phase |

### B in detail: what "11 + 1 + 4" means in NCM terms

Today the plan (levels 1–G of all chunks) is made **before** any text exists, so it cannot react to what the
fine levels actually wrote. Your idea adds a feedback step:

1. **11 passes:** generate the story as now (G global levels + local levels + halo).
2. **+1 pass, "read back":** summarise each finished chunk into a few vectors (2–4 per chunk) from its final
   reconstruction. Unlike the plan, this summary sees the actual text.
3. **+4 passes, "revise":** re-generate the last fine levels (for example levels 8–11) of every chunk, in parallel,
   attending to the read-back summaries of all chunks. Each chunk can now fix contradictions and seams with
   knowledge of what its neighbours really said.

What makes it new: Block-NCM's plan is a forecast; the read-back is a record. The paper's question is
"does letting chunks see what their neighbours actually wrote fix the fluency the plan cannot?"

Design rules taken from our own failures:
- **Freeze the trained Block-NCM** and add the read-back and revision as new modules with a **zero-initialised
  gate**, so step 0 equals today's model. This is the plan-adapter trick that worked in Paper 1 (closed logit
  bias with its own learning rate).
- **Train on the model's own drafts**, not on real text: the revision has to fix its own mistakes.
- **Keep conditioning corruption on.** Without it, fine-tuning destroyed generation (16.5 → 47.7).
- **Use full depth for the 4 revision passes first.** The half-depth refiner failed (genPPL 32–42); try lighter
  versions only after full depth works.

Controls the paper needs (each answers a reviewer's "is it just…?"):

| run | passes | answers |
|---|---|---|
| Block-NCM G=5 + halo 4 | 11 | baseline |
| 11 + 4, revision without read-back | 15 | is it just more passes? |
| 11 + 1 + 4 | 16 | the method |
| 8 + 1 + 4 | 13 | does it still win at about the same time? |
| best-of-2 Block-NCM | 22 (2 × 11) | does it beat spending the same time on selection? |
| 11 + 1 + 4 + plan search | 16 + plan cost | do the methods add up? |

## 5. Phases, with go / no-go gates

**Phase 0 (now): Paper 1.** Send to the professor, put it on arXiv, ask about the department GPU. No Paper 2 code.

**Phase 1 (after arXiv; ~3 days; mostly CPU): fix the measuring stick.**
- One eval script for every run: genPPL (GPT-2 Large, T=1.0), unigram entropy, distinct-4, seam ratio, latency at
  batch 1 / 8 / 32 (with `expandable_segments`), 3 seeds × 256 samples.
- Build the entity-consistency check (E): extract names and objects per chunk with a simple rule-based tagger
  (TinyStories names are simple); count contradictions across chunks. Validate it on real stories first (real
  text should score near zero).
- Gate: the metric separates real text from B0 (which has no plan). If not, drop it.

**Phase 2 (~1 week, no training): test-time search (A).**
- Grid: plan search n ∈ {1, 4, 8} × best-of-K ∈ {1, 2, 4, 8}, plus MBR with NCM's own likelihood.
- Batch the scorer properly and time everything at batch 1; plot genPPL against seconds per story.
- Use all three seeds' checkpoints (the local G=5 run is the weakest one).
- Gate: if the best point reaches ≤ 16 at ≤ 1/5 of AR's time, this alone is a solid workshop result. Continue
  either way.

**Phase 3 (~2 weeks, training): draft-aware revision (B).**
- Build read-back + revision modules on the frozen G=5 + halo 4 model; unit-test that step 0 equals the base model.
- Train on Kaggle T4s (or the department V100), same 1,500-step budget as Paper 1 first.
- Run the six controls in Sec. 4.
- Gate after the first run: 11 + 1 + 4 must beat 11 + 4 (no read-back) by more than the seed spread (~0.3). If it
  doesn't, the read-back adds nothing: write it up as a negative result and lean on A + C.

**Phase 4 (~1 week): recipe fix (C) and cheap ablations (D).**
- Mix all-length stories into M=4 training (the single-chunk gate cut 21.0 → 17.0); retrain G=5 + halo 4 and
  re-run the best method on top.
- G=5 + halo 8 and G=6 + halo 8, one seed each, then seeds for the winner.

**Phase 5: write-up.** Quality-vs-seconds plot as the main figure; per-chunk table; entity-consistency table;
honest limits. Then decide the venue with the professor.

## 6. Do's and don'ts

Do:
- Report entropy and distinct-4 next to every genPPL, always at temperature 1.0.
- Count the extra cost: every method gets a seconds-per-story number at batch 1 and batch 32.
- Use three seeds for every claim in the abstract; say "one seed" wherever it is one seed.
- Freeze what works and add new parts behind zero-initialised gates.
- Write down the prediction before each run (as in Paper 1's gates), then compare.

Don't:
- Don't trust projected numbers (ours or Gemini's) until a run measures them.
- Don't cite a paper you haven't opened.
- Don't plan experiments at 8–12 chunks on TinyStories: the data isn't there. M=6 is our longest test.
- Don't lower the temperature to get a better genPPL.
- Don't build the router bot or a token-space AR denoiser for this paper.
- Don't start Phase 1 until Paper 1 has been sent to the professor.

## 7. Open questions

- Compute: is the department's GPU available to students? It decides whether Phase 3 runs at 1,500 steps or more.
- Longer data: TinyStories stops at ~6 chunks. HazyResearch lists an OpenWebText model, but as of 2026-11-19 the
  checkpoint was not released (only a TODO file). Check again before Phase 3.
- Venue: one quality paper (A + B + C), or A as a short workshop paper first and B as the main paper?
