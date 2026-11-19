# Plan → verify → refine: design notes for the next stage (after paper 1)

Source: Rishab's proposal (story-level plan, conclusion check, line-level detail with NCM, paragraph breaks
where the representation shifts, "train after 50, exponential schedule"), checked against our measurements and
the NCM architecture. Paper 1 stays scoped to Block-NCM; this is the next branch.

## 1. What was checked

| claim | verdict | evidence |
|---|---|---|
| Chunk boundaries should follow sentences/paragraphs, not every 128 tokens | **adopt (paper 1)** | fixed cuts split sentences ("He ran to" / "the mill"); seams are Block-NCM's weakest point (boundary ratio 1.18 vs 1.07 for B2; real text 1.008) |
| Large story-level model + small line-level model | adopt (paper 2) | matches the strongest cost lever in `method_scenarios.py`: a 0.25× refiner gives 1.53 positions per token and moves the AR cost crossover from batch 34 to 177 |
| Freeze levels 4–10, train levels 0–3 | **not possible as stated** | NCM uses one transformer for all levels; "planner" and "refiner" weights are the same parameters |
| Frozen local levels will use the plan | **no, untrained** | at step 0, Ours' local-level NLL was 2.875 vs 2.471 for B0: the untrained plan hurts |
| Check the "conclusion" on coarse codes | approximate only | coarse codes are spatial (chunk m's blurry summary), not a conclusion slot; decoding them gives rough text |
| Paragraph breaks where weights change | reformulate | weights are frozen at inference; use activation shift or entropy |
| Self-check + retrain every 50, exponential schedule | promising, needs a verifiable task | stories have no ground truth; small updates were shown to damage generation (gate runs) |

## 2. Architecture: plan-only adapters

Block-NCM already runs global levels (1..G) and local levels (G+1..K) as **separate passes**, and the local
levels see the plan only through its cached keys/values (`blockncm/fast.py`). That allows a clean split of
the *shared* weights by pass type:

- **Story planner** = global passes with a LoRA adapter on attention and MLP (active only when t < G).
  Trained with conditioning corruption.
- **Line refiner** = local passes with the **frozen released weights**.
- **Coupling** = local queries (frozen) attend to the plan's keys/values, which are produced by the adapted
  global pass. Optionally a small adapter on the plan's K/V projections only.

Consequence: with no plan in context (single chunk), the local levels compute exactly the released model, so the
16.5 → 20.5 fine-tuning cost disappears **by construction** on 592 of 640 positions per chunk. The open question
is whether frozen local queries can learn to read planner-shaped keys/values. Encouraging sign: the *untrained*
halo (neighbours' level-t inputs) already lowers the released model's M=4 loss (2.456 vs 2.508).

**Fairness:** B0 and B2 must get the same adapter budget (B0: adapter on nothing global, so B0 = released model
plus chunk embeddings; B2: adapter on all passes).

**First test (cheap):** LoRA r=16 on global passes only, local frozen, G=3, M=4, corruption 0.5, 1,500 steps.
Measure (a) single-chunk genPPL (should stay 16.5), (b) local-level NLL vs B0 (must beat it, or the local levels
are ignoring the plan), (c) boundary ratio.

## 3. Sentence-aligned chunks (paper 1 ablation)

`experiments/prepare_chunks.py --align sentence` packs whole sentences into each 128-token chunk and pads the
rest, so seams fall between sentences. A sentence longer than 128 tokens is cut. Compare with fixed cuts on the
boundary ratio, at the cost of padding (fewer real tokens per M chunks; the fill rate is reported).

## 4. Paragraph trigger (post-hoc, zero training)

Insert a break between segments i and i+1 when the shift s_i = 1 − cos(h_i, h_{i+1}) is a local peak and
s_i ≥ μ + κσ (κ ≈ 1), with a minimum of 2–3 lines per paragraph. h_i = mean coarse (level-G) representation of
segment i, or use token-entropy spikes. **Evaluation:** real TinyStories already contain line breaks, so the
trigger can be scored by precision/recall against real "\n" positions before using it on generated text.

## 5. Verify → log → retrain loop

Needs a task with an exact checker. **Testbed: the released Sudoku NCM** (puzzle → solution, verifiable
instantly). XSum (ROUGE) is a softer second option. Not TinyStories.

Schedule (heuristics to test, not established results):
- **Difficulty weight:** an item that passed on attempt r gets weight w = min(e^{α(r−1)}, w_max), α ≈ 0.5,
  w_max = 5, so hard-won passes count more than easy ones.
- **Replay:** each round mixes 30% original data (with corruption, to keep calibration) and 70% verified log,
  sampled across rounds with recency weight γ^{t−j}, γ ≈ 0.7.
- **Growing rounds:** retrain at N_t = 50·ρ^t verified items (ρ ≈ 1.4), or keep 50 with learning rate
  η_t = η_0·0.85^t. Our gate runs showed small updates damage generation, so rounds must not be tiny late on.
- **Final check after refinement:** verifying the plan does not guarantee the refined output is right (a
  detail can still be wrong), so check the full output before logging it.
- Keep failed attempts too: they can train a cheap local verifier (DPO, as in V-STaR) to replace the teacher.

Baseline for the claim "learns in fewer tries": plain supervised fine-tuning on the same number of verified items.

## 6. Order of work

1. Paper 1 queue (running): B0/Ours/B2 with corruption, G=5/6, halo, plan search.
2. Sentence-aligned chunks (paper 1 ablation): re-encode (GPU, ~1.5 h), retrain B0 + Ours, compare boundary ratio.
3. Plan-only adapters, first test (Sec. 2).
4. Paragraph trigger scored against real line breaks (CPU).
5. Verify-log-retrain loop on Sudoku (separate study).
6. Distilled small line refiner (paper 2).
