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

**Implemented: `blockncm/adapters.py` (`PlanAdapterNCM`), `train.py --plan-adapter 16`.** Details that
turned out to matter (all checked in `experiments/test_adapters.py`):
- **Plan logit bias,** one per layer and head, initialised negative so local levels start by ignoring the plan.
  At −8 the released model's local-level NLL is 1.801, close to no plan (1.791), against 2.458 with the plan
  and no bias. It is implemented as one extra key dimension: gradients through a float attention mask
  crash the GPU's fused attention kernel.
- **The bias needs its own learning rate.** Adam moves a parameter by about one learning rate per step, so
  at lr 1e-4 a −8 bias would move ~0.15 in 1,500 steps and the plan would never open. Defaults: −6 and
  lr 1e-2.
- **Prefix and chunk embeddings.** The adapters skip the prefix positions, and chunk embeddings go on
  global-level inputs only. Then, with the bias closed, local levels equal the released model (max diff
  4.8e-7 with random adapters), and a single chunk with no plan equals it exactly.
- **Gradients.** Local-level loss alone reaches the global adapters through the cached plan K/V (they are
  never detached); generator weights receive no gradient.
- **Fair B0** under this scheme is the released model itself (no global levels to adapt): zero-shot M=4
  genPPL 14.4, boundary ratio 1.25.

**First test (queued):** LoRA r=16 on global passes only, local frozen, G=3, M=4, corruption 0.5, 1,500 steps.
Measure (a) single-chunk genPPL (should stay 16.5), (b) local-level NLL vs B0 (must beat it, or the local levels
are ignoring the plan), (c) boundary ratio.

## 2b. Layer-drop refiner (first step toward the asymmetric planner/refiner)

`blockncm/refiner.py` (`RefinerNCM`): local levels ≥ `--refiner-start-level` run a **copy** of a subset S of
the blocks (NCM shares weights across levels, so training the refiner must not touch the global passes);
local block j reads the cached plan K/V of global layer S[j]. Training: cross-entropy plus KL to the
**full-depth model on the same inputs and plan** (the teacher is the trained full-depth Block-NCM, not the
released model, which has never seen a plan, so KL to it would penalise plan use). Global passes frozen.
Checks (`experiments/test_refiner.py`): 12/12 layers equal FastChunkedNCM exactly; levels below the start level
unchanged; only the refiner copy gets gradients; plan K/V taken from the kept layer ids.

Depth probe (`experiments/depth_probe.py`, no training) on Ours G=3 (corruption recipe), 256 val docs, local
levels 4–11 code-weighted NLL: full depth 1.708; 6/12 symmetric [0,1,2,9,10,11] 5.591 (+3.88), uniform
[0,2,4,6,8,11] 5.604 (+3.90), prefix 6.426 (+4.72); 3/12 prefix 7.260 (+5.55). Untrained layer drop is far
off; distillation has to recover ~3.9 nats. The refiner runs on Kaggle (free T4, `kaggle/run_refiner.sh`).
The cost model predicts ~1.85× faster generation at 6/12 layers from level 4; the queued run measures it.

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
