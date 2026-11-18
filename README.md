# Block-NCM: long-form generation with short-context Next-Concept Models

Research code for **chunked Block-NCM**, an extension of Next-Concept Modeling
([HazyResearch/ncm](https://github.com/HazyResearch/ncm), arXiv:2610.14162).

A released NCM generates a 128-token text coarse-to-fine in K = 11 parallel levels. Block-NCM generates
an **M × 128-token** document from the same short-context tokenizer and generator:

1. each 128-token chunk is encoded independently by the frozen released VQ-VAE tokenizer;
2. the first **G** levels are generated **globally**: every chunk's tokens at level t attend to each other;
3. the remaining levels are generated **locally**, with chunks folded into the batch dimension; each
   chunk attends only to itself and to the **plan**, which is level G of all chunks.

`G = 0` gives independent chunks, `G = K` a fully global model, and `0 < G < K` Block-NCM. Every variant
takes **K sequential forward passes**, versus M·K for generating chunks one after another.

## Layout

| Path | What it is |
|---|---|
| `blockncm/model.py` | `ChunkedNCM`: one mask rule for training and generation, wrapping the released generator |
| `blockncm/load.py` | Loads the released tokenizer and generator exactly as the NCM repo's evaluation does |
| `experiments/prepare_chunks.py` | Encodes TinyStories documents into M chunks of NCM codes |
| `experiments/train.py` | Fine-tunes a chunked model for given (M, G) |
| `experiments/sample_eval.py` | Generates samples; scores genPPL (GPT-2 Large), entropy, chunk-boundary coherence, latency |
| `experiments/test_equivalence.py` | Checks that `ChunkedNCM` with M=1 reproduces the released model exactly |
| `experiments/story_lengths.py` | Story-length statistics that decide the feasible M |
| `docs/long_form_design.md` | Design, baselines, experiments and milestones |
| `paper/` | LaTeX complexity section (`sections/complexity_and_bounds.tex`) and its constants script |
| `RESULTS.md` | Results log |

## Setup (WSL2 Ubuntu, RTX 50-series)

```bash
python3 -m venv ~/venvs/ncm && source ~/venvs/ncm/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs CUDA 12.8+
git clone https://github.com/HazyResearch/ncm ~/ncm && pip install -r ~/ncm/requirements.txt
# fix HazyResearch/ncm issue #11 (idx_to_zhats aliasing)
sed -i '651s/z_hat.add_(z_q_level)/z_hat = z_hat + z_q_level/' ~/ncm/models/quant.py
```

Download `tinystories/*` from `hazyresearch/ncm-checkpoints` into `~/ncm/checkpoints/tinystories`, and
`tinystories/train_bpe_8192.{bin,_offsets.npy}` from `hazyresearch/ncm-datasets` into `~/ncm/data/tinystories`.

```bash
export NCM_ROOT=~/ncm
python experiments/test_equivalence.py --ckpt ~/ncm/checkpoints/tinystories --data ~/ncm/data/tinystories
python experiments/prepare_chunks.py --M 4 --data ~/ncm/data/tinystories --ckpt ~/ncm/checkpoints/tinystories --out ~/ncm/data/chunked
python experiments/train.py --M 4 --G 3 --name ours_M4_G3
python experiments/sample_eval.py --run ~/ncm/runs/ours_M4_G3 --M 4 --G 3
```

## License and attribution

Builds on HazyResearch/ncm (MIT License); cite Kumbong et al. (2026) when using this code.
