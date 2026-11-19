#!/usr/bin/env bash
# Layer-drop refiner experiment, self-contained for a Kaggle GPU notebook (T4/P100; Internet ON).
#
# Notebook cell:
#   !git clone -q https://github.com/Rishab-Gupta24/block-ncm /kaggle/working/block-ncm
#   !bash /kaggle/working/block-ncm/kaggle/run_refiner.sh
#
# Steps: NCM repo (pinned + issue #11 patch) -> released TinyStories checkpoint and data from Hugging Face ->
# encode M=4 chunks (MAX_DOCS documents; 1,500 steps x 8 docs only use ~12k) -> GPU checks -> full-depth
# Block-NCM teacher (G=3, corruption 0.5, KV-cached plan) -> depth probe -> two refiners (best 6/12 layers,
# from level 4 and from level 7) -> NLL breakdown -> generation quality + measured latency, refiner vs teacher.
# Results: /kaggle/working/results (progress.txt has timestamps). fp16 autocast on GPUs without bf16.
set -u
W=/kaggle/working
REPO=$W/block-ncm
OUT=$W/results
MAX_DOCS=${MAX_DOCS:-30000}
STEPS=${STEPS:-1500}
export NCM_ROOT=$W/ncm PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT
log() { echo "=== $* $(date -u +%H:%M:%S)" | tee -a $OUT/progress.txt; }
quiet() { grep -v "parameter count" || true; }

nvidia-smi | tee $OUT/gpu.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" | tee -a $OUT/gpu.txt

log "setup"
[ -d $NCM_ROOT ] || git clone -q https://github.com/HazyResearch/ncm $NCM_ROOT
(cd $NCM_ROOT && git checkout -q c702742)
sed -i '651s/z_hat.add_(z_q_level)/z_hat = z_hat + z_q_level/' $NCM_ROOT/models/quant.py
sed -n 651p $NCM_ROOT/models/quant.py | tee -a $OUT/progress.txt
pip install -q omegaconf hydra-core "transformers>=4.51" "huggingface_hub[cli]" regex 2>&1 | tail -2

log "download checkpoint and data"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hazyresearch/ncm-checkpoints", allow_patterns=["tinystories/*"], local_dir="$NCM_ROOT/checkpoints")
snapshot_download("hazyresearch/ncm-datasets", repo_type="dataset",
                  allow_patterns=["tinystories/train_bpe_8192*"], local_dir="$NCM_ROOT/data")
EOF
CK=$NCM_ROOT/checkpoints/tinystories
DA=$NCM_ROOT/data/tinystories
CH=$W/chunked
RU=$W/runs
cd $REPO

log "encode M=4 ($MAX_DOCS docs)"
python experiments/prepare_chunks.py --M 4 --max-docs $MAX_DOCS --data $DA --ckpt $CK --out $CH 2>&1 | quiet | tail -3

log "GPU checks"
python experiments/test_fast.py --device cuda --bs 2 --ckpt $CK --chunked $CH 2>&1 | quiet | tee $OUT/test_fast.txt
python experiments/test_refiner.py --device cuda --ckpt $CK --chunked $CH 2>&1 | quiet | tee $OUT/test_refiner.txt

COMMON="--M 4 --G 3 --ckpt $CK --chunked $CH --runs $RU --steps $STEPS --bs 8 --eval-every 500 --grad-ckpt --corrupt-mean 0.5"
T=$RU/oursc_M4_G3
log "teacher: full-depth Block-NCM G=3"
python experiments/train.py $COMMON --lr 3e-5 --fast --name oursc_M4_G3 2>&1 | quiet > $OUT/train_teacher.txt
grep -E "autocast| val |done in" $OUT/train_teacher.txt | tee -a $OUT/progress.txt

log "depth probe"
python experiments/depth_probe.py --run $T --G 3 --n 256 --ckpt $CK --chunked $CH 2>&1 | quiet | tee $OUT/depth_probe.txt
LAYERS=$(python -c "import json; print(','.join(map(str, json.load(open('$T/depth_probe_start4.json'))['best']['6']['layers'])))")
echo "best 6/12 layers: $LAYERS" | tee -a $OUT/progress.txt

for S in 4 7; do
  N=refc_M4_G3_L6_s$S
  log "refiner $N (layers $LAYERS from level $S)"
  python experiments/train.py $COMMON --lr 1e-4 --name $N --refiner-layers $LAYERS --refiner-start-level $S \
    --distill 1.0 --init-from $T 2>&1 | quiet > $OUT/train_$N.txt
  grep -E "refiner:|autocast| val |done in" $OUT/train_$N.txt | tee -a $OUT/progress.txt
done

log "NLL breakdown"
python experiments/nll_breakdown.py --M 4 --max-docs $MAX_DOCS --runs-dir $RU --ckpt $CK --chunked $CH --data $DA \
  --runs oursc_M4_G3:3 refc_M4_G3_L6_s4:3 refc_M4_G3_L6_s7:3 --out $OUT/nll_breakdown_refiner.json 2>&1 | quiet \
  > $OUT/nll_breakdown_refiner.txt
sed -n '/non-padding codes only/,$p' $OUT/nll_breakdown_refiner.txt | tee -a $OUT/progress.txt

for N in oursc_M4_G3 refc_M4_G3_L6_s4 refc_M4_G3_L6_s7; do
  log "generation + latency: $N"
  python experiments/sample_eval.py --run $RU/$N --M 4 --G 3 --fast --halo 0 --ckpt $CK \
    --out $OUT/eval_${N}_M4.json 2>&1 | quiet > $OUT/eval_${N}_M4.txt
  grep -E '^  "(genppl|genppl_sd|entropy|boundary_ratio|latency_s_per_sample)"' $OUT/eval_${N}_M4.txt | tee -a $OUT/progress.txt
done

cp $RU/*/log.jsonl $OUT/ 2>/dev/null; for d in $RU/*/; do cp $d/log.jsonl $OUT/$(basename $d)_log.jsonl 2>/dev/null; done
cp $T/depth_probe_start4.json $OUT/ 2>/dev/null
rm -rf $NCM_ROOT/data $CH   # keep the saved output small: results + run checkpoints
log "done"
