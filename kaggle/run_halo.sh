#!/usr/bin/env bash
# Halo / G combinations on Kaggle's two T4s (one training run per GPU, in parallel). Internet ON.
#
# Notebook cell:
#   !git clone -q https://github.com/Rishab-Gupta24/block-ncm /kaggle/working/block-ncm
#   !bash /kaggle/working/block-ncm/kaggle/run_halo.sh
#
# MODE (env var, else the file kaggle/MODE, else "full"): smoke = every stage at tiny size; full = the real run.
# Full mode uses ALL eligible TinyStories docs and the same val split as the local runs (Milestones 3b/5), so the
# NLL breakdown is on the same 1,228 val docs. Differences from local: fp16 (T4) instead of bf16. The released
# model's breakdown is included as a cross-platform calibration point.
#
# Runs (recipe: corruption 0.5, lr 3e-5, 1,500 steps x 8 docs, KV-cached plan):
#   GPU 0: oursc_M4_G5_halo4, then oursc_M4_G6_halo4
#   GPU 1: oursc_M4_G3_halo8, then oursc_M4_G3_halo2
set -u
W=/kaggle/working
REPO=$W/block-ncm
OUT=$W/results
MODE=${MODE:-$(cat $REPO/kaggle/MODE 2>/dev/null | tr -d '[:space:]')}
MODE=${MODE:-full}
if [ "$MODE" = smoke ]; then
  MAX_DOCS=1500; STEPS=20; EVAL_EVERY=10; VAL_DOCS=16; GEN="--n 8 --seeds 42"; BS4=8; BS6=8; NBOOT=50
else
  MAX_DOCS=0; STEPS=1500; EVAL_EVERY=500; VAL_DOCS=512; GEN=""; BS4=32; BS6=16; NBOOT=1000
fi
export NCM_ROOT=$W/ncm PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT
FAIL=0
log() { echo "=== $* $(date -u +%H:%M:%S)" | tee -a $OUT/progress.txt; }
quiet() { grep -v "parameter count" || true; }
check() {
  # a training run must also have finished: model.pt is rewritten at every evaluation, so it exists after a kill
  if [ -s "$1" ] && ! grep -qiE "loss nan|non-finite|Traceback" "$2" && { [[ "$1" != *model.pt ]] || grep -q "^done in" "$2"; }; then echo "    ok: $1" | tee -a $OUT/progress.txt
  else echo "    MISSING: $1 -- last lines of $2:" | tee -a $OUT/progress.txt; tail -30 "$2" | tee -a $OUT/progress.txt; FAIL=1; fi
}

log "mode $MODE (max docs $MAX_DOCS, steps $STEPS)"
nvidia-smi -L | tee $OUT/gpu.txt
NGPU=$(nvidia-smi -L | wc -l)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.device_count(), 'GPUs')" | tee -a $OUT/gpu.txt

log "setup"
[ -d $NCM_ROOT ] || git clone -q https://github.com/HazyResearch/ncm $NCM_ROOT
(cd $NCM_ROOT && git checkout -q c702742)
sed -i '651s/z_hat.add_(z_q_level)/z_hat = z_hat + z_q_level/' $NCM_ROOT/models/quant.py
pip install -q omegaconf hydra-core "transformers>=4.51" huggingface_hub regex 2>&1 | tail -2

log "download checkpoint and data"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hazyresearch/ncm-checkpoints", revision="d342ce8aaea58830eef3b8ad473664673e41f692", allow_patterns=["tinystories/*"], local_dir="$NCM_ROOT/checkpoints")
snapshot_download("hazyresearch/ncm-datasets", repo_type="dataset",
                  allow_patterns=["tinystories/train_bpe_8192*"], local_dir="$NCM_ROOT/data")
EOF
CK=$NCM_ROOT/checkpoints/tinystories
DA=$NCM_ROOT/data/tinystories
CH=$W/chunked
RU=$W/runs
cd $REPO

log "encode M=4 (max docs $MAX_DOCS)"
python experiments/prepare_chunks.py --M 4 --max-docs $MAX_DOCS --data $DA --ckpt $CK --out $CH 2>&1 | quiet > $OUT/encode.txt
tail -2 $OUT/encode.txt | tee -a $OUT/progress.txt
check $CH/val_M4.npy $OUT/encode.txt

log "GPU check (fast path)"
CUDA_VISIBLE_DEVICES=0 python experiments/test_fast.py --device cuda --bs 2 --no-timing --ckpt $CK --chunked $CH 2>&1 | quiet > $OUT/test_fast.txt
grep -E "^\[|passed" $OUT/test_fast.txt | tee -a $OUT/progress.txt
grep -q "all fast-path checks passed" $OUT/test_fast.txt || { echo "    FAILED test_fast"; tail -30 $OUT/test_fast.txt; FAIL=1; }

COMMON="--M 4 --ckpt $CK --chunked $CH --runs $RU --steps $STEPS --bs 8 --lr 3e-5 --eval-every $EVAL_EVERY --val-docs $VAL_DOCS --grad-ckpt --corrupt-mean 0.5 --fast"
train_list() {  # train_list GPU "name:G:H ..."
  local gpu=$1; shift
  for spec in $@; do
    IFS=: read N G H <<< "$spec"
    echo "=== [gpu $gpu] train $N (G=$G halo=$H) $(date -u +%H:%M:%S)" >> $OUT/progress.txt
    CUDA_VISIBLE_DEVICES=$gpu python experiments/train.py $COMMON --G $G --halo $H --name $N 2>&1 | quiet > $OUT/train_$N.txt
    grep -E "autocast| val |done in|nan|inf" $OUT/train_$N.txt | tail -6 | sed "s/^/[gpu $gpu] /" >> $OUT/progress.txt
  done
}
L0="oursc_M4_G5_halo4:5:4 oursc_M4_G6_halo4:6:4"
L1="oursc_M4_G3_halo8:3:8 oursc_M4_G3_halo2:3:2"
log "train ($NGPU GPUs)"
if [ "$NGPU" -ge 2 ]; then
  train_list 0 $L0 & P0=$!
  train_list 1 $L1 & P1=$!
  wait $P0 $P1
else
  train_list 0 $L0 $L1
fi
cat $OUT/progress.txt | sed -n '/=== train/,$p' | tail -30
for spec in $L0 $L1; do N=${spec%%:*}; check $RU/$N/model.pt $OUT/train_$N.txt; done

log "NLL breakdown (released model as calibration)"
CUDA_VISIBLE_DEVICES=0 python experiments/nll_breakdown.py --M 4 --max-docs $MAX_DOCS --runs-dir $RU --ckpt $CK --chunked $CH --data $DA \
  --n-boot $NBOOT --runs released:0 oursc_M4_G3_halo2:3 oursc_M4_G3_halo8:3 oursc_M4_G5_halo4:5 oursc_M4_G6_halo4:6 \
  --out $OUT/nll_breakdown_halo.json 2>&1 | quiet > $OUT/nll_breakdown_halo.txt
sed -n '/non-padding codes only/,$p' $OUT/nll_breakdown_halo.txt | tee -a $OUT/progress.txt
check $OUT/nll_breakdown_halo.json $OUT/nll_breakdown_halo.txt

eval_list() {  # eval_list GPU "name:G ..."
  local gpu=$1; shift
  for spec in $@; do
    IFS=: read N G H <<< "$spec"
    for M in 4 6; do
      local bs=$BS4; [ $M = 6 ] && bs=$BS6
      local f=$OUT/eval_${N}_M$M
      while [ $bs -ge 2 ]; do
        CUDA_VISIBLE_DEVICES=$gpu python experiments/sample_eval.py --run $RU/$N --M $M --G $G --fast --ckpt $CK $GEN \
          --bs $bs --out $f.json 2>&1 | quiet > $f.txt
        grep -q '"genppl"' $f.txt && break
        bs=$((bs / 2))
      done
      { echo "=== [gpu $gpu] eval $N M=$M (bs $bs)"; grep -E '^  "(genppl|genppl_sd|entropy|boundary_ratio|latency_s_per_sample)"' $f.txt; } >> $OUT/progress.txt
    done
  done
}
log "generation M=4 and M=6"
if [ "$NGPU" -ge 2 ]; then
  eval_list 0 $L0 & P0=$!
  eval_list 1 $L1 & P1=$!
  wait $P0 $P1
else
  eval_list 0 $L0 $L1
fi
sed -n '/eval oursc/,$p' $OUT/progress.txt
for spec in $L0 $L1; do N=${spec%%:*}; for M in 4 6; do check $OUT/eval_${N}_M$M.json $OUT/eval_${N}_M$M.txt; done; done

for d in $RU/*/; do cp $d/log.jsonl $OUT/$(basename $d)_log.jsonl 2>/dev/null; done
rm -rf $NCM_ROOT/data $CH $RU/*/ckpt_* 2>/dev/null   # keep the saved output small
if [ $FAIL = 0 ]; then log "ALL STAGES PASSED ($MODE)"; else log "SOME STAGES FAILED ($MODE): see MISSING/FAILED above"; fi
