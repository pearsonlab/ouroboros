#!/usr/bin/env bash
# Drive a sequential Arneodo seed pool: train each seed for `EPOCHS` epochs,
# then run the checkpoint eval to pick the best seed by mid-voc rescaled
# autonomy on val (day85). After all seeds finish, write a comparison table
# + generate recon WAV + plot for the winner.
set -e

cd /home/pearson/code/ouroboros

POOL_DIR="${POOL_DIR:-./finch_blk445_syllC_arneodo_pool}"
DATA_GLOB="${DATA_GLOB:-/home/pearson/ouroboros_data/blk445_syllC/day*}"
TRAIN_GLOB="${TRAIN_GLOB:-/home/pearson/ouroboros_data/blk445_syllC/day84}"
N_SEEDS="${N_SEEDS:-16}"
EPOCHS="${EPOCHS:-100}"
SEG="${SEG:-5}"
MAX_VOCS="${MAX_VOCS:-1000}"

mkdir -p "${POOL_DIR}"
SUMMARY="${POOL_DIR}/pool_summary.csv"
echo "seed,train_r2,val_r2,rescaled_auto,spec_corr,pitch_pen,bounded" > "${SUMMARY}"

for s in $(seq 0 $((N_SEEDS - 1))); do
  SEED_DIR="${POOL_DIR}/seed${s}"
  mkdir -p "${SEED_DIR}"
  echo "=== seed ${s} train start at $(date +%H:%M:%S) ==="
  .venv/bin/python -m examples.train_arneodo_big \
      --data-glob "${TRAIN_GLOB}" \
      --out-dir "${SEED_DIR}" \
      --max-vocs ${MAX_VOCS} \
      --epochs ${EPOCHS} --seg ${SEG} \
      --n-layers 3 --d-state 4 --d-conv 4 --expand-factor 10 \
      --drive-lowpass-ms 1.0 \
      --batch-size 8 --context-len 0.1 \
      --seed ${s} --n-jobs 8 \
      --target-r2 0.99 \
      > "${SEED_DIR}/train.log" 2>&1

  echo "=== seed ${s} eval start at $(date +%H:%M:%S) ==="
  # eval the latest checkpoint only (the script will print all available)
  EVAL_OUT="${SEED_DIR}/eval.txt"
  .venv/bin/python scripts/eval_checkpoints.py \
      --run-dir "${SEED_DIR}/arneodo" \
      --data-glob "${DATA_GLOB}" 2>&1 \
      | grep -vE "RuntimeWarning|nanmean|loading from|model tau|^\s*$|Train r2|train r2|val r2|^ *spec_corr|power_mat|return vv" \
      > "${EVAL_OUT}"

  # take the LAST line of the eval table as the final-checkpoint number
  LAST=$(grep -E "^ *[0-9]+ " "${EVAL_OUT}" | tail -1)
  echo "seed ${s}: ${LAST}"
  # parse columns: epoch train_r2 val_r2 rescaled_auto spec amp_pen pitch_pen bounded
  read epoch train_r2 val_r2 auto spec amp pitch bounded <<< "${LAST}"
  echo "${s},${train_r2},${val_r2},${auto},${spec},${pitch},${bounded}" >> "${SUMMARY}"
done

echo "=== all ${N_SEEDS} seeds done at $(date +%H:%M:%S) ==="
echo "summary:"
cat "${SUMMARY}"

# pick best seed by rescaled_auto (column 4 in summary)
BEST_SEED=$(tail -n +2 "${SUMMARY}" | sort -t, -k4 -gr | head -1 | cut -d, -f1)
echo "=== best seed by autonomy: ${BEST_SEED} ==="

# find the latest checkpoint of the best seed
BEST_CKPT=$(ls "${POOL_DIR}/seed${BEST_SEED}/arneodo/"*.tar 2>/dev/null \
            | xargs -n1 basename \
            | sed 's/checkpoint_\([0-9]*\)\.tar/\1/' \
            | sort -n | tail -1)
BEST_CKPT_PATH="${POOL_DIR}/seed${BEST_SEED}/arneodo/checkpoint_${BEST_CKPT}.tar"
echo "best ckpt: ${BEST_CKPT_PATH}"

echo "=== writing recon WAVs + comparison plots for best seed ==="
.venv/bin/python scripts/resynth_recon.py \
    --ckpt "${BEST_CKPT_PATH}" \
    --data-glob "${DATA_GLOB}" \
    --out-dir "${POOL_DIR}/best_recons" \
    --n-vocs 3 2>&1 | grep -vE "RuntimeWarning|power_mat|nanmean|return vv"

.venv/bin/python scripts/plot_recon_grid.py \
    --ckpt "${BEST_CKPT_PATH}" \
    --data-glob "${DATA_GLOB}" \
    --out "${POOL_DIR}/best_recon_comparison.png" 2>&1 | grep -vE "RuntimeWarning|power_mat|nanmean|return vv"

.venv/bin/python scripts/plot_coldstart_grid.py \
    --ckpt "${BEST_CKPT_PATH}" \
    --data-glob "${DATA_GLOB}" \
    --out "${POOL_DIR}/best_coldstart_comparison.png" 2>&1 | grep -vE "RuntimeWarning|power_mat|nanmean|return vv"

echo "=== POOL DONE ==="
echo "summary: ${SUMMARY}"
echo "best seed: ${BEST_SEED}  ckpt: ${BEST_CKPT_PATH}"
echo "plots: ${POOL_DIR}/best_recon_comparison.png"
echo "       ${POOL_DIR}/best_coldstart_comparison.png"
