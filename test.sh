#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================================
# Batch testing for PR-BTD-DULRTC
#
# Fixed N_iter = 10.
#
# Checkpoints:
#   runs/mask1/best.pt
#   runs/mask5/best.pt
#   runs/mask10/best.pt
#   runs/mask15/best.pt
#   runs/fiber1/best.pt
#   runs/fiber5/best.pt
#   runs/fiber10/best.pt
#   runs/fiber15/best.pt
#
# Output:
#   TEST_OUT/Mask1/
#   TEST_OUT/Mask5/
#   TEST_OUT/Fiber1/
#   ...
# ============================================================

ROOT="/data/home/hky/dataset/DULRTC_triple"
TEST_OUT="/data/home/hky/DULRTC/hky_try_3/test"

CKPT_ROOT="runs"

DATASET="dulrtc_triple"
R=3
K=3
N_ITER=10
BATCH_SIZE=1
NUM_WORKERS=2
GPU_ID=0

# 只保存前 10 个环境图片；设为 0 或负数表示保存全部
MAX_SAVE_FIGURES=10

MASK_TYPES=(mask fiber)
OMEGA_NUMS=(15 10 5 1)

echo "========== Start Testing =========="
mkdir -p "${TEST_OUT}"

for mask_type in "${MASK_TYPES[@]}"; do
    for omega_num in "${OMEGA_NUMS[@]}"; do

        exp_name="${mask_type}${omega_num}"
        ckpt_path="${CKPT_ROOT}/${exp_name}/best.pt"

        if [[ ! -f "${ckpt_path}" ]]; then
            echo "[skip] checkpoint not found: ${ckpt_path}"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "[test] ${exp_name}"
        echo "[ckpt] ${ckpt_path}"
        echo "------------------------------------------------------------"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
            --checkpoint "${ckpt_path}" \
            --dataset "${DATASET}" \
            --root "${ROOT}" \
            --omega-num "${omega_num}" \
            --mask-type "${mask_type}" \
            --R "${R}" \
            --K "${K}" \
            --N-iter "${N_ITER}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${NUM_WORKERS}" \
            --output-path "${TEST_OUT}" \
            --max-save-figures "${MAX_SAVE_FIGURES}"

    done
done

echo ""
echo "========== Start Evaluation =========="

python evaluation.py --root "${TEST_OUT}"

echo "========== All Done =========="