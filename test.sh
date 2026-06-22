#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================================
# Batch testing for PR-BTD-DULRTC
#
# Fixed N_iter = 10.
#
# Seeds should match the training script.
#
# Checkpoint dirs:
#   runs_seed0/mask1/best.pt
#   runs_seed42/mask1/best.pt
#   runs_seed2024/mask1/best.pt
#
# Output dirs:
#   /data/home/hky/DULRTC/DUSPF_RME/test_seed0
#   /data/home/hky/DULRTC/DUSPF_RME/test_seed42
#   /data/home/hky/DULRTC/DUSPF_RME/test_seed2024
# ============================================================

ROOT="/data/home/hky/dataset/DULRTC_triple"
TEST_OUT_BASE="/data/home/hky/DULRTC/DUSPF_RME/test"

CKPT_ROOT_BASE="runs"

DATASET="dulrtc_triple"
R=3
K=3
N_ITER=10
BATCH_SIZE=1
NUM_WORKERS=2
GPU_ID=0

# Match training seeds
SEEDS=(0 42 2024)

# Save figures for the first 10 environments only.
# Set to 0 or negative to save all figures.
MAX_SAVE_FIGURES=10

MASK_TYPES=(fiber)
OMEGA_NUMS=(15 10 5 1)

echo "========== Start Testing =========="

for seed in "${SEEDS[@]}"; do
    CKPT_ROOT="${CKPT_ROOT_BASE}_seed${seed}"
    TEST_OUT="${TEST_OUT_BASE}_seed${seed}"

    echo ""
    echo "============================================================"
    echo "[seed] ${seed}"
    echo "[ckpt root] ${CKPT_ROOT}"
    echo "[test out]  ${TEST_OUT}"
    echo "============================================================"

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
            echo "[test] ${exp_name}, seed=${seed}"
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
    echo "========== Start Multi-seed Evaluation =========="

    python evaluation.py \
        --roots \
        "${TEST_OUT_BASE}_seed0" \
        "${TEST_OUT_BASE}_seed42" \
        "${TEST_OUT_BASE}_seed2024" \
        --out "eval/evaluation_multiseed_summary"

    echo "========== All Done =========="

done

echo ""
echo "========== All Done =========="