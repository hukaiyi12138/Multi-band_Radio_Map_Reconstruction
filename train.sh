#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch training launcher for PR-BTD-DULRTC
#
# Fixed N_iter = 10.
#
# One tmux window for each job:
#   mask/fiber x omega_num=1,5,10,15
#
# Total jobs: 2 * 4 = 8
#
# Output dirs:
#   runs/mask1
#   runs/mask5
#   runs/mask10
#   runs/mask15
#   runs/fiber1
#   runs/fiber5
#   runs/fiber10
#   runs/fiber15
# ============================================================

# ---------------- user config ----------------
SESSION_NAME="lrt"
TRAIN_PY="train.py"

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="lrt"

DATASET="dulrtc_triple"
ROOT="/data/home/hky/dataset/DULRTC_triple"
SAVE_ROOT="runs"

R=3
K=3
N_ITER=10
N_RAY_SAMPLES=16
PROX_HIDDEN=32
PROX_LAYERS=3
EPOCHS=50
BATCH_SIZE=1
LR="1e-3"
WEIGHT_DECAY="0.0"
GRAD_CLIP="1.0"
NUM_WORKERS=2
SEED=0
MU_GRAD="0"        # 梯度损失权重,0.0 = 退回纯 L1

GPUS=(0 1 2 3 4 5)

MASK_TYPES=(mask fiber)
OMEGA_NUMS=(1 5 10 15)

EXTRA_ARGS=""

# ---------------- checks ----------------
if ! command -v tmux >/dev/null 2>&1; then
    echo "[error] tmux is not installed or not found in PATH."
    exit 1
fi

if [[ ! -f "${TRAIN_PY}" ]]; then
    echo "[error] Cannot find ${TRAIN_PY} in current directory: $(pwd)"
    exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[error] Cannot find conda.sh: ${CONDA_SH}"
    echo "        Please check your miniconda/anaconda path."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "${SAVE_ROOT}"

# ---------------- create tmux session ----------------
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[info] tmux session '${SESSION_NAME}' already exists. New windows will be added."
else
    tmux new-session -d -s "${SESSION_NAME}" -n "launcher"
    tmux send-keys -t "${SESSION_NAME}:launcher" \
        "echo 'PR-BTD-DULRTC training launcher session: ${SESSION_NAME}'" C-m
fi

job_id=0

for mask_type in "${MASK_TYPES[@]}"; do
    for omega_num in "${OMEGA_NUMS[@]}"; do
        gpu_id="${GPUS[$((job_id % ${#GPUS[@]}))]}"

        exp_name="${mask_type}${omega_num}"
        save_dir="${SAVE_ROOT}/${exp_name}"
        log_path="${save_dir}/tmux_train.log"
        window_name="${exp_name}"

        mkdir -p "${save_dir}"

        # 避免重复创建同名窗口
        if tmux list-windows -t "${SESSION_NAME}" -F '#W' | grep -qx "${window_name}"; then
            echo "[skip] window already exists: ${window_name}"
            job_id=$((job_id + 1))
            continue
        fi

        cmd="source ${CONDA_SH} && \
conda activate ${CONDA_ENV} && \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && \
CUDA_VISIBLE_DEVICES=${gpu_id} python ${TRAIN_PY} \
  --dataset ${DATASET} \
  --root ${ROOT} \
  --omega-num ${omega_num} \
  --mask-type ${mask_type} \
  --R ${R} \
  --K ${K} \
  --N-iter ${N_ITER} \
  --n-ray-samples ${N_RAY_SAMPLES} \
  --prox-hidden ${PROX_HIDDEN} \
  --prox-layers ${PROX_LAYERS} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --num-workers ${NUM_WORKERS} \
  --seed ${SEED} \
  --mu-grad ${MU_GRAD} \
  --save-root ${SAVE_ROOT} \
  ${EXTRA_ARGS} 2>&1 | tee ${log_path}"

        tmux new-window -t "${SESSION_NAME}" -n "${window_name}"
        tmux send-keys -t "${SESSION_NAME}:${window_name}" "cd $(pwd)" C-m
        tmux send-keys -t "${SESSION_NAME}:${window_name}" "echo '[start] ${exp_name} on GPU ${gpu_id}, N_iter=${N_ITER}'" C-m
        tmux send-keys -t "${SESSION_NAME}:${window_name}" "${cmd}" C-m

        echo "[launch] ${exp_name} -> tmux:${SESSION_NAME}:${window_name}, GPU=${gpu_id}, log=${log_path}"

        job_id=$((job_id + 1))
    done
done

echo ""
echo "[done] Launched ${job_id} training jobs."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "List windows: tmux list-windows -t ${SESSION_NAME}"
echo "Kill all jobs: tmux kill-session -t ${SESSION_NAME}"