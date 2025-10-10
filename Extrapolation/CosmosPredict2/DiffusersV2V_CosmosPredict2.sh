#!/bin/bash
#SBATCH --job-name=CosmosPredict2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=H100
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node57

# -------- shell hygiene --------
set -euo pipefail
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

# -------- conda / modules --------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosmos311

# -------- reproducibility & logging --------
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# -------- CUDA / NCCL sanity --------
# Expose only the GPUs SLURM granted this job step
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
# Reasonable defaults; flip IB if your fabric supports it
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
# If your cluster has no InfiniBand, uncomment:
# export NCCL_IB_DISABLE=1

# Optional perf knobs (tune as needed)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MALLOC_TRIM_THRESHOLD_=134217728   # 128MB

# -------- paths / args (edit me) --------
SCRIPT="/home/infres/ziyliu-24/FakeParts2/Extrapolation/CosmosPredict2/DiffusersV2V_CosmosPredict2.py"
VIDEOS_CSV="/projects/hi-paris/DeepFakeDataset/DeepFake_V2/10k_real_video_captions_ziyi.csv"
OUT_DIR="/projects/hi-paris/DeepFakeDataset/DeepFake_V2/V2V/nvidia/Cosmos-Predict2-14B-Video2World"
NUM_SAMPLES=1000
NP=${SLURM_GPUS_ON_NODE:-2}

# -------- quick hardware + torch sanity --------
nvidia-smi -L || true
srun --gres=gpu:"${NP}" python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "No CUDA devices visible to this job."
PY

# -------- torchrun launch (single-node) --------
# Use --standalone to avoid rdvz boilerplate for 1 node.

srun --gres=gpu:"${NP}" torchrun \
  --standalone \
  --nproc_per_node="${NP}" \
  "${SCRIPT}" \
  -v "${VIDEOS_CSV}" \
  -o "${OUT_DIR}" \
  -n "${NUM_SAMPLES}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"