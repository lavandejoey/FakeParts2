#!/bin/bash
#SBATCH --job-name=VideoChat
#SBATCH --output=logs/DiffusersT2V_VideoChat_%j.out
#SBATCH --error=logs/DiffusersT2V_VideoChat_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
set -euo pipefail

# Print environment and script info
echo "============================================================"
echo "SLURM JOB: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepfake311

# Set experiment variables
srun python3 -W ignore ./Transformer_VideoChat.py \
  -v "/projects/hi-paris/DeepFakeDataset/DeepFake_V2/10k_real" \
  -o "/projects/hi-paris/DeepFakeDataset/DeepFake_V2/10k_real_video_captions_ziyi.csv" \
  -n 10000

EXIT_CODE=$?

echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE