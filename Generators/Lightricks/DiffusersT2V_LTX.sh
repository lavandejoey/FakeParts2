#!/bin/bash
#SBATCH --job-name="LTX"
#SBATCH --output=logs/DiffusersT2V_LTX_%j.out
#SBATCH --error=logs/DiffusersT2V_LTX_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=L40S
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

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
srun python3 -W ignore \
  ./DiffusersT2V_LTX.py \
  -p "/projects/hi-paris/DeepFakeDataset/T2V_prompts/prompts_gemini/gemini_3" \
  -o "/projects/hi-paris/DeepFakeDataset/DeepFake_V2/T2V/Lightricks/LTX-Video-0.9.7-distilled" \
  -n 1000

EXIT_CODE=$?

echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
