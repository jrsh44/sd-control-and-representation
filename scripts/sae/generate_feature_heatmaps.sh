#!/bin/bash
#SBATCH --job-name=feat_heatmap
#SBATCH --output=logs/feat_heatmap_%j.out
#SBATCH --error=logs/feat_heatmap_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e
mkdir -p logs

# Activate environment
if [ -z "${CONDA_PREFIX}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate sd-control
fi

echo "=========================================="
echo "SAE Feature Heatmap Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "=========================================="

# Configuration
SAE_PATH="/mnt/evafs/groups/mi2lab/bjezierski/results_tmp/sae_models/unet_up_1_att_1_exp16_topk32_lr4em4_epochs5_batch4096_job0.pt"
FEATURE_SELECTION_RESULTS="/path/to/feature_selection_results.pt"  # UPDATE THIS
PROMPT="a photo of a cat"
LAYER="UNET_UP_1_ATT_1"
MODEL="SD_V1_5"
TIMESTEPS=(0 10 20 30 40)
TOP_K=5
OUTPUT_PATH="results/feature_heatmaps"

# Run script
python scripts/sae/generate_feature_heatmaps.py \
    --sae_path "${SAE_PATH}" \
    --feature_selection_results "${FEATURE_SELECTION_RESULTS}" \
    --prompt "${PROMPT}" \
    --layer "${LAYER}" \
    --model "${MODEL}" \
    --timesteps ${TIMESTEPS[@]} \
    --top_k_features ${TOP_K} \
    --output_path "${OUTPUT_PATH}" \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --alpha 0.4

echo "=========================================="
echo "✓ Complete"
echo "=========================================="
