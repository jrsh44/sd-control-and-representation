#!/bin/bash
#SBATCH --account=mi2lab
#SBATCH --partition=short
#SBATCH --time=0-23:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=lpips_scores
#SBATCH --output=logs/lpips_scores_%j.out
#SBATCH --error=logs/lpips_scores_%j.err

# Script to calculate LPIPS scores between no_intervention and intervention images
# This script uses VGG backbone with learned weights for perceptual similarity

# Example usage:
# sbatch scripts/image_evaluation/calculate_lpips_scores.sh

# Set source folder and output CSV
SOURCE_FOLDER="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/images_per_timestep"
OUTPUT_CSV="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/scores_per_timestep/lpips_scores.csv"

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script
echo "Starting LPIPS score calculation..."
echo "Source folder: $SOURCE_FOLDER"
echo "Output CSV: $OUTPUT_CSV"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"

uv run scripts/image_evaluation/calculate_lpips_scores.py \
  --source_folder "$SOURCE_FOLDER" \
  --output_csv "$OUTPUT_CSV"

echo "Job completed!"
