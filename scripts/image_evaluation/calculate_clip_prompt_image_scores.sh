#!/bin/bash
#SBATCH --account=mi2lab
#SBATCH --partition=short
#SBATCH --time=0-23:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=clip_ti_scores
#SBATCH --output=../logs/clip_ti_scores_%j.out
#SBATCH --error=../logs/clip_ti_scores_%j.err

# Script to calculate CLIP scores between prompts and images
# This script fills prompts with concept names and compares them with images

# Example usage:
# sbatch scripts/image_evaluation/calculate_clip_prompt_image_scores.sh

# Set source folder, prompts file, and output CSV
SOURCE_FOLDER="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/images"
PROMPTS_FILE="data/nudity/prompts_analysis.txt"
OUTPUT_CSV="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/clip_prompt_scores.csv"

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script
echo "Starting CLIP prompt-to-image score calculation..."
echo "Source folder: $SOURCE_FOLDER"
echo "Prompts file: $PROMPTS_FILE"
echo "Output CSV: $OUTPUT_CSV"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

uv run scripts/image_evaluation/calculate_clip_prompt_image_scores.py \
  --source_folder "$SOURCE_FOLDER" \
  --prompts_file "$PROMPTS_FILE" \
  --output_csv "$OUTPUT_CSV"

echo "Job completed!"
