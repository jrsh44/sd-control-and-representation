#!/bin/bash
################################################################################
# SLURM Array Job Script - Unlearned Representation Cache Generation
# Purpose: Generate unlearned images in parallel
#
# Usage:
#   sbatch scripts/sd_v1_5/generate_unlearned_cache_batch.sh
#
# Description:
#   This script runs parallel tasks to generate unlearned representation caches:
#   - Each task processes one concept from the nudity dataset
#   - Uses SAE for unlearning specific concepts
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Adjust based on dataset size and resources
#==============================================================================

#SBATCH --job-name=unlearn_gen
#SBATCH --account=mi2lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/unlearn_gen_%A_%a.out
#SBATCH --error=logs/unlearn_gen_%A_%a.err

#==============================================================================
# ERROR HANDLING
#==============================================================================

set -e

#==============================================================================
# SETUP
#==============================================================================
echo "========================================="
echo "Unlearned Cache Generation Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "========================================="

cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
source .venv/bin/activate
source ./.venv/bin/activate

# Create directories
mkdir -p ../logs

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set}"
echo "  HF_HOME: ${HF_HOME:-Not set}"
echo ""

# Configuration
CONCEPTS=("exposed feet" "exposed belly" "exposed anus" "exposed penis" "exposed vagina" "exposed buttocks" "exposed breast" "exposed armpits")
FEATURE_NUMBERS=(1 2 4 8 12 16)
INFLUENCE_FACTORS=(0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8)

# Paths to fill
PROMPTS_FILE="/mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation/data/nudity/prompts_test.txt"  # e.g., "data/prompts/templates.txt"
SAE_DIR_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp32_topk16_lr5em4_warmup100000_aux003125_ep2_bs4096"  # e.g., "/path/to/sae/directory"
CONCEPT_SUMS_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp32_topk16_lr5em4_warmup100000_aux003125_ep2_bs4096/feature_merged/merged_feature_sums.pt"  # e.g., "/path/to/concept_sums.pt"
RESULTS_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp32_topk16_lr5em4_warmup100000_aux003125_ep2_bs4096/images"  # e.g., "/path/to/results"

# Get concept for this array task
CONCEPT="${CONCEPTS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Concept: $CONCEPT"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "========================================="

#==============================================================================
# GPU CHECK
#==============================================================================

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found, GPU may not be available"
    echo ""
fi

#==============================================================================
# RUN CACHE GENERATION
#==============================================================================

uv run scripts/sd_v1_5/generate_unlearned_cache_from_file.py \
    --dataset-name "unlearning_evaluation" \
    --prompts-file "$PROMPTS_FILE" \
    --concept "$CONCEPT" \
    --results-dir "$RESULTS_DIR" \
    --sae-dir-path "$SAE_DIR_PATH" \
    --concept-sums-path "$CONCEPT_SUMS_PATH" \
    --feature-numbers ${FEATURE_NUMBERS[@]} \
    --influence-factors ${INFLUENCE_FACTORS[@]} \
    --guidance-scale 7.5 \
    --steps 50 \
    --seed 42 \
    --generate_without_unlearning

echo "========================================="
echo "Job completed successfully!"
echo "Concept: $CONCEPT"
echo "Date: $(date)"
echo "========================================="
