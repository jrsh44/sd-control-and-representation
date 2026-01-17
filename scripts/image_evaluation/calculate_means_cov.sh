#!/bin/bash
################################################################################
# SLURM Array Job Script - Batch Statistics Extraction
#
# Purpose:
#   Extract and save Inception v3 statistics from multiple concept directories
#   in parallel using SLURM array jobs
#
# Usage:
#   sbatch scripts/image_evaluation/calculate_means_cov.sh
#
# Output:
#   - Mean and covariance matrices saved as .npy files
#   - Logs: logs/stats_array_%A_%a.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name stats_batch
#SBATCH --array=0-7  # 8 concepts: exposed_vagina, exposed_anus, exposed_feet, buttocks, breast, belly, armpits, penis
#SBATCH --time 0-01:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --partition short
#SBATCH --output logs/stats_array_%A_%a.log  # %A=job ID, %a=array index

# Optional email notification
# #SBATCH --mail-user=01180694@pw.edu.pl
# #SBATCH --mail-type=FAIL,END

#==============================================================================
# ERROR HANDLING
#==============================================================================
set -eu

#==============================================================================
# SETUP
#==============================================================================

echo "=========================================="
echo "Batch Statistics Extraction - Array Job"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-no_job_id}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
source .venv/bin/activate
source ./.venv/bin/activate

# Create directories
mkdir -p logs

#==============================================================================
# CONFIGURATION
#==============================================================================

# Base path for all experiments
BASE_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096"

# Array of concepts to process
CONCEPTS=(
    "exposed_vagina"
    "exposed_anus"
    "exposed_buttocks"
    "exposed_breast"
    "exposed_belly"
    "exposed_armpits"
    "exposed_penis"
)

# Get the concept for this array task
CONCEPT="${CONCEPTS[$SLURM_ARRAY_TASK_ID]}"

# Construct paths
CONCEPT_DIR="${BASE_PATH}/images_per_timestep/${CONCEPT}"
OUTPUT_DIR="${BASE_PATH}/fid/nudity_per_timestep/${CONCEPT}"

# Python script
PYTHON_SCRIPT="scripts/image_evaluation/calculate_means_cov.py"

echo ""
echo "Configuration:"
echo "  Concept: ${CONCEPT}"
echo "  Concept directory (parent): ${CONCEPT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "=========================================="

#==============================================================================
# VALIDATION
#==============================================================================

if [ ! -d "${CONCEPT_DIR}" ]; then
    echo "ERROR: Concept directory not found: ${CONCEPT_DIR}"
    exit 1
fi

# Count subdirectories with images
SUBDIR_COUNT=$(find "${CONCEPT_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ $SUBDIR_COUNT -eq 0 ]; then
    echo "ERROR: No subdirectories found in ${CONCEPT_DIR}"
    exit 1
fi

# Count total images
IMAGE_COUNT=$(find "${CONCEPT_DIR}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "ERROR: No images found in ${CONCEPT_DIR}"
    exit 1
fi

echo ""
echo "Found ${SUBDIR_COUNT} subdirectories with ${IMAGE_COUNT} total images in ${CONCEPT}"
echo ""

#==============================================================================
# RUN STATISTICS EXTRACTION
#==============================================================================

echo "Starting batch statistics extraction..."
echo "Processing all subdirectories in ${CONCEPT}..."
echo ""

uv run python ${PYTHON_SCRIPT} \
    --parent-dir "${CONCEPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --skip-existing

EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Batch statistics extraction completed successfully"
    echo "  Concept: ${CONCEPT}"
    echo "  Subdirectories processed: ${SUBDIR_COUNT}"
    echo "  Total images: ${IMAGE_COUNT}"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo "  Files: {subdirectory_name}_mean.npy, {subdirectory_name}_cov.npy"
else
    echo "✗ Statistics extraction FAILED (code: ${EXIT_CODE})"
    echo "  Concept: ${CONCEPT}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE