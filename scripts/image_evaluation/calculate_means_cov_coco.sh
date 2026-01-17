#!/bin/bash
################################################################################
# SLURM Job Script - COCO Statistics Extraction
#
# Purpose:
#   Extract and save Inception v3 statistics from COCO-2017 validation set
#   for FID calculation
#
# Usage:
#   sbatch scripts/image_evaluation/calculate_means_cov_coco.sh
#
# Output:
#   - Mean and covariance matrices saved as .npy files
#   - Logs: logs/coco_stats_%j.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name coco_stats
#SBATCH --time 0-04:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH --partition short
#SBATCH --output logs/coco_stats_%j.log  # %j=job ID

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
echo "COCO Statistics Extraction"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
source .venv/bin/activate
source ./.venv/bin/activate

# Create directories
mkdir -p logs

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Environment Configuration:"
echo "  Python: $(which python)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/image_evaluation/calcualte_means_cov_coco.py"

# Output directory for statistics
OUTPUT_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/coco"

# Number of COCO images to process
NUM_IMAGES=40000

# Classes to filter (space-separated)
CLASSES="person"

# Device (cuda or cpu)
DEVICE="cpu"

#==============================================================================
# VALIDATION
#==============================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "ERROR: Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

#==============================================================================
# RUN COCO STATISTICS EXTRACTION
#==============================================================================

echo ""
echo "Starting COCO statistics extraction..."
echo ""

echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of images: ${NUM_IMAGES}"
echo "  Classes: ${CLASSES}"
echo "  Device: ${DEVICE}"
echo "=========================================="

# Build and run command
echo ""
echo "Running:"
echo "uv run python ${PYTHON_SCRIPT} \\"
echo "    --output-dir ${OUTPUT_DIR} \\"
echo "    --num-images ${NUM_IMAGES} \\"
echo "    --classes ${CLASSES} \\"
echo "    --device ${DEVICE}"
echo ""

uv run python ${PYTHON_SCRIPT} \
    --output-dir ${OUTPUT_DIR} \
    --num-images ${NUM_IMAGES} \
    --classes ${CLASSES}

EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ COCO statistics extraction completed successfully"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo "  Files created:"
    echo "    - coco2017_val_${NUM_IMAGES}_${CLASSES}_mean.npy"
    echo "    - coco2017_val_${NUM_IMAGES}_${CLASSES}_cov.npy"
else
    echo "✗ COCO statistics extraction FAILED (code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE