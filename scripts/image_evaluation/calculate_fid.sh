#!/bin/bash
################################################################################
# SLURM Job Script - FID Score Calculation for Multiple Concepts (Array Job)
#
# Purpose:
#   Calculate FID scores for multiple nudity concepts against COCO-2017
#   Uses SLURM array jobs to process concepts in parallel
#
# Usage:
#   sbatch scripts/image_evaluation/calculate_fid.sh
#
# Output:
#   - FID results saved to CSV files for each concept
#   - Logs: logs/fid_job_{JOB_ID}_{ARRAY_TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name fid_calc
#SBATCH --array=0-7
#SBATCH --time 0-01:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --partition short
#SBATCH --output logs/fid_job_%A_%a.log  # %A=array job ID, %a=task ID

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
echo "FID Score Calculation Job (Array Task)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-no_slurm_job_id}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-no_task_id}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
source .venv/bin/activate

# Create directories
mkdir -p logs

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/image_evaluation/calculate_fid.py"

# Base paths (adjust these as needed)
COCO_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/coco"
BASE_PARENT_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/nudity_per_timestep"
OUTPUT_BASE_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/fid/scores"

# Concepts to process
CONCEPTS=(
    "exposed_vagina"
    "exposed_anus"
    "exposed_buttocks"
    "exposed_breast"
    "exposed_belly"
    "exposed_armpits"
    "exposed_penis"
)

# Optional: Set to 1 to enable per-timestep flag
PER_TIMESTEP=1

#==============================================================================
# GET CONCEPT FOR THIS ARRAY TASK
#==============================================================================

# Get the concept for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CONCEPT="${CONCEPTS[$TASK_ID]}"

echo ""
echo "Array Task Configuration:"
echo "  Task ID: ${TASK_ID}"
echo "  Concept: ${CONCEPT}"
echo "  Total concepts: ${#CONCEPTS[@]}"
echo ""

#==============================================================================
# VALIDATION
#==============================================================================
if [ ! -d "$COCO_DIR" ]; then
  echo "ERROR: COCO directory not found: $COCO_DIR"
  exit 1
fi

if [ ! -d "$BASE_PARENT_DIR" ]; then
  echo "ERROR: Base parent directory not found: $BASE_PARENT_DIR"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

#==============================================================================
# BUILD PATHS FOR THIS CONCEPT
#==============================================================================

PARENT_DIR="${BASE_PARENT_DIR}/${CONCEPT}"
OUTPUT_FILE="${OUTPUT_BASE_DIR}/${CONCEPT}_pt_fid.csv"

# Check if concept directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "ERROR: Directory not found for concept '${CONCEPT}': ${PARENT_DIR}"
    exit 1
fi

echo "Configuration:"
echo "  COCO dir: ${COCO_DIR}"
echo "  Parent dir: ${PARENT_DIR}"
echo "  Output file: ${OUTPUT_FILE}"
echo "  Per-timestep: ${PER_TIMESTEP}"
echo ""

#==============================================================================
# RUN FID CALCULATION
#==============================================================================

echo "=========================================="
echo "Processing concept: ${CONCEPT}"
echo "=========================================="

# Build command
CMD="python ${PYTHON_SCRIPT} \
    --coco-dir \"${COCO_DIR}\" \
    --parent-dir \"${PARENT_DIR}\" \
    --output \"${OUTPUT_FILE}\""

# Add per-timestep flag if enabled
if [ "$PER_TIMESTEP" -eq 1 ]; then
    CMD="${CMD} --per-timestep"
fi

echo "Running:"
echo "${CMD}"
echo ""

# Run the command
eval ${CMD}
EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully processed: ${CONCEPT}"
    echo "  Results saved to: ${OUTPUT_FILE}"
else
    echo "✗ Failed to process: ${CONCEPT} (exit code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE