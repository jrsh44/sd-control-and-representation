#!/bin/bash
################################################################################
# SLURM Array Job Script - SAE Feature Selection
#
# Purpose:
#   Compute per-neuron activation sums for concept detection across multiple datasets
#   Each array task processes one dataset, then merges results at the end
#
# Usage:
#   sbatch scripts/sae/feature_selection.sh
#
# Output:
#   - Partial files: {FEATURES_DIR}/{prefix}_job{N}_{hash}.pt
#   - Merged file: {FEATURES_DIR}/{prefix}.pt
#   - Logs: ../logs/sae_select_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name sae_select
#SBATCH --array=0-2          # 3 tasks: 3 nudity concepts
#SBATCH --time 0-01:30:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH --partition short
#SBATCH --output ../logs/sae_select_%A_%a.log  # %A=job ID, %a=task ID

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
echo "SAE Feature Selection Array Job"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
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

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/sae/feature_selection.py"

# Task 0: cc3m-wds (no filtering)
# Task 1: nudity with 'exposed anus'
# Task 2: nudity with 'exposed feet'
# Task 3: nudity with 'exposed face'

# Arrays for configuration (indexed by SLURM_ARRAY_TASK_ID)
DATASET_PATHS=(
#   "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/cc3m-wds/representations/unet_up_1_att_1"
  "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1"
  "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1"
  "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1"
)

DATASET_NAMES=(
#   "cc3m-wds"
  "nudity"
  "nudity"
  "nudity"
)

# Concept values (empty string for cc3m-wds means no filtering)
CONCEPT_VALUES=(
#   ""
  "exposed anus"
  "exposed armpits"
  "buttocks"
)

# Select configuration for this task
DATASET_PATH="${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}"
DATASET_NAME="${DATASET_NAMES[$SLURM_ARRAY_TASK_ID]}"
CONCEPT_VALUE="${CONCEPT_VALUES[$SLURM_ARRAY_TASK_ID]}"

# Model configuration
LAYER_NAME="unet_up_1_att_1"
SAE_DIR_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/${LAYER_NAME}/exp16_topk32_lr5em5_ep2_bs4096"

# Concept name (same for all tasks with concepts)
CONCEPT_NAME="object"

# Output configuration
FEATURES_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/${LAYER_NAME}/exp16_topk32_lr5em5_ep2_bs4096/feature_sums"

#==============================================================================
# VALIDATION
#==============================================================================
if [ ! -d "$DATASET_PATH" ]; then
  echo "ERROR: Dataset path not found: $DATASET_PATH"
  exit 1
fi

if [ ! -d "$SAE_DIR_PATH" ]; then
  echo "ERROR: SAE directory not found: $SAE_DIR_PATH"
  exit 1
fi

#==============================================================================
# RUN FEATURE SELECTION
#==============================================================================

echo ""
echo "Starting feature selection..."
echo ""

echo "Configuration:"
echo "  Dataset: ${DATASET_PATH}"
echo "  Dataset Name: ${DATASET_NAME}"
echo "  SAE: ${SAE_DIR_PATH}"

# Build command - add concept arguments only if concept_value is not empty
CMD="uv run ${PYTHON_SCRIPT} \
    --dataset_path \"${DATASET_PATH}\" \
    --dataset_name \"${DATASET_NAME}\" \
    --sae_dir_path \"${SAE_DIR_PATH}\" \
    --features_dir_path \"${FEATURES_DIR}\""

if [ -n "$CONCEPT_VALUE" ]; then
    echo "  Concept: '${CONCEPT_NAME}' == '${CONCEPT_VALUE}'"
    CMD="${CMD} --concept \"${CONCEPT_NAME}\" --concept_value \"${CONCEPT_VALUE}\""
else
    echo "  Filtering: None (processing all samples)"
fi

echo "  Features dir: ${FEATURES_DIR}"
echo "=========================================="


# Optional: skip wandb
# CMD="${CMD} --skip-wandb"

echo ""
echo "Running:"
echo "${CMD}"
echo ""

eval ${CMD}

EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Feature selection completed successfully for task ${SLURM_ARRAY_TASK_ID}"
    echo "  Dataset: ${DATASET_NAME}"
    if [ -n "$CONCEPT_VALUE" ]; then
        echo "  Concept: ${CONCEPT_VALUE}"
    fi
    echo "  Output directory: ${FEATURES_DIR}"
else
    echo "Feature selection FAILED (code: ${EXIT_CODE})"
    echo "  Task: ${SLURM_ARRAY_TASK_ID}"
    echo "  Dataset: ${DATASET_NAME}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE