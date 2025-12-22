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
#SBATCH --array=0-1          # Array size matches number of datasets
#SBATCH --time 0-23:30:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 256G
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

# Array of dataset paths to process
DATASET_PATHS=(
  "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1"
  "/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/cc3m-wds/representations/unet_up_1_att_1"
)

# Select dataset for this task
DATASET_PATH="${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}"

# Extract dataset name from path (e.g., "nudity" or "cc3m-wds")
DATASET_NAME=$(basename $(dirname $(dirname "$DATASET_PATH")))

# Model configuration
LAYER_NAME="unet_up_1_att_1"
SAE_DIR_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/${LAYER_NAME}/exp16_topk32_lr5em5_ep2_bs4096"

# Concept configuration
CONCEPT_NAME="object"
CONCEPT_VALUE="exposed anus"

# Output configuration
FEATURES_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/${LAYER_NAME}/exp16_topk32_lr5em5_ep2_bs4096/feature_sums"
FEATURE_SUMS_PREFIX="${LAYER_NAME}_concept_${CONCEPT_NAME}_exposed_anus"

# Generation parameters
# TOP_K=32
# BATCH_SIZE=8192
# EPSILON=1e-8

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

mkdir -p "$FEATURES_DIR"

#==============================================================================
# RUN FEATURE SELECTION
#==============================================================================

echo "Starting feature selection for dataset: ${DATASET_NAME}..."
echo ""

echo "Configuration:"
echo "  Dataset: ${DATASET_PATH}"
echo "  SAE: ${SAE_DIR_PATH}"
echo "  Concept: '${CONCEPT_NAME}' == '${CONCEPT_VALUE}'"
echo "  Features dir: ${FEATURES_DIR}"
echo "  Feature prefix: ${FEATURE_SUMS_PREFIX}"
echo "=========================================="

CMD="uv run ${PYTHON_SCRIPT} \
    --dataset_path \"${DATASET_PATH}\" \
    --dataset_name \"${DATASET_NAME}\" \
    --concept \"${CONCEPT_NAME}\" \
    --concept_value \"${CONCEPT_VALUE}\" \
    --sae_dir_path \"${SAE_DIR_PATH}\" \
    --features_dir_path \"${FEATURES_DIR}\" \
    --feature_sums_prefix \"${FEATURE_SUMS_PREFIX}\" "


# Optional: skip wandb
# CMD="${CMD} --skip-wandb"

echo ""
echo "Running:"
echo "${CMD}"
echo ""

eval ${CMD}
EXIT_CODE=$?

#==============================================================================
# MERGE RESULTS (only run on task 0 after all tasks complete)
#==============================================================================
if [ $EXIT_CODE -eq 0 ] && [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Waiting for all array tasks to complete..."
    echo "=========================================="
    
    # Wait for all tasks to finish
    # This simple approach: keep checking if all partial files exist
    EXPECTED_FILES=${#DATASET_PATHS[@]}
    
    while true; do
        COMPLETED=$(ls ${FEATURES_DIR}/${FEATURE_SUMS_PREFIX}_job*_*.pt 2>/dev/null | wc -l)
        if [ $COMPLETED -ge $EXPECTED_FILES ]; then
            echo "All tasks completed. Starting merge..."
            break
        fi
        echo "  Waiting... ($COMPLETED/$EXPECTED_FILES files ready)"
        sleep 30
    done
    
    # Run merge script
    MERGE_SCRIPT="scripts/data/merge_feature_sums.py"
    MERGE_INPUT_PATTERN="${FEATURES_DIR}/${FEATURE_SUMS_PREFIX}_job*.pt"
    MERGE_OUTPUT_PATH="${FEATURES_DIR}/${FEATURE_SUMS_PREFIX}.pt"
    
    echo ""
    echo "Merging partial results..."
    MERGE_CMD="uv run ${MERGE_SCRIPT} \
        --input_pattern \"${MERGE_INPUT_PATTERN}\" \
        --output_path \"${MERGE_OUTPUT_PATH}\""
    
    # Note: NOT using --cleanup to validate results
    
    echo "Running:"
    echo "${MERGE_CMD}"
    echo ""
    
    eval ${MERGE_CMD}
    MERGE_EXIT_CODE=$?
    
    if [ $MERGE_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Merge completed successfully"
        echo "  Output: ${MERGE_OUTPUT_PATH}"
        echo "  Partial files kept for validation"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "✗ Merge FAILED (code: ${MERGE_EXIT_CODE})"
        echo "=========================================="
        EXIT_CODE=$MERGE_EXIT_CODE
    fi
fi

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
        echo "Feature selection and merge completed successfully"
        echo "Final output: ${MERGE_OUTPUT_PATH}"
    else
        echo "Feature selection completed for task ${SLURM_ARRAY_TASK_ID}"
    fi
else
    echo "Feature selection FAILED (code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE