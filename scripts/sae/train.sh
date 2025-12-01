#!/bin/bash
################################################################################
# SLURM Array Job Script - Train Sparse Autoencoder
#
# Purpose:
#   Train SAE on cached representations with configurable hyperparameters
#   Each array task uses different expansion factor, top-k, and learning rate
#
# Usage:
#   sbatch scripts/sae/train.sh
#
# Output:
#   - Model: {RESULTS_DIR}/{model_name}/sae/{layer}/trained_models/exp{X}_topk{Y}_lr{Z}_epochs{E}_batch{B}.pt
#   - Logs: ../logs/sae_train_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sae_train                # Name in queue
#SBATCH --time 0-6:00:00                   # Max 6 hours
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 12                  # CPU cores for data processing
#SBATCH --mem 100G                          # 100GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/sae_train_%A_%a.log  # %A=job ID, %a=task ID
#SBATCH --array=0-11%3                       # Job array for multiple configurations


# Optional email notification
# #SBATCH --mail-user=01180694@pw.edu.pl
# #SBATCH --mail-type=FAIL,END

#==============================================================================
# ERROR HANDLING
#==============================================================================
set -eu  # Exit on error or undefined variable

#==============================================================================
# SETUP
#==============================================================================

echo "=========================================="
echo "SAE Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/mjarosz/sd-control-and-representation
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
PYTHON_SCRIPT="scripts/sae/train.py"

# Dataset configuration
DATASET_NAME="unlearn_canvas"

# Model configuration
MODEL_NAME="finetuned_sd_saeuron"
LAYER_NAME="unet_up_1_att_1"

# Dataset paths
TRAIN_DATASET_PATH="${RESULTS_DIR:-results}/${MODEL_NAME}/${DATASET_NAME}/representations/train/${LAYER_NAME}"
VALIDATION_DATASET_PATH="${RESULTS_DIR:-results}/${MODEL_NAME}/${DATASET_NAME}/representations/validation/${LAYER_NAME}"
# VALIDATION_DATASET_PATH=""  # Uncomment to disable validation

#==============================================================================
# CONFIGURATION PROFILES
# Choose a configuration by setting CONFIG_ID (default: 0)

#==============================================================================

# Use SLURM_ARRAY_TASK_ID if running as job array, otherwise use CONFIG_ID
CONFIG_ID=${SLURM_ARRAY_TASK_ID:-${CONFIG_ID:-0}}

echo "Using Configuration ID: ${CONFIG_ID}"

# Define configurations as arrays
# Format: "expansion_factor:top_k:learning_rate:num_epochs:batch_size"
CONFIGS=(
    "8:16:4e-4:5:4096"
    "16:32:4e-4:5:4096"
    "16:64:4e-4:5:4096"
    "32:64:4e-4:5:4096"
    "8:16:1e-4:5:4096"
    "16:32:1e-4:5:4096"
    "16:64:1e-4:5:4096"
    "32:64:1e-4:5:4096"
    "8:16:1e-5:5:4096"
    "16:32:1e-5:5:4096"
    "16:64:1e-5:5:4096"
    "32:64:1e-5:5:4096"
)

# Validate CONFIG_ID
if [ ${CONFIG_ID} -ge ${#CONFIGS[@]} ]; then
    echo "ERROR: CONFIG_ID=${CONFIG_ID} is out of range (max: $((${#CONFIGS[@]} - 1)))"
    exit 1
fi

# Parse selected configuration
IFS=':' read -r EXPANSION_FACTOR TOP_K LEARNING_RATE NUM_EPOCHS BATCH_SIZE <<< "${CONFIGS[$CONFIG_ID]}"

echo ""
echo "Selected Configuration [${CONFIG_ID}]:"
echo "  Expansion Factor: ${EXPANSION_FACTOR}"
echo "  Top-K: ${TOP_K}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Num Epochs: ${NUM_EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo ""

# Compute SAE path
SAE_DIR="${RESULTS_DIR:-results}/${MODEL_NAME}/sae/${LAYER_NAME}/trained_models"
LEARNING_RATE_STR=$(echo "${LEARNING_RATE}" | sed 's/e-/em/g' | sed 's/e+/ep/g' | sed 's/\.//g')
SAE_PATH="${SAE_DIR}/exp${EXPANSION_FACTOR}_topk${TOP_K}_lr${LEARNING_RATE_STR}_epochs${NUM_EPOCHS}_batch${BATCH_SIZE}.pt"

#==============================================================================
# RUN TRAINING
#==============================================================================

echo "Starting training..."
echo ""

echo "Configuration ID: ${CONFIG_ID}"
echo "Configuration:"
echo "  Train dataset: ${TRAIN_DATASET_PATH}"
echo "  Validation dataset: ${VALIDATION_DATASET_PATH:-None (training without validation)}"
echo "  Cache format: Memmap"
echo "  SAE path: ${SAE_PATH}"
echo "  Expansion factor: ${EXPANSION_FACTOR}"
echo "  Top-K: ${TOP_K}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "=========================================="

# Build command with required arguments
CMD="uv run ${PYTHON_SCRIPT} \
    --train_dataset_path ${TRAIN_DATASET_PATH} \
    --sae_path ${SAE_PATH} \
    --dataset_name ${DATASET_NAME} \
    --expansion_factor ${EXPANSION_FACTOR} \
    --top_k ${TOP_K} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE}"

# Add optional validation dataset if provided
if [ -n "${VALIDATION_DATASET_PATH}" ] && [ "${VALIDATION_DATASET_PATH}" != "" ]; then
    CMD="${CMD} --test_dataset_path ${VALIDATION_DATASET_PATH}"
    echo "Validation: ENABLED"
else
    echo "Validation: DISABLED"
fi

# Add --skip-wandb flag if you want to skip wandb logging
# CMD="${CMD} --skip-wandb"

echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Execute the command
eval ${CMD}

# Capture exit code
EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SAE training completed successfully"
else
    echo "✗ SAE training FAILED (exit code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE