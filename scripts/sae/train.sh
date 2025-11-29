#!/bin/bash
################################################################################
# SLURM Job Script - SAE Training
# Purpose:
#   Train a Sparse Autoencoder on cached representations and save the model.
#
# Usage:
#   sbatch scripts/sae/train.sh
#
# Description:
#   - Trains SAE on cached SD layer representations
#   - Uses memmap format for fast data loading
#   - Optional validation dataset
#
# Results are saved to:
#   - Model: results/sae_models/<layer_name>_sae.pt
#   - Logs: logs/sae_train_<JOB_ID>.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sae_train                # Name in queue
#SBATCH --time 0-18:00:00                   # Max 18 hours
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 32                  # CPU cores for data processing
#SBATCH --mem 100G                          # 100GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/sae_train_%A_%a.log  # %A=job ID, %a=task ID


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
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to your project directory
cd /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation

# Create required directories
mkdir -p logs

# Load environment variables from .env if present
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# Display environment summary
echo ""
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set (will use default)}"
echo "  HF_HOME: ${HF_HOME:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
# Define all parameters for SAE training
#==============================================================================

# Python script (updated path)
PYTHON_SCRIPT="scripts/sae/train.py"

# Dataset paths - Update these to your cached representation directories
LAYER_NAME="unet_up_1_att_1"
MODEL_NAME="finetuned_sd_saeuron"
TRAIN_DATASET_PATH="${RESULTS_DIR:-results}/${MODEL_NAME}/${DATASET_NAME}/representations/train/${LAYER_NAME}"

# Dataset name (used for WandB logging and organization)
DATASET_NAME="unlearn_canvas"

# Optional: Validation dataset (comment out or leave empty to train without validation)
VALIDATION_DATASET_PATH="${RESULTS_DIR:-results}/${MODEL_NAME}/${DATASET_NAME}/representations/validation/${LAYER_NAME}"
# VALIDATION_DATASET_PATH=""  # Uncomment to disable validation

# SAE model parameters
EXPANSION_FACTOR=16
TOP_K=32
LEARNING_RATE=4e-4
NUM_EPOCHS=5
BATCH_SIZE=4096

# Compute SAE path
SAE_DIR="${RESULTS_DIR:-results}/sae_models"
LEARNING_RATE_STR=$(echo "${LEARNING_RATE}" | sed 's/e-/em/g' | sed 's/e+/ep/g' | sed 's/\.//g')
SAE_PATH="${SAE_DIR}/${LAYER_NAME}_exp${EXPANSION_FACTOR}_topk${TOP_K}_lr${LEARNING_RATE_STR}_epochs${NUM_EPOCHS}_batch${BATCH_SIZE}.pt"

#==============================================================================
# RUN TRAINING
#==============================================================================
echo "Starting SAE training..."
echo "=========================================="
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