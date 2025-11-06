#!/bin/bash
################################################################################
# SLURM Job Script - SAE Training
# Purpose:
#   Train a Sparse Autoencoder (SAE) on activations and save the model.
#
# Usage:
#   sbatch scripts/test_train_sae.sh
#
# Description:
#   - Runs only once (no array)
#
# Results are saved to:
#   - Model: results/model/sae_model.pt
#   - Logs: logs/train_sae_<JOB_ID>.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account=mi2lab
#SBATCH --job-name=sae_train
#SBATCH --time=0-24:00:00             # Max runtime: 24 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Request 1 GPU (remove if training on CPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --output=logs/train_sae_%j.log   # %j = Job ID

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
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation

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
# Define all parameters you want to test
#==============================================================================

# Python script
PYTHON_SCRIPT="src/tests/test_sae_train.py"

# SAE model parameters
MODEL_TYPE="sd_1_5"
LAYER_NAME="enum_layer_2"

# SAE training parameters
EXPANSION_FACTOR=15
TOP_K=32
LEARNING_RATE=1e-3
NUM_EPOCHS=5
BATCH_SIZE=1024


#==============================================================================
# RUN TRAINING
#==============================================================================
echo "Starting SAE training..."
echo "=========================================="

# Run the training script using uv (recommended) or python directly
uv run python ${PYTHON_SCRIPT} \
    --model_type ${MODEL_TYPE} \
    --layer_name ${LAYER_NAME} \
    --expansion_factor ${EXPANSION_FACTOR} \
    --top_k ${TOP_K} \
    --learning_rate ${LEARNING_RATE} \
    --num_of_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE}

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
