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
#   - Model: {RESULTS_DIR}/{model_name}/sae/{datasets_name}/{layer}/exp{X}_topk{Y}_lr{Z}_ep{E}_bs{B}/model.pt
#   - Config: {RESULTS_DIR}/{model_name}/sae/{datasets_name}/{layer}/exp{X}_topk{Y}_lr{Z}_ep{E}_bs{B}/config.json
#   - Logs: ../logs/sae_train_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sae_train                # Name in queue
#SBATCH --time 0-23:30:00                   # Max 22 hours
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 16                  # CPU cores for data processing
#SBATCH --mem 256G                          # 256GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/sae_train_%A_%a.log  # %A=job ID, %a=task ID
#SBATCH --array=0-5%2                      # Job array for multiple configurations
#SBATCH --nodelist=dgx-1

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
DATASETS_NAME="cc3m-wds_nudity"

# Model configuration
MODEL_NAME="sd_v1_5"

# LAYER_NAME="unet_up_1_att_1"
LAYER_NAME="unet_up_1_att_1"

# Dataset paths (can be multiple, space-separated)
DATASET_PATHS="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/cc3m-wds/representations/unet_up_1_att_1 /mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/nudity/representations/unet_up_1_att_1"

# Validation settings
VALIDATION_PERCENT="10"  # Use N% for validation (set to 0 to disable)
VALIDATION_SEED="42"     # Seed for reproducible splits

# Learning rate scheduler settings
WARMUP_STEPS="100000"           # Number of batches for linear warmup (0 = disabled)
WARMUP_START_FACTOR="0.01" # Starting LR as fraction of base LR (e.g., 0.01 = 1%)
MIN_LR_RATIO="0.1"         # Minimum LR as fraction of base LR after cosine decay

# Auxiliary loss settings
AUX_LOSS_ALPHA="0.03125"   # Weight for auxiliary loss (default: 1/32 = 0.03125)

# Wandb settings
SKIP_WANDB="false"  # Set to "true" to disable wandb logging

#==============================================================================
# CONFIGURATION PROFILES
# Choose a configuration by setting CONFIG_ID (default: 0)

#==============================================================================

# Use SLURM_ARRAY_TASK_ID if running as job array, otherwise use CONFIG_ID
CONFIG_ID=${SLURM_ARRAY_TASK_ID:-${CONFIG_ID:-0}}

echo "Using Configuration ID: ${CONFIG_ID}"

# Define configurations as arrays
# Format: "expansion_factor:top_k:learning_rate:num_epochs:batch_size"

# exp: 8, 16, 24, 32
# top-k: 16, 32, 64
# learning_rate: 1e-3, 1e-4, 1e-5
# scheduler: 5000, 0
# aux_loss_alpha: 0.03125

CONFIGS=(
    "16:32:5e-5:2:4096"
    "16:64:5e-5:2:4096"
    "32:64:5e-5:2:4096"
    "16:32:1e-5:2:4096"
    "16:64:1e-5:2:4096"
    "32:64:1e-5:2:4096"
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
echo "  Warmup Steps: ${WARMUP_STEPS}"
echo "  Warmup Start Factor: ${WARMUP_START_FACTOR}"
echo "  Min LR Ratio: ${MIN_LR_RATIO}"
echo "  Aux Loss Alpha: ${AUX_LOSS_ALPHA}"
echo ""

# Compute SAE output directory and paths
# Structure: {RESULTS_DIR}/{model}/sae/{datasets_name}/{layer}/exp{X}_topk{Y}_lr{Z}_ep{E}_bs{B}/
LEARNING_RATE_STR=$(echo "${LEARNING_RATE}" | sed 's/e-/em/g' | sed 's/e+/ep/g' | sed 's/\.//g')
CONFIG_NAME="exp${EXPANSION_FACTOR}_topk${TOP_K}_lr${LEARNING_RATE_STR}_ep${NUM_EPOCHS}_bs${BATCH_SIZE}"
SAE_DIR="${RESULTS_DIR:-results}/${MODEL_NAME}/sae/${DATASETS_NAME}/${LAYER_NAME}/${CONFIG_NAME}"
SAE_PATH="${SAE_DIR}/model.pt"
CONFIG_PATH="${SAE_DIR}/config.json"

#==============================================================================
# RUN TRAINING
#==============================================================================

echo "Starting training..."
echo ""

echo "Configuration ID: ${CONFIG_ID}"
echo "Configuration:"
echo "  Datasets name: ${DATASETS_NAME}"
echo "  Dataset path(s): ${DATASET_PATHS}"
echo "  Validation: ${VALIDATION_PERCENT}% split"
echo "  Cache format: Memmap"
echo "  SAE directory: ${SAE_DIR}"
echo "  SAE model: ${SAE_PATH}"
echo "  SAE config: ${CONFIG_PATH}"
echo "  Expansion factor: ${EXPANSION_FACTOR}"
echo "  Top-K: ${TOP_K}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Warmup start factor: ${WARMUP_START_FACTOR}"
echo "  Min LR ratio: ${MIN_LR_RATIO}"
echo "  Aux loss alpha: ${AUX_LOSS_ALPHA}"
echo "  Skip wandb: ${SKIP_WANDB}"
echo "=========================================="

# Build command
CMD="uv run ${PYTHON_SCRIPT} \
    --dataset_paths ${DATASET_PATHS} \
    --sae_path ${SAE_PATH} \
    --config_path ${CONFIG_PATH} \
    --datasets_name ${DATASETS_NAME} \
    --expansion_factor ${EXPANSION_FACTOR} \
    --top_k ${TOP_K} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --validation_percent ${VALIDATION_PERCENT} \
    --validation_seed ${VALIDATION_SEED} \
    --warmup_steps ${WARMUP_STEPS} \
    --warmup_start_factor ${WARMUP_START_FACTOR} \
    --min_lr_ratio ${MIN_LR_RATIO} \
    --aux_loss_alpha ${AUX_LOSS_ALPHA}"

# Add --skip-wandb flag if enabled
if [ "${SKIP_WANDB}" = "true" ]; then
    CMD="${CMD} --skip-wandb"
fi

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