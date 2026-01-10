#!/bin/bash
################################################################################
# SLURM Job Script - SAE Feature Selection (CC3M-WDS)
#
# Purpose:
#   Compute per-neuron activation sums for CC3M-WDS dataset using id_range filtering
#   Processes remaining 3 chunks (skipping first 115.2M samples already completed)
#
# Usage:
#   sbatch scripts/sae/feature_selection_cc3m-wds.sh
#
# Output:
#   - Feature sums: {FEATURES_DIR}/
#   - Logs: ../logs/sae_select_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name sae_select
#SBATCH --array=0-3  # 4 parallel jobs for 23.04M samples
#SBATCH --time 0-05:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 16
#SBATCH --mem 128G
#SBATCH --partition short
#SBATCH --output ../logs/sae_select_%A_%a.log  # %A=job ID, %a=array task ID

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
echo "SAE Feature Selection - CC3M-WDS"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
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

# Dataset configuration
DATASET_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/cc3m-wds_fs/representations/unet_up_1_att_1"
DATASET_NAME="cc3m-wds"

# Model configuration
LAYER_NAME="unet_up_1_att_1"
SAE_DIR_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096"

# Output configuration
FEATURES_DIR="/mnt/evafs/groups/mi2lab/mjarosz/results/sd_v1_5/sae/cc3m-wds_nudity/unet_up_1_att_1/exp36_topk32_lr1em3_warmup100000_aux00625_ep2_bs4096/feature_sums"

# Index range configuration (23.04M samples split into 4 chunks)
TOTAL_SAMPLES=25600000
CHUNK_SIZE=6400000  # 25.6M / 4 = 6.4M per job

LOWER_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
UPPER_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))

# Processing configuration
# CRITICAL: Larger batch = fewer I/O operations on 1TB network memmap
BATCH_SIZE=32768  # Doubled to reduce I/O frequency (was 16384)

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
echo "  Filter Type: id_range"
echo "  Index Range: [${LOWER_INDEX}, ${UPPER_INDEX})"
echo "  Chunk: $((SLURM_ARRAY_TASK_ID + 1))/4 (${CHUNK_SIZE} samples) [Array Task: ${SLURM_ARRAY_TASK_ID}]"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Features dir: ${FEATURES_DIR}"
echo "=========================================="

# Build command with id_range filtering
CMD="uv run ${PYTHON_SCRIPT} \
    --dataset_path \"${DATASET_PATH}\" \
    --dataset_name \"${DATASET_NAME}\" \
    --sae_dir_path \"${SAE_DIR_PATH}\" \
    --features_dir_path \"${FEATURES_DIR}\" \
    --filter_type id_range \
    --lower_index ${LOWER_INDEX} \
    --upper_index ${UPPER_INDEX} \
    --batch_size ${BATCH_SIZE}"

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
    echo "Feature selection completed successfully"
    echo "  Dataset: ${DATASET_NAME}"
    echo "  Chunk: $((SLURM_ARRAY_TASK_ID + 1))/4 [Array Task: ${SLURM_ARRAY_TASK_ID}]"
    echo "  Index Range: [${LOWER_INDEX}, ${UPPER_INDEX})"
    echo "  Output directory: ${FEATURES_DIR}"
else
    echo "Feature selection FAILED (code: ${EXIT_CODE})"
    echo "  Dataset: ${DATASET_NAME}"
    echo "  Chunk: $((SLURM_ARRAY_TASK_ID + 1))/4 [Array Task: ${SLURM_ARRAY_TASK_ID}]"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE