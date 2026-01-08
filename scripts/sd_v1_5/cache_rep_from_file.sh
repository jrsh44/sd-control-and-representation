#!/bin/bash
################################################################################
# SLURM Array Job Script - Cache Representations from Prompt File
#
# Purpose:
#   Generate representation caches from a prompt file (format: prompt_nr;prompt)
#   Split workload across multiple parallel SLURM array jobs
#
# Usage:
#   sbatch scripts/sd_v1_5/cache_rep_from_file.sh
#
# Output:
#   - Cache: {RESULTS_DIR}/{model_name}/{dataset_name}/representations/
#   - Logs: ../logs/rep_gen_from_file_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Adjust based on your dataset size and available resources
#==============================================================================

#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sd_rep_gen_from_file          # Name in queue
#SBATCH --time 0-03:00:00                    # Max 8 hours per task
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 8                  # CPU cores for data processing
#SBATCH --mem 32G                           # 80GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/rep_gen_from_file_%A_%a.log   # %A=job ID, %a=task ID
#SBATCH --array=0-3%4                       # 4 tasks, max 1 running at once

# Optional: email notifications
#SBATCH --mail-user=01180694@pw.edu.pl
#SBATCH --mail-type=FAIL,END

#==============================================================================
# ERROR HANDLING
#==============================================================================

set -eu  # Exit on error or undefined variable

#==============================================================================
# SETUP
#==============================================================================

echo "=========================================="
 echo "Cache Generation Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/mjarosz/sd-control-and-representation
source ./.venv/bin/activate

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# Create directories
mkdir -p "../logs"
mkdir -p ${RESULTS_DIR:-../results}

echo ""
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set}"
echo "  HF_HOME: ${HF_HOME:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/sd_v1_5/cache_rep_from_file.py"

# Dataset configuration
DATASET_NAME="cc3m-wds_fs"

# Model configuration (options: SD_V1_5, FINETUNED_SAURON, SD_V3)
MODEL_NAME="SD_V1_5"

# Prompts configuration
PROMPTS_FILE="data/cc3m-wds/prompts_fs.txt"
NUM_PROMPTS=2000
ARRAY_TOTAL=4

# Layers to capture
LAYERS=(
    "TEXT_EMBEDDING_FINAL"
    "UNET_UP_1_ATT_1"
)

# Generation parameters
GUIDANCE_SCALE=7.5
STEPS=50
SEED=42

# Other settings
SKIP_WANDB=false

LAYERS_STR="${LAYERS[@]}"

echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Model: ${MODEL_NAME}"
echo "  Prompts File: ${PROMPTS_FILE}"
echo "  Total Prompts: ${NUM_PROMPTS}"
echo "  Array Total: ${ARRAY_TOTAL}"
echo "  Prompts per Job: $((${NUM_PROMPTS} / ${ARRAY_TOTAL}))"
echo "  Task ID: ${SLURM_ARRAY_TASK_ID} (of ${ARRAY_TOTAL})"
echo "  Layers: ${LAYERS_STR}"
echo "  Guidance Scale: ${GUIDANCE_SCALE}"
echo "  Steps: ${STEPS}"
echo "  Seed: ${SEED}"
echo "=========================================="
echo ""

#==============================================================================
# GPU CHECK
#==============================================================================

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found, GPU may not be available"
    echo ""
fi

#==============================================================================
# RUN GENERATION
#==============================================================================

echo "Starting generation..."
echo ""

# Build command
CMD="uv run ${PYTHON_SCRIPT} \
    --prompts-file ${PROMPTS_FILE} \
    --dataset-name ${DATASET_NAME} \
    --model-name ${MODEL_NAME} \
    --num-prompts ${NUM_PROMPTS} \
    --array-id ${SLURM_ARRAY_TASK_ID} \
    --array-total ${ARRAY_TOTAL} \
    --layers ${LAYERS_STR} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --steps ${STEPS} \
    --seed ${SEED} \
    --log-images-every 1"

# Add --skip-wandb flag if requested
if [ "${SKIP_WANDB}" = true ]; then
    CMD="${CMD} --skip-wandb"
fi

# Run the command
eval ${CMD}

# Capture exit code
EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Task ${SLURM_ARRAY_TASK_ID} - $(date)"
    echo "  Processed prompts: $((${NUM_PROMPTS} / ${ARRAY_TOTAL}))"
else
    echo "✗ FAILED (exit code: ${EXIT_CODE}) - Task ${SLURM_ARRAY_TASK_ID} - $(date)"
fi

exit $EXIT_CODE
