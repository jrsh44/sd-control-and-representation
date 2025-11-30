#!/bin/bash
################################################################################
# SLURM Array Job Script - Prompt File Representation Generation
# Purpose: Generate cached representations from prompt file in parallel
#
# Usage:
#   sbatch scripts/sd_v1_5/cache_rep_from_file.sh
#
# Description:
#   This script runs parallel tasks to generate representation caches from
#   a prompt file with format: prompt_nr;prompt
#
#   Results are saved to:
#   - Cache: {RESULTS_DIR}/{model_name}/cached_representations/cc3m-wds/{layer_name}/
#   - Logs: ../logs/cache_generation_{JOB_ID}_{TASK_ID}.log
#
#   Memmap cache files generated:
#   - data.npy: memmap array of representations
#   - metadata.pkl: full metadata with prompts
#   - index.json: lightweight metadata for fast filtering
#   - info.json: dataset info
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Adjust based on your dataset size and available resources
#==============================================================================

#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sd_rep_gen_from_file          # Name in queue
#SBATCH --time 0-6:00:00                    # Max 6 hours per task
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 6                  # CPU cores for data processing
#SBATCH --mem 64G                           # 64GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/rep_gen_from_file_%A_%a.log   # %A=job ID, %a=task ID
#SBATCH --array=0-8%3                       # 9 tasks, max 3 running at once

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
echo "Prompt File Cache Generation Job"
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

# Prompts configuration
PROMPTS_FILE="data/cc3m-wds/train.txt"
DATASET_NAME="cc3m-wds"
NUM_PROMPTS=36000
ARRAY_TOTAL=9

# Layers to capture
LAYERS=(
    "TEXT_EMBEDDING_FINAL"
    "UNET_UP_1_ATT_1"
)

# Generation parameters
GUIDANCE_SCALE=7.5
STEPS=50
SEED=42

SKIP_WANDB=false

LAYERS_STR="${LAYERS[@]}"

echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Prompts File: ${PROMPTS_FILE}"
echo "  Dataset Name: ${DATASET_NAME}"
echo "  Total Prompts: ${NUM_PROMPTS}"
echo "  Array Total: ${ARRAY_TOTAL}"
echo "  Prompts per Job: $((${NUM_PROMPTS} / ${ARRAY_TOTAL}))"
echo "  Task ID: ${SLURM_ARRAY_TASK_ID} (of ${ARRAY_TOTAL})"
echo "  Layers: ${LAYERS_STR}"
echo "  Cache Format: Memmap"
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
# RUN CACHE GENERATION
#==============================================================================

echo "Starting cache generation..."
echo ""

# Build command
CMD="uv run ${PYTHON_SCRIPT} \
    --prompts-file ${PROMPTS_FILE} \
    --dataset-name ${DATASET_NAME} \
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
