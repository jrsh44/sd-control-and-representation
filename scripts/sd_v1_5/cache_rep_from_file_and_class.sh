#!/bin/bash
################################################################################
# SLURM Array Job Script - Cache Representations with Class Templates
#
# Purpose:
#   Generate representation caches by filling class templates with class names
#   Each array task processes a different batch of classes
#
# Usage:
#   sbatch scripts/sd_v1_5/cache_rep_from_file_and_class.sh
#
# Output:
#   - Cache: {RESULTS_DIR}/{model_name}/{dataset_name}/representations/
#   - Logs: ../logs/sd_rep_gen_from_file_and_class_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Adjust based on your dataset size and available resources
#==============================================================================

#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name sd_rep_gen_from_file_and_class            # Name in queue
#SBATCH --time 0-18:00:00                    # Max 3 hours per task
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 8                  # CPU cores for data processing
#SBATCH --mem 64G                           # 64GB RAM (for large batches)
#SBATCH --partition short,experimental                   # Queue name
#SBATCH --output ../logs/sd_rep_gen_from_file_and_class_%A_%a.log   # %A=job ID, %a=task ID
#SBATCH --array=0-3%4                       # 4 tasks, max 2 running at once

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
PYTHON_SCRIPT="scripts/sd_v1_5/cache_rep_from_file_and_class.py"

# Dataset configuration
DATASET_NAME="nudity"

# Model configuration (options: SD_V1_5, FINETUNED_SAURON, SD_V3)
MODEL_NAME="SD_V1_5"

# Prompts configuration
BASE_PROMPTS="data/nudity/prompts.txt"

# Classes to process
CLASSES_BATCH_0=(
    "exposed anus"
    "exposed armpits"
    "belly"
    "exposed belly"
)

CLASSES_BATCH_1=(
    "buttocks"
    "exposed buttocks"
    "female face"
    "male face"
)

CLASSES_BATCH_2=(
    "feet"
    "exposed feet"
    "breast"
    "exposed breast"
)

CLASSES_BATCH_3=(
    "vagina"
    "exposed vagina"
    "male breast"
    "exposed penis"
)

# Layers to capture
LAYERS=(
    "TEXT_EMBEDDING_FINAL"
    "UNET_UP_1_ATT_1"
)

# Generation parameters
GUIDANCE_SCALE=7.5
NUM_STEPS=50
SEED=42

# Other settings
SKIP_WANDB=false

# Log images every N steps
LOG_IMAGES_EVERY=1

#==============================================================================
# TASK MAPPING
# Map task ID to classes batch
#==============================================================================

# Get classes for this task
case ${SLURM_ARRAY_TASK_ID} in
    0)
        CURRENT_CLASSES=("${CLASSES_BATCH_0[@]}")
        ;;
    1)
        CURRENT_CLASSES=("${CLASSES_BATCH_1[@]}")
        ;;
    2)
        CURRENT_CLASSES=("${CLASSES_BATCH_2[@]}")
        ;;
    3)
        CURRENT_CLASSES=("${CLASSES_BATCH_3[@]}")
        ;;
    *)
        echo "ERROR: Invalid task ID ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Convert layers array to space-separated string
LAYERS_STR="${LAYERS[@]}"

echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Model: ${MODEL_NAME}"
echo "  Prompts file: ${BASE_PROMPTS}"
echo "  Classes: ${CURRENT_CLASSES[@]}"
echo "  Layers: ${LAYERS_STR}"
echo "  Guidance Scale: ${GUIDANCE_SCALE}"
echo "  Steps: ${NUM_STEPS}"
echo "  Seed: ${SEED}"
echo "  Log images every: ${LOG_IMAGES_EVERY}"
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

# Build command with classes
CMD="uv run ${PYTHON_SCRIPT} \
    --dataset-name ${DATASET_NAME} \
    --prompts-file ${BASE_PROMPTS} \
    --model-name ${MODEL_NAME} \
    --classes"

# Add each class as a separate argument
for class in "${CURRENT_CLASSES[@]}"; do
    CMD="${CMD} \"${class}\""
done

# Add remaining parameters
CMD="${CMD} \
    --layers ${LAYERS_STR} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --steps ${NUM_STEPS} \
    --seed ${SEED} \
    --log-images-every ${LOG_IMAGES_EVERY}"

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
    echo "  Processed classes: ${CURRENT_CLASSES[@]}"
else
    echo "✗ FAILED (exit code: ${EXIT_CODE}) - Task ${SLURM_ARRAY_TASK_ID} - $(date)"
fi

exit $EXIT_CODE
