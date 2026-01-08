#!/bin/bash
################################################################################
# SLURM Array Job Script - Unlearned Representation Cache Generation
# Purpose: Generate unlearned cached representations for multiple styles in parallel
#
# Usage:
#   sbatch scripts/sd_v1_5/generate_unlearned_cache_batch.sh
#
# Description:
#   This script runs parallel tasks to generate unlearned representation caches:
#   - Each task processes one artistic style
#   - Uses SAE for unlearning specific concepts
#   - Captures multiple layer representations from SD 1.5
#   - Saves to memmap format for 100x faster loading
#   - Results: {RESULTS_DIR}/{model_name}/cached_representations/{layer_name}/
#   - Logs: ../logs/unlearned_cache_gen_{JOB_ID}_{TASK_ID}.log
#
#   Memmap cache files generated:
#   - data.npy: memmap array of representations
#   - metadata.pkl: full metadata with prompts
#   - index.json: lightweight metadata for fast filtering
#   - info.json: dataset info
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Adjust based on dataset size and resources
#==============================================================================

#SBATCH --account mi2lab                    # Your compute account
#SBATCH --job-name unlearned_cache_gen      # Name in queue
#SBATCH --time 0-6:00:00                    # Max 6 hours per style
#SBATCH --nodes 1                           # One node per task
#SBATCH --ntasks-per-node 1                 # One task per node
#SBATCH --gres gpu:1                        # One GPU (required for SD)
#SBATCH --cpus-per-task 12                  # CPU cores for data processing
#SBATCH --mem 64G                           # 64GB RAM (for large batches)
#SBATCH --partition short                   # Queue name
#SBATCH --output ../logs/unlearned_cache_gen_%A_%a.log   # %A=job ID, %a=task ID
#SBATCH --array 0-1%2                       # 2 styles, max 2 running at once

# Optional: email notifications
#SBATCH --mail-user 01180694@pw.edu.pl
#SBATCH --mail-type FAIL,END

#==============================================================================
# ERROR HANDLING
#==============================================================================

set -eu  # Exit on error or undefined variable

#==============================================================================
# SETUP
#==============================================================================

echo "=========================================="
echo "Unlearned Cache Generation Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation

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
PYTHON_SCRIPT="scripts/sd_v1_5/generate_unlearned_cache_batch.py"

# Prompts directory
PROMPTS_DIR="data/unlearn_canvas/prompts/test"

# Styles to process
STYLES=(
    "Surrealism"
    "Impressionism"
)

# SAE and unlearning parameters
SAE_PATH="results/sae/unet_up_1_att_1_sae.pt"
CONCEPT_MEANS_PATH="results/means/concept_means.pt"
INFLUENCE_FACTOR=30.0
FEATURES_NUMBER=2
EPSILON=1e-8
IGNORE_MODIFICATION=false

# Layers to capture
LAYERS=(
    "TEXT_EMBEDDING_FINAL"
    "UNET_UP_1_ATT_1"
    "UNET_MID_RES_1"
)

# Generation parameters
GUIDANCE_SCALE=7.5
NUM_STEPS=50
SEED=42

# WandB settings
SKIP_WANDB=false

#==============================================================================
# TASK MAPPING
#==============================================================================

# Get style for this task
CURRENT_STYLE=${STYLES[$SLURM_ARRAY_TASK_ID]}

# Convert layers array to space-separated string
LAYERS_STR="${LAYERS[@]}"

echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Style: ${CURRENT_STYLE}"
echo "  Prompts: ${PROMPTS_DIR}"
echo "  SAE: ${SAE_PATH}"
echo "  Means: ${CONCEPT_MEANS_PATH}"
echo "  Factor: ${INFLUENCE_FACTOR}"
echo "  Features: ${FEATURES_NUMBER}"
echo "  Epsilon: ${EPSILON}"
echo "  Layers: ${LAYERS_STR}"
echo "  Guidance: ${GUIDANCE_SCALE}"
echo "  Steps: ${NUM_STEPS}"
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

echo "Starting unlearned cache generation for style: ${CURRENT_STYLE}"
echo ""

# Build command
CMD="uv run ${PYTHON_SCRIPT} \
    --prompts_dir ${PROMPTS_DIR} \
    --layers ${LAYERS_STR} \
    --sae_path ${SAE_PATH} \
    --concept_means_path ${CONCEPT_MEANS_PATH} \
    --influence_factor ${INFLUENCE_FACTOR} \
    --features_number ${FEATURES_NUMBER} \
    --epsilon ${EPSILON} \
    --ignore_modification ${IGNORE_MODIFICATION} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --steps ${NUM_STEPS} \
    --seed ${SEED}"

# Add --style flag only if not empty
if [ -n "${CURRENT_STYLE}" ]; then
    CMD="${CMD} --style ${CURRENT_STYLE}"
fi

# Add --skip_wandb if true
if [ "${SKIP_WANDB}" = true ]; then
    CMD="${CMD} --skip_wandb"
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
    echo "✓ SUCCESS - Style: ${CURRENT_STYLE} - $(date)"
else
    echo "✗ FAILED (exit code: ${EXIT_CODE}) - Style: ${CURRENT_STYLE} - $(date)"
fi

exit $EXIT_CODE