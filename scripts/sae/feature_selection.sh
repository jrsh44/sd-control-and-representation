#!/bin/bash
################################################################################
# SLURM Job Script - SAE Feature Selection (Concept-Based)
# Purpose:
#   Load a trained SAE and compute per-neuron activation difference
#   between samples where a concept is present vs. absent.
#
# Usage:
#   sbatch scripts/sae/select_features.sh
#
# Output:
#   - Scores: results/sae_scores/<layer>_concept_<name>_<value>.npy
#   - Logs: logs/sae_select_<JOB_ID>.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name sae_select
#SBATCH --time 0-24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --mem 64G
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
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
mkdir -p logs

# Load .env
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Environment:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set}"
echo "  HF_HOME: ${HF_HOME:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/sae/feature_selection.py"

# Dataset & SAE
LAYER_NAME="unet_up_1_att_1"
MODEL_NAME="finetuned_sd_saeuron"
# DATASET_PATH="/mnt/evafs/groups/mi2lab/mjarosz/${RESULTS_DIR:-results}/${MODEL_NAME}/cached_representations/${LAYER_NAME}"
DATASET_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/cached_representations/unet_up_1_att_1"
  
TOP_K=32

# Concept to analyze
CONCEPT_NAME="object"        # e.g., object, style, timestep
CONCEPT_VALUE="cats"          # e.g., cat, dog, Impressionism

# SAE model (must exist)
# SAE_DIR="${RESULTS_DIR:-results}/sae_models"
# SAE_PATH="${SAE_DIR}/${LAYER_NAME}_exp16_topk32_lr1em4_epochs5_batch4096.pt"
SAE_PATH="/mnt/evafs/groups/mi2lab/mjarosz/results_npy/finetuned_sd_saeuron/sae/unet_up_1_att_1_sae.pt"

# Output scores
SCORES_DIR="${RESULTS_DIR:-results}/sae_scores"
SCORES_PATH="${SCORES_DIR}/${LAYER_NAME}_concept_${CONCEPT_NAME}_${CONCEPT_VALUE}.npy"

# Runtime
BATCH_SIZE=4096
EPSILON=1e-8

#==============================================================================
# VALIDATION
#==============================================================================
if [ ! -d "$DATASET_PATH" ]; then
  echo "ERROR: Dataset path not found: $DATASET_PATH"
  exit 1
fi

if [ ! -f "$SAE_PATH" ]; then
  echo "ERROR: SAE model not found: $SAE_PATH"
  exit 1
fi

mkdir -p "$SCORES_DIR"

#==============================================================================
# RUN FEATURE SELECTION
#==============================================================================
echo "Starting SAE feature selection..."
echo "=========================================="
echo "Configuration:"
echo "  Dataset: ${DATASET_PATH}"
echo "  SAE: ${SAE_PATH}"
echo "  Concept: '${CONCEPT_NAME}' == '${CONCEPT_VALUE}'"
echo "  Output: ${SCORES_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epsilon: ${EPSILON}"
echo "  Top-k: ${TOP_K}"
echo "=========================================="

CMD="uv run ${PYTHON_SCRIPT} \
    --dataset_path \"${DATASET_PATH}\" \
    --concept \"${CONCEPT_NAME}\" \
    --concept_value \"${CONCEPT_VALUE}\" \
    --sae_path \"${SAE_PATH}\" \
    --feature_scores_path \"${SCORES_PATH}\" \
    --batch_size ${BATCH_SIZE} \
    --epsilon ${EPSILON} \
    --top_k ${TOP_K}"

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
    echo "Feature selection completed"
    echo "Scores: ${SCORES_PATH}"
else
    echo "Feature selection FAILED (code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE