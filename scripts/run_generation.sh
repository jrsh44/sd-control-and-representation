#!/bin/bash
################################################################################
# SLURM Array Job Script - Style-based Representation Generation
# Purpose: Generate cached representations for multiple styles in parallel
#
# Usage:
#   sbatch scripts/run_generation.sh
#
# Description:
#   Runs parallel tasks (1-60) to generate representations:
#   - Each task handles a different artistic style
#   - Maximum 5 tasks running simultaneously
#   - Uses sd_1_5_generate_cachev2.py for generation
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================

#SBATCH --account mi2lab               # Your compute account
#SBATCH --job-name sd_rep_gen          # Job name
#SBATCH --time 0-02:00:00              # Max runtime (1 day)
#SBATCH --nodes 1                      # Number of nodes
#SBATCH --ntasks-per-node 1            # Tasks per node
#SBATCH --gres gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task 12              # CPU cores per task
#SBATCH --mem 32G                      # RAM per node
#SBATCH --partition short              # Queue name
#SBATCH --output logs/rep_gen_%A_%a.log# %A=job ID, %a=task ID
#SBATCH --array=1-2%2                  # 2 tasks, max 2 running at once

#==============================================================================
# ERROR HANDLING
#==============================================================================

set -eu  # Exit on error and undefined variables

#==============================================================================
# CONFIGURATION
#==============================================================================

# Project directory
PROJECT_DIR="/mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation"
cd "${PROJECT_DIR}"

# Create logs if needed
mkdir -p logs

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Python script configuration
PYTHON_SCRIPT="src/sd_1_5_generate_cachev2.py"
PROMPTS_DIR="data/unlearn_canvas/prompts/test"
LAYERS=(
    "TEXT_EMBEDDING_FINAL"
    "UNET_MID_ATT"
    "UNET_DOWN_1_ATT_1"
    "UNET_DOWN_2_ATT_1"
    "UNET_UP_2_ATT_2"
    "UNET_UP_1_ATT_2"
)

# Read styles from file
STYLES_FILE="data/unlearn_canvas/test/styles.txt"
if [ ! -f "$STYLES_FILE" ]; then
    echo "Error: Styles file not found: $STYLES_FILE"
    exit 1
fi

# Read styles into array
readarray -t STYLES < "$STYLES_FILE"

# Remove any empty lines and whitespace
STYLES=("${STYLES[@]//[$'\t\r\n']}")  # Remove newlines and tabs
STYLES=("${STYLES[@]#"${STYLES[@]%%[![:space:]]*}"}")  # Remove leading spaces
STYLES=("${STYLES[@]%"${STYLES[@]##*[![:space:]]}"}")  # Remove trailing spaces
STYLES=("${STYLES[@]//#*}") # Remove comment lines
STYLES=("${STYLES[@]}")  # Remove empty elements

# Update array size in SBATCH directive
ARRAY_SIZE=${#STYLES[@]}
if [ $ARRAY_SIZE -eq 0 ]; then
    echo "Error: No styles found in $STYLES_FILE"
    exit 1
fi

echo "Found $ARRAY_SIZE styles in $STYLES_FILE"

#==============================================================================
# LOGGING SETUP
#==============================================================================

echo "=========================================="
echo "Job Array Task Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

#==============================================================================
# STYLE SELECTION
#==============================================================================

# Array indices in bash start at 0, but SLURM_ARRAY_TASK_ID starts at 1
STYLE_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_STYLE="${STYLES[STYLE_INDEX]}"

echo "Processing style: ${CURRENT_STYLE}"

#==============================================================================
# MAIN EXECUTION
#==============================================================================

echo "Starting representation generation..."

# Convert layers array to space-separated string
LAYERS_STR="${LAYERS[@]}"

# Run the Python script
uv run python "${PYTHON_SCRIPT}" \
    --prompts-dir "${PROMPTS_DIR}" \
    --style "${CURRENT_STYLE}" \
    --layers ${LAYERS_STR} \
    --preferred-device cuda \
    --guidance-scale 7.5 \
    --steps 50 \
    --seed 42

EXIT_CODE=$?

#==============================================================================
# COMPLETION SUMMARY
#==============================================================================

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: ${EXIT_CODE})"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE