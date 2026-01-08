#!/bin/bash
################################################################################
# SLURM Array Job Script - Image Generation Test
# Purpose: Test image generation with Stable Diffusion using different prompts
#
# Usage:
#   sbatch scripts/test_image_generation.sh
#
# Description:
#   This script runs 6 parallel tasks (0-5) to generate images:
#   - Each task uses a different prompt
#   - Tests both CUDA and CPU device preferences
#   
#   Tasks 0-2: Prompt 1, 2, 3 with CUDA
#   Tasks 3-5: Prompt 1, 2, 3 with CPU fallback
#   
#   Results are saved to:
#   - Images: results/images/task_{TASK_ID}_{timestamp}_{prompt}.png
#   - Metadata: results/images/task_{TASK_ID}_{timestamp}_metadata.txt
#   - Logs: logs/job_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Tell SLURM what resources you need
#==============================================================================

#SBATCH --account mi2lab          # Your compute account name
#SBATCH --job-name sd_img_test          # Name shown in queue
#SBATCH --time 0-00:30:00               # Max runtime (Days-Hours:Min:Sec)
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --ntasks-per-node 1             # Tasks per node
#SBATCH --gres gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task 6               # CPU cores (for data loading)
#SBATCH --mem 16G                       # RAM
#SBATCH --partition short               # Queue name
#SBATCH --output logs/job_%A_%a.log     # %A=job ID, %a=task ID
#SBATCH --array 0-5%3                   # 6 tasks, max 3 running at once

# Optional: email notifications
# #SBATCH --mail-user 01180706@pw.edu.pl
# #SBATCH --mail-type FAIL,END

#==============================================================================
# ERROR HANDLING
#==============================================================================

# Exit if any command fails or uses undefined variables
set -eu

#==============================================================================
# SETUP
#==============================================================================

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Create directories if needed
mkdir -p logs
mkdir -p results/images

# Navigate to your project directory
cd /mnt/evafs/groups/mi2lab/mjarosz/sd-control-and-representation

# Load environment variables if you have them
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# Display key environment variables for debugging
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set (will use default)}"
echo "  HF_HOME: ${HF_HOME:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
# Define all parameters you want to test
#==============================================================================

# Python script
PYTHON_SCRIPT="./scripts/tests/image_generation.py"

# Prompts to test
PROMPTS=(
    "A frog wearing a top hat and monocle"
    "An orange tabby cat sitting on a skateboard"
    "A futuristic cityscape at sunset with flying cars"
)

# Device preferences
DEVICE_PREFERENCES=(
    "cuda"
    "cpu"
)

# Image generation parameters
GUIDANCE_SCALE=7.5
NUM_STEPS=50
SEED=42

#==============================================================================
# TASK MAPPING
# Convert task ID (0-5) to specific parameter combination
#
# How it works:
# - Tasks 0-2: Prompts 1, 2, 3 with CUDA preference
# - Tasks 3-5: Prompts 1, 2, 3 with CPU preference
#==============================================================================

# Calculate parameter indices
PROMPT_IDX=$((SLURM_ARRAY_TASK_ID % 3))              # Cycles 0-2
DEVICE_IDX=$((SLURM_ARRAY_TASK_ID / 3))              # 0 for tasks 0-2, 1 for tasks 3-5

# Get actual parameter values
CURRENT_PROMPT=${PROMPTS[$PROMPT_IDX]}
CURRENT_DEVICE=${DEVICE_PREFERENCES[$DEVICE_IDX]}

# Display configuration
echo ""
echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Prompt: ${CURRENT_PROMPT}"
echo "  Device Preference: ${CURRENT_DEVICE}"
echo "  Guidance Scale: ${GUIDANCE_SCALE}"
echo "  Steps: ${NUM_STEPS}"
echo "  Seed: ${SEED}"
echo "=========================================="
echo ""

#==============================================================================
# RUN EXPERIMENT
#==============================================================================

echo "Starting experiment..."

# Run the Python script with the parameters
uv run python ${PYTHON_SCRIPT} \
    --prompt "${CURRENT_PROMPT}" \
    --preferred-device ${CURRENT_DEVICE} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --steps ${NUM_STEPS} \
    --seed ${SEED}

# Capture whether it succeeded or failed
EXIT_CODE=$?

#==============================================================================
# SUMMARY
#==============================================================================

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (exit code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE