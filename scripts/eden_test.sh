#!/bin/bash
################################################################################
# SLURM Array Job Script - Cluster Test
# Purpose: Test job array functionality with different parameter combinations
#
# Usage:
#   sbatch scripts/eden_test.sh
#
# Description:
#   This script runs 24 parallel tasks (0-23) split between two test scripts:
#   - Tasks 0-11:  Run eden_cluster_test_1.py
#   - Tasks 12-23: Run eden_cluster_test_2.py
#   
#   Each task tests different combinations of:
#   - 4 model backbones (resnet18, resnet50, vgg16, mobilenet)
#   - 3 datasets (mnist, cifar10, imagenet)
#   - 2 splits (validation, training)
#   
#   Results are saved to:
#   - Script 1: results/test_result_task_{0-11}.txt
#   - Script 2: results/alternative/alternative_test_task_{12-23}.txt
#   - Logs: logs/job_{JOB_ID}_{TASK_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
# Tell SLURM what resources you need
#==============================================================================

#SBATCH --account mjarosz          # Your compute account name
#SBATCH --job-name eden_test        # Name shown in queue
#SBATCH --time 0-00:10:00               # Max runtime (Days-Hours:Min:Sec)
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --ntasks-per-node 1             # Tasks per node
#SBATCH --gres 1                    # Request 1 GPU
#SBATCH --cpus-per-task 12              # CPU cores (for data loading)
#SBATCH --mem 16G                       # RAM
#SBATCH --partition short               # Queue name
#SBATCH --output logs/job_%A_%a.log     # %A=job ID, %a=task ID
#SBATCH --array 0-23%5                  # 24 tasks, max 5 running at once

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
mkdir -p data
mkdir -p results

# Navigate to your project directory
cd /mnt/evafs/groups/mi2lab/mjarosz/sd-control-and-representation

# Load environment variables if you have them
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

#==============================================================================
# EXPERIMENT CONFIGURATION
# Define all parameters you want to test
#==============================================================================

# Python scripts to run
PYTHON_SCRIPTS=(
    "code/tests/eden_cluster_test_1.py"
    "code/tests/eden_cluster_test_2.py"
)

# Model backbones
BACKBONES=(
    "resnet18"
    "resnet50"
    "vgg16"
    "mobilenet"
)

# Datasets
DATASETS=(
    "mnist"
    "cifar10"
    "imagenet"
)

# Splits
SPLITS=(
    ""              # validation (default)
    "--train-split" # training
)

#==============================================================================
# TASK MAPPING
# Convert task ID (0-23) to specific parameter combination
#
# How it works:
# - Tasks 0-11:  Use eden_cluster_test.py
# - Tasks 12-23: Use eden_cluster_test_2.py
# Both use same parameters: 4 backbones × 3 datasets × 2 splits = 24 combinations
#==============================================================================

# Determine which script to use
if [ $SLURM_ARRAY_TASK_ID -lt 12 ]; then
    SCRIPT_IDX=0
    ADJUSTED_TASK_ID=$SLURM_ARRAY_TASK_ID
else
    SCRIPT_IDX=1
    ADJUSTED_TASK_ID=$((SLURM_ARRAY_TASK_ID - 12))
fi

PYTHON_SCRIPT=${PYTHON_SCRIPTS[$SCRIPT_IDX]}

# Calculate parameter indices (same logic for both scripts)
BACKBONE_IDX=$((ADJUSTED_TASK_ID % 4))                  # Cycles 0-3
DATASET_IDX=$(( (ADJUSTED_TASK_ID / 4) % 3 ))          # 0, 1, or 2
SPLIT_IDX=$((ADJUSTED_TASK_ID / 12))                   # 0 or 1 (but always 0 for tasks 0-11)

# Get actual parameter values
CURRENT_BACKBONE=${BACKBONES[$BACKBONE_IDX]}
CURRENT_DATASET=${DATASETS[$DATASET_IDX]}
CURRENT_SPLIT=${SPLITS[$SPLIT_IDX]}

# Create readable split name
if [ -z "$CURRENT_SPLIT" ]; then
    SPLIT_NAME="validation"
else
    SPLIT_NAME="train"
fi

# Display configuration
echo ""
echo "Task Configuration:"
echo "  Script: ${PYTHON_SCRIPT}"
echo "  Backbone: ${CURRENT_BACKBONE}"
echo "  Dataset:  ${CURRENT_DATASET}"
echo "  Split:    ${SPLIT_NAME}"
echo "=========================================="
echo ""

#==============================================================================
# RUN EXPERIMENT
#==============================================================================

echo "Starting experiment..."

# Run the selected Python script with the same parameters
uv run python ${PYTHON_SCRIPT} \
    --backbone ${CURRENT_BACKBONE} \
    --dataset ${CURRENT_DATASET} \
    --batch-size 256 \
    --workers 8 \
    --seed 42 \
    ${CURRENT_SPLIT}

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