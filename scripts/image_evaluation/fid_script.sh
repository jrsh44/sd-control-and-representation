#!/bin/bash
################################################################################
# SLURM Job Script - FID Score Calculation
#
# Purpose:
#   Calculate FID scores between reference images and multiple subdirectories
#
# Usage:
#   sbatch scripts/image_evaluation/fid_script.sh
#
# Output:
#   - FID results saved to file
#   - Logs: logs/fid_job_{JOB_ID}.log
################################################################################

#==============================================================================
# RESOURCE ALLOCATION
#==============================================================================
#SBATCH --account mi2lab
#SBATCH --job-name fid_calc
#SBATCH --time 0-02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --partition short
#SBATCH --output logs/fid_job_%j.log  # %j=job ID

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
echo "FID Score Calculation Job"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Running on: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Navigate to project directory
cd /mnt/evafs/groups/mi2lab/jcwalina/sd-control-and-representation
source .venv/bin/activate
source ./.venv/bin/activate

# Create directories
mkdir -p logs

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Environment Configuration:"
echo "  RESULTS_DIR: ${RESULTS_DIR:-Not set}"
echo ""

#==============================================================================
# EXPERIMENT CONFIGURATION
#==============================================================================

# Python script
PYTHON_SCRIPT="scripts/image_evaluation/fid_script.py"

# Path to reference images (the baseline/original images)
REFERENCE_PATH=""

# Path to parent directory containing subdirectories with images to compare
SUBDIRS_PARENT_PATH=""

# Optional: output file to save results
OUTPUT_FILE=""

#==============================================================================
# VALIDATION
#==============================================================================
if [ -z "$REFERENCE_PATH" ]; then
  echo "ERROR: REFERENCE_PATH is not set"
  exit 1
fi

if [ -z "$SUBDIRS_PARENT_PATH" ]; then
  echo "ERROR: SUBDIRS_PARENT_PATH is not set"
  exit 1
fi

if [ ! -d "$REFERENCE_PATH" ]; then
  echo "ERROR: Reference path not found: $REFERENCE_PATH"
  exit 1
fi

if [ ! -d "$SUBDIRS_PARENT_PATH" ]; then
  echo "ERROR: Subdirectories parent path not found: $SUBDIRS_PARENT_PATH"
  exit 1
fi

#==============================================================================
# RUN FID CALCULATION
#==============================================================================

echo ""
echo "Starting FID calculation..."
echo ""

echo "Configuration:"
echo "  Reference path: ${REFERENCE_PATH}"
echo "  Subdirs parent: ${SUBDIRS_PARENT_PATH}"
if [ -n "$OUTPUT_FILE" ]; then
    echo "  Output file: ${OUTPUT_FILE}"
fi
echo "=========================================="

# Build command
CMD="uv run python ${PYTHON_SCRIPT} \
    --reference_path \"${REFERENCE_PATH}\" \
    --subdirs_parent_path \"${SUBDIRS_PARENT_PATH}\"
    --output_file \"${OUTPUT_FILE}\""



# Add output file if specified
if [ -n "$OUTPUT_FILE" ]; then
    CMD="${CMD} --output_file \"${OUTPUT_FILE}\""
fi

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
    echo "✓ FID calculation completed successfully"
    echo "  Reference: ${REFERENCE_PATH}"
    echo "  Compared with: ${SUBDIRS_PARENT_PATH}"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "  Results saved to: ${OUTPUT_FILE}"
    fi
else
    echo "✗ FID calculation FAILED (code: ${EXIT_CODE})"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE