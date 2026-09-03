#!/bin/bash

# Run OutputStatsV2.py many times, once per model output folder_code, in parallel via a job array.
# Submit with: sbatch outputstatsV2_parallel.sh
# (--array below must match the number of entries in FOLDER_CODES -- see note there.)

#SBATCH --job-name=outputstatsV2_parallel
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --qos=cpu-normal
#SBATCH --array=0-5
#SBATCH --output=/scratch/alpine/ccox1@xsede.org/processing_logs/job_%A/run_model-%A_%a.out

module purge
module load anaconda
conda activate lm4analysis

# Repo checkout on Alpine -- update to the actual clone path.
REPO_DIR=/projects/ccox1@xsede.org/lm4px_data_analysis
cd "$REPO_DIR/output_analysis" || exit 1

# One sites file, run against each folder_code below.
SITES_FILE=/projects/ccox1@xsede.org/lm4px_modelling/lm4px_run_scripts/run_configs/sites/snotel_sites_v2.txt

# List every model output folder_code to process. One array task per entry, so keep the
# #SBATCH --array range above in sync with the number of entries here (0-indexed, inclusive).
FOLDER_CODES=(
    s00000000_ghec_v0003
    s01011010_ghec_v0004
    s01011010_gbc_v0005
    s01011010_ghelaps_v0008
    s01011010_ghelaps_veg_v0009
    s01012010_ghelaps_v0011
)

FOLDER_CODE=${FOLDER_CODES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_CSV="outputV2_${SITES_FILE%.txt}_${FOLDER_CODE}.csv"

echo "Task $SLURM_ARRAY_TASK_ID: $SITES_FILE / $FOLDER_CODE -> $OUTPUT_CSV"

python OutputStatsV2.py "$SITES_FILE" "$FOLDER_CODE" --output "$OUTPUT_CSV"
