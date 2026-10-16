#!/bin/bash

# Zip the NLDAS site files and upload them to a Zenodo draft with archive_nldas.py.
# Submit with:  sbatch archive_nldas.sh <deposition_id>
# Omit <deposition_id> to create a new draft; pass it to add to / resume an existing one
# (zips already on Zenodo with a matching md5 are skipped).

#SBATCH --job-name=archive_nldas
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --qos=cpu-normal
#SBATCH --output=/scratch/alpine/ccox1@xsede.org/archive_logs/archive_nldas_%j.out

module purge
module load anaconda
conda activate lm4analysis

# Repo checkout on Alpine -- update to the actual clone path.
REPO_DIR=/projects/ccox1@xsede.org/lm4px_data_analysis
cd "$REPO_DIR/archiving" || exit 1

NLDAS_DIR=/pl/active/snotelmodelling/nldas
TMP_DIR=/scratch/alpine/$USER/archive_nldas_tmp
mkdir -p "$TMP_DIR"

ARGS=("$NLDAS_DIR" --tmp-dir "$TMP_DIR")
if [ -n "$1" ]; then
    ARGS+=(--deposition-id "$1")
fi
# Add --no-compress to ARGS if deflate isn't shrinking the .nc files much.
# Add --metadata nldas_metadata.json to (re)apply the record metadata.

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "python archive_nldas.py ${ARGS[*]}"

# -u: unbuffered, so progress shows up in the log as it happens
python -u archive_nldas.py "${ARGS[@]}"
status=$?

echo "Finished $(date) with exit code $status"
exit $status
