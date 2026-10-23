#!/bin/bash
#SBATCH --job-name=six_fxb5_T1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=30000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Submit with `sbatch run.sh` in this folder. Any GPU: the script finds at its start how many
# batches per epoch fit on the card. A job that reaches the time limit is continued from its
# newest checkpoint by submitting this script again.
module load gcc12-env/12.3.0
module load miniconda3/23.5.2
conda activate pinn_env

cd "$SLURM_SUBMIT_DIR" || exit 1
MPLBACKEND=Agg PYTHONUNBUFFERED=1 python main.py > full_time_ext_${SLURM_JOB_ID}.txt 2>&1
