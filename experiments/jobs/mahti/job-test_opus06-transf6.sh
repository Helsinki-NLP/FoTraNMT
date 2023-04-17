#!/bin/bash
#SBATCH --job-name=test_opus06-transf6
#SBATCH --account=project_2005099
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH -o slurm-%x_%J.out
#SBATCH --dependency=afterok:1513751
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export EXP_DIR=$ROOT_DIR/opus-experiments

srun bash $EXP_DIR/scripts/test_opus.sh \
  "/scratch/project_2005099/models/opus06/opus06-transf6" \
  "opus06-transf6" \
  "ar-en en-fr en-zh de-en en-nl en-ru" \
  "ar-fr ar-zh fr-zh ar-de ar-nl ar-ru de-fr de-nl de-ru de-zh fr-nl fr-ru nl-ru nl-zh ru-zh"
