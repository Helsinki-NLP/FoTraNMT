#!/bin/bash
#SBATCH --job-name=test_opus09-50-adaf-lin
#SBATCH --account=project_2005099
#SBATCH --partition=gpusmall
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export EXP_DIR=$ROOT_DIR/opus-experiments

srun bash $EXP_DIR/scripts/test_opus.sh \
  "/scratch/project_2005099/models/opus09/opus09-50-adaf-lin" \
  "opus09-50-adaf-lin" \
  "ar-en en-fr en-zh de-en en-nl en-ru en-tr en-vi en-th" \
  "ar-fr ar-zh fr-zh ar-de ar-nl ar-ru de-fr de-nl de-ru de-zh fr-nl fr-ru nl-ru nl-zh ru-zh"
