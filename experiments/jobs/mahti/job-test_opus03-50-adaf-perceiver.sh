#!/bin/bash
#SBATCH --job-name=test_opus03-50-adaf-perceiver-rn
#SBATCH --account=project_2005099
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH -o slurm-%x_%J.out

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export EXP_DIR=$ROOT_DIR/opus-experiments

srun bash $EXP_DIR/scripts/test_opus.sh \
  "/scratch/project_2005099/models/opus03/opus03-50-adaf-perceiver-rn" \
  "opus03-50-adaf-perceiver-rn" \
  "ar-en en-fr en-zh" \
  "ar-fr ar-zh fr-zh" \
  "50000 75000"
