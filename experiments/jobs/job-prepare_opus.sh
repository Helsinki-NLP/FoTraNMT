#!/bin/bash
#SBATCH --job-name=prepare_opus
#SBATCH --account=project_2005099
#SBATCH --time=10:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%x_%J.out

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export OUT_DIR=/scratch/project_2005099/data/opus/prepare_opus_data_out

srun bash ${EXP_DIR}/scripts/prepare_opus_data.mahti.sh ${OUT_DIR}
