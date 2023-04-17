#!/bin/bash
#SBATCH --job-name=prepare_opus_tc
#SBATCH --account=project_2005099
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%x_%J.out

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export EXP_DIR=$ROOT_DIR/opus-experiments
export OUT_DIR=/scratch/project_2005099/data/opus/prepare_opus_data_tc_out

srun bash ${EXP_DIR}/scripts/prepare_opus_data_tc.sh ${OUT_DIR}
