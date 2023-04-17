#!/bin/bash
#SBATCH --job-name=train_opus03-50-adaf-none
#SBATCH --account=project_2005099
# SBATCH --partition=gputest
# SBATCH --time=00:14:59
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge
module load pytorch/1.8
python -m pip install --user configargparse subword-nmt sacrebleu ipdb

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export LOG_DIR=$EXP_DIR/logs

nvidia-smi dmon -s mu -d 5 -o TD > $LOG_DIR/gpu_load-opus03.50.adaf.none.log &
srun python -u ${ONMT}/train.py \
                  -config ${EXP_DIR}/config/config-opus03-50-adaf-none.yml \
                  -train_steps 150000
