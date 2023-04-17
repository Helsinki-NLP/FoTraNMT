#!/bin/bash
#SBATCH --job-name=train_opus24-50-adaf-none
#SBATCH --account=project_2005099
#SBATCH --partition=gputest
#SBATCH --time=00:14:59
# SBATCH --partition=gpu
# SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
# SBATCH --mail-user=michele.boggia@helsinki.fi

module purge
module load pytorch/1.8
python -m pip install --user configargparse subword-nmt sacrebleu ipdb

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export LOG_DIR=$EXP_DIR/logs
export EXP_ID=opus24-50-adaf-none

srun bash python_multinode_wrapper.sh ${ONMT}/train.py \
            -config ${EXP_DIR}/config/config-opus24-50-adaf.yml \
            -save_model /scratch/project_2005099/models/opus24/${EXP_ID} \
            -enc_layers 6 \
            -train_steps 100000 \
            -master_ip $SLURMD_NODENAME \
            -master_port 8885
