#!/bin/bash
#SBATCH --job-name=train_opus48-50-adaf-none
#SBATCH --account=project_2005099
# SBATCH --time=00:15:00
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export LOG_DIR=$EXP_DIR/logs
export SAND_BOX=/scratch/project_2005099/members/raganato/sandBoxPytorch2106_2
export EXP_ID=opus48-50-adaf-none

srun bash python_multinode_wrapper.sh ${ONMT}/train.py \
            -config ${EXP_DIR}/config/config-opus48-50-adaf.yml \
            -save_model /scratch/project_2005099/models/opus48/${EXP_ID} \
            -enc_layers 6 \
            -train_steps 100000 \
            -master_ip $SLURMD_NODENAME \
            -master_port 8885 #\
#            -tensorboard \
#            -tensorboard_log_dir ${LOG_DIR}/tensorboard/${EXP_ID} \
#            -report_stats_from_parameters
