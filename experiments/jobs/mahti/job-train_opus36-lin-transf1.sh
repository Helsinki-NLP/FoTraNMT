#!/bin/bash
#SBATCH --job-name=train_opus36-lin-transf1
#SBATCH --account=project_2005099
# SBATCH --time=00:15:00
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=32G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export LOG_DIR=$EXP_DIR/logs
export SAND_BOX=/scratch/project_2005099/members/raganato/sandBoxPytorch2106_2
export EXP_ID=opus36-lin-transf1

srun bash python_multinode_wrapper.sh ${ONMT}/train.py \
            -config ${EXP_DIR}/config/config-opus36-50-adaf.yml \
            -save_model /scratch/project_2005099/models/opus36/${EXP_ID} \
            -ab_layers lin transformer \
            -hidden_ab_size 2048 \
            -enc_layers 4 \
            -dec_layers 6 \
            -train_steps 100000 \
            -master_ip $SLURMD_NODENAME \
            -master_port 8885 #\
#            -tensorboard \
#            -tensorboard_log_dir ${LOG_DIR}/tensorboard/${EXP_ID} \
#            -report_stats_from_parameters
