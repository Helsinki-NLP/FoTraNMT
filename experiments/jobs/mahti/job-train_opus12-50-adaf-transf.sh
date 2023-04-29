#!/bin/bash
#SBATCH --job-name=train_opus12-50-adaf-transf
#SBATCH --account=project_2005099
# SBATCH --partition=gputest
# SBATCH --time=00:15:00
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
export EXP_ID=opus12-50-adaf-transf

nvidia-smi dmon -s mu -d 5 -o TD > ${LOG_DIR}/gpu_load-${EXP_ID}.log &
srun singularity_wrapper exec --nv $SAND_BOX \
      python -u ${ONMT}/train.py \
            -config ${EXP_DIR}/config/config-opus12-50-adaf.yml \
            -save_model /scratch/project_2005099/models/opus12/${EXP_ID} \
            -ab_layers transformer \
            -hidden_ab_size 2048 \
            -enc_layers 5 \
            -train_steps 100000 #\
#            -tensorboard \
#            -tensorboard_log_dir ${LOG_DIR}/tensorboard/${EXP_ID} \
#            -report_stats_from_parameters