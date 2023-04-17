#!/bin/bash
#SBATCH --job-name=train_opus03-50-adaf-perceiver2-rn
#SBATCH --account=project_2005099
# SBATCH --partition=gputest
# SBATCH --time=00:15:00
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export ONMT=$ROOT_DIR/OpenNMT-py-v2
export EXP_DIR=$ROOT_DIR/opus-experiments
export LOG_DIR=$EXP_DIR/logs
export SAND_BOX=/scratch/project_2005099/members/raganato/sandBoxPytorch2106_2
export EXP_ID=opus03-50-adaf-perceiver2-rn

nvidia-smi dmon -s mu -d 5 -o TD > ${LOG_DIR}/gpu_load-${EXP_ID}.log &
srun singularity_wrapper exec --nv $SAND_BOX \
      python -u ${ONMT}/train.py \
            -config ${EXP_DIR}/config/config-opus03-50-adaf.yml \
            -save_model /scratch/project_2005099/models/opus03/${EXP_ID} \
            -ab_layers perceiver perceiver \
            -ab_fixed_length 50 \
            -hidden_ab_size 4096 \
            -ab_layer_norm rmsnorm \
            -enc_layers 4 \
            -train_steps 100000
