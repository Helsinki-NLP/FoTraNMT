#!/bin/bash
#SBATCH --job-name=test_opus36-lin-transf4
#SBATCH --account=project_2005099
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
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
  "/scratch/project_2005099/models/opus36/opus36-lin-transf4" \
  "opus36-lin-transf4" \
  "ar-en en-fr en-zh de-en en-nl en-ru en-tr en-vi en-th en-es el-en bg-en en-he en-fi en-ja en-sv en-fa en-mk en-eu en-id bn-en en-ko en-it en-lv en-mt en-et en-ro bs-en en-sr en-is en-uk en-hu en-lt cs-en en-sk en-sq" \
  "ar-fr ar-zh fr-zh ar-de ar-nl ar-ru de-fr de-nl de-ru de-zh fr-nl fr-ru nl-ru nl-zh ru-zh"
