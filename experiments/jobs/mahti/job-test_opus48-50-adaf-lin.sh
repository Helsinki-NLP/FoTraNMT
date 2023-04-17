#!/bin/bash
#SBATCH --job-name=test_opus48-50-adaf-lin
#SBATCH --account=project_2005099
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=4G
#SBATCH -o slurm-%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.boggia@helsinki.fi

module purge

export ROOT_DIR=/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT
export EXP_DIR=$ROOT_DIR/opus-experiments

srun bash $EXP_DIR/scripts/test_opus.sh \
  "/scratch/project_2005099/models/opus48/opus48-50-adaf-lin" \
  "opus48-50-adaf-lin" \
  "ar-en en-fr en-zh de-en en-nl en-ru en-tr en-vi en-th en-es el-en bg-en en-he en-fi en-ja en-sv en-fa en-mk en-eu en-id bn-en en-ko en-it en-lv en-mt en-et en-ro bs-en en-sr en-is en-uk en-hu en-lt cs-en en-sk en-sq en-ms da-en en-si en-no en-pt en-ml en-pl en-hr en-ur ca-en en-sl en-mg" \
  "ar-fr ar-zh fr-zh ar-de ar-nl ar-ru de-fr de-nl de-ru de-zh fr-nl fr-ru nl-ru nl-zh ru-zh"
