#!/bin/bash

# Script to translate opus100 dev set and compute bleu scores, both for supervised and zero-shot translations
# Script arguments:
#  1. path of the model checkpoints (glob style)
#  2. experiment name (used to name the output table)
#  3. supervised language pairs to be tested, separated by whitespaces. Languages of each pair should be in alphabetical order
#  4. unsupervised language pairs to be tested (formatted as previous arg)
#
# To install sentencepiece in mahti I followed the instructions in the repo, with some modifications:
#
# $ cd $TMPDIR # to use local storage
# $ module load cmake
# $ wget https://github.com/google/sentencepiece/archive/refs/heads/master.zip
# $ unzip master.zip
# $ cd sentencepiece-master
# $ mkdir build
# $ cd build
# $ cmake -DHPX_WITH_MALLOC=system .. # see https://github.com/STEllAR-GROUP/hpx/issues/2524
# $ make -j $(nproc)
#
# Finally, I moved the directory from the temporary local disk to /scratch/project_2005099/sentencepiece-master
# In puhti I did the same thing, but loading also the gcc module. At the beginning:
# $ module load gcc
# $ module load cmake

root_path="/scratch/project_2005099/members/micheleb/projects/scaleUpMNMT"
onmt_path="${root_path}/OpenNMT-py-v2"
out_path="${root_path}/opus-experiments/results"
sentencepiece_path="/scratch/project_2005099/sentencepiece-master/build/src"
processed_data="/scratch/project_2005099/data/opus/prepare_opus_data_tc_out"
opus_data="${processed_data}/opus-100-corpus/v1.0"
checkpoint_path=${1:-"/scratch/project_2005099/models/opus03/opus03.50.adaf.none"}
exp_name=${2:-"opus03-50-adaf-none"}
sandbox_path="/scratch/project_2005099/members/raganato/sandBoxPytorch2106_2"
language_pairs_inp_string=${3:-"ar-en en-fr en-zh"}
zero_shot_pairs_inp_string=${4:-"ar-fr ar-zh fr-zh"}
checkpoints_inp_string=${5:-"50000 75000 100000"}

IFS=' ' read -ra language_pairs <<< "$language_pairs_inp_string"
IFS=' ' read -ra zero_shot_pairs <<< "$zero_shot_pairs_inp_string"
IFS=' ' read -ra checkpoints <<< "$checkpoints_inp_string"

# initialize table
out_table="${out_path}/bleu_scores_${exp_name}.tsv"
{
  printf LP
  printf "\t%s" "${checkpoints[@]}"
  echo ""
} > $out_table

CUR_DIR=$(pwd)

# Supervised translation directions
for lp in "${language_pairs[@]}"
do
    IFS=- read l1 l2 <<< $lp

    for td in $l1-$l2 $l2-$l1
    do
        IFS=- read sl tl <<< $td
        td_results=()  # array to store scores for all checkpoints for the current translation direction
        echo "Testing src/tgt: $sl/$tl"
        tl_code=$(jq -c ".${tl} | .[\"639-2\"]" $processed_data/iso_639-1.json | tr -d '"')
        if [ "$tl" == "sh" ] || [ "$tl" == "bs" ] || [ "$tl" == "sr" ]
        then
          sp_model=$(find $processed_data -type f -name "opusTC.hbs.*.spm")
        else
          sp_model=$(find $processed_data -type f -name "opusTC.$tl_code.*.spm")
        fi
        for cp in "${checkpoints[@]}"
        do
            checkpoint="${checkpoint_path}_step_${cp}_"
            base=$(basename "$checkpoint")
            singularity_wrapper exec --nv $sandbox_path python -u $onmt_path/translate.py \
                -gpu 0 \
                -data_type text \
                -src_lang "$sl" \
                -tgt_lang "$tl" \
                -model "$checkpoint" \
                -src $processed_data/supervised/$lp/opus.$lp-dev.$sl.sp \
                -output $out_path/test.$sl-$tl.${base}hyp.sp
            echo "Sentencepiece decoding"
            cd $sentencepiece_path
            ./spm_decode -model=$sp_model -input_format=piece \
                < "${out_path}/test.${sl}-${tl}.${base}hyp.sp" \
                > "${out_path}/test.${sl}-${tl}.${base}hyp"
            cd $CUR_DIR
            echo "Computing BLEU score"
            bleu=$(singularity_wrapper exec --nv $sandbox_path \
                sacrebleu --score-only "${opus_data}/supervised/${lp}/opus.${lp}-dev.${tl}" \
                < "${out_path}/test.${sl}-${tl}.${base}hyp")
            td_results+=("$bleu")
            echo "${td} ${cp} ${bleu}"
        done
        {
          printf "%s" $td
          printf "\t%s" "${td_results[@]}"
          echo ""
        } >> $out_table
    done
done

# Zero-shot translation directions
for lp in "${zero_shot_pairs[@]}"
do
    IFS=- read l1 l2 <<< $lp

    for td in $l1-$l2 $l2-$l1
    do
        IFS=- read sl tl <<< $td
        td_results=()  # array to store scores for all checkpoints for the current translation direction
        echo "Testing src/tgt: $sl/$tl"
        tl_code=$(jq -c ".${tl} | .[\"639-2\"]" $processed_data/iso_639-1.json | tr -d '"')
        sp_model=$(find $processed_data -type f -name "opusTC.$tl_code.*.spm")
        for cp in "${checkpoints[@]}"
        do
            checkpoint="${checkpoint_path}_step_${cp}_"
            base=$(basename "$checkpoint")
            singularity_wrapper exec --nv $sandbox_path python -u $onmt_path/translate.py \
                -gpu 0 \
                -data_type text \
                -src_lang "$sl" \
                -tgt_lang "$tl" \
                -model "$checkpoint" \
                -src $processed_data/zero-shot/$lp/opus.$lp-test.$sl.sp \
                -output $out_path/test.$sl-$tl.${base}hyp.sp
            echo "Sentencepiece decoding"
            cd $sentencepiece_path
            ./spm_decode -model=$sp_model -input_format=piece \
                < "${out_path}/test.${sl}-${tl}.${base}hyp.sp" \
                > "${out_path}/test.${sl}-${tl}.${base}hyp"
            cd $CUR_DIR
            echo "Computing BLEU score"
            bleu=$(singularity_wrapper exec --nv $sandbox_path \
                sacrebleu --score-only "${opus_data}/zero-shot/${lp}/opus.${lp}-test.${tl}" \
                < "${out_path}/test.${sl}-${tl}.${base}hyp")
            td_results+=("$bleu")
            echo "${td} ${cp} ${bleu}"
        done
        {
          printf "%s" $td
          printf "\t%s" "${td_results[@]}"
          echo ""
        } >> $out_table
    done
done
