#!/bin/bash

##################################################################################
# The script downloads the opus 100 dataset.
# OPUS 100: https://opus.nlpl.eu/opus-100.php
# It process the data using the sentencepiece models of
# https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/tatoeba/SentencePieceModels.md
#
# How I ran it:
# $ cd /scratch/project_2005099/members/micheleb/projects/scaleUpMNMT/OpenNMT-py-v2/data
# $ ./prepare_opus_data.mahti.sh /scratch/project_2005099/data/opus/prepare_opus_data_out
##################################################################################

# provide script usage instructions
if [ $# -eq 0 ]
then
    echo "usage: $0 <data_dir>"
    exit 1
fi

# set relevant paths
SP_PATH=/scratch/project_2005099/sentencepiece-master/build/src
DATA_PATH=$1
VOCAB_SIZE_ENG=32k
VOCAB_SIZES=("32k" "16k" "8k" "4k")  # used for all languages other than zh and ja
VOCAB_SIZE_LARGE=64k  # used for zh and ja

CUR_DIR=$(pwd)

# Download the default datasets into the $DATA_PATH; mkdir if it doesn't exist
mkdir -p $DATA_PATH

if true; then
  cd $DATA_PATH
  if [ ! -d opus-100-corpus ]
  then
    echo "Downloading and extracting Opus100"
    wget -q --trust-server-names https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz
    tar -xzf opus-100-corpus-v1.0.tar.gz
  fi
  if [ ! -f iso_639-1.json ]
  then
    echo "Downloading mapping from ISO 639-1 to 639-2"
    wget -q https://raw.githubusercontent.com/haliaeetus/iso-639/master/data/iso_639-1.json
  fi
  cd $CUR_DIR
fi

language_pairs=( $( ls $DATA_PATH/opus-100-corpus/v1.0/supervised/ ) )

if true; then

  # Download sentencepiece model and vocab for English
  if [ ! -f $DATA_PATH/opusTC.eng.$VOCAB_SIZE.spm ]
  then
    cd $DATA_PATH
    echo "Downloading vocab and sentencepiece model for eng with vocab size $VOCAB_SIZE_ENG"
    model_file=opusTC.eng.$VOCAB_SIZE_ENG.spm
    vocab_file=$model_file.vocab
    vocab_file_onmt=opusTC.en.vocab.onmt
    wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$vocab_file
    cut -f 1 $vocab_file > $vocab_file_onmt # prepare vocab in OpenNMT-py format
    wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$model_file
    cd $CUR_DIR
  fi

  # Download sentencepiece models and vocabs for all other languages
  for lp in "${language_pairs[@]}"
  do
    IFS=- read sl tl <<< $lp
    if [[ $sl = "en" ]]
    then
      other_lang=$tl
    else
      other_lang=$sl
    fi

    if [ $other_lang == "sh" ] || [ $other_lang == "bs" ] || [ $other_lang == "sr" ]
    then
      echo "Maybe downloading hbs model and vocab with size $VOCAB_SIZE_ENG and prepare a copy for $other_lang"
      model_file=opusTC.hbs.$VOCAB_SIZE_ENG.spm
      vocab_file=$model_file.vocab
      vocab_file_onmt=opusTC.$other_lang.vocab.onmt
      cd $DATA_PATH
      if [ ! -f $vocab_file ]
      then
        wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$vocab_file
        wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$model_file
      fi
      cut -f 1 $vocab_file > $vocab_file_onmt # prepare vocab in OpenNMT-py format
      cd $CUR_DIR
    else
      # get ISO 639-2 language code
      lang_code=$(jq -c ".${other_lang} | .[\"639-2\"]" $DATA_PATH/iso_639-1.json | tr -d '"')
      if [ $other_lang == "zh" ] || [ $other_lang == "ja" ]
      then
        echo "Downloading vocab and sentencepiece model for $other_lang with vocab size $VOCAB_SIZE_LARGE"
        model_file=opusTC.$lang_code.$VOCAB_SIZE_LARGE.spm
        vocab_file=$model_file.vocab
        vocab_file_onmt=opusTC.$other_lang.vocab.onmt
        cd $DATA_PATH
        wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$vocab_file
        cut -f 1 $vocab_file > $vocab_file_onmt # prepare vocab in OpenNMT-py format
        wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$model_file
        cd $CUR_DIR
      else
        for lang_vocab_size in "${VOCAB_SIZES[@]}"
        do
          echo "Downloading vocab and sentencepiece model for $other_lang with vocab size $lang_vocab_size"
          model_file=opusTC.$lang_code.$lang_vocab_size.spm
          vocab_file=$model_file.vocab
          vocab_file_onmt=opusTC.$other_lang.vocab.onmt
          cd $DATA_PATH
          wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$vocab_file
          cd $CUR_DIR
          if [ -f "$DATA_PATH"/$vocab_file ]
          then
            cd $DATA_PATH
            cut -f 1 $vocab_file > $vocab_file_onmt # prepare vocab in OpenNMT-py format
            wget -q https://object.pouta.csc.fi/Tatoeba-MT-spm/$model_file
            cd $CUR_DIR

            break
          else
            echo "Vocab file not found"
          fi
        done
      fi
    fi
  done
fi

# Process supervised data
if true; then
  for lp in "${language_pairs[@]}"
  do
    echo "Parsing $lp data"
    IFS=- read sl tl <<< $lp
    if [[ $sl = "en" ]]
    then
      other_lang=$tl
    else
      other_lang=$sl
    fi
    # look for other_lang sentencepiece model

    if [ $other_lang == "sh" ] || [ $other_lang == "bs" ] || [ $other_lang == "sr" ]
    then
      model_file="$DATA_PATH/opusTC.hbs.$VOCAB_SIZE_ENG.spm"
    else
      lang_code=$(jq -c ".${other_lang} | .[\"639-2\"]" $DATA_PATH/iso_639-1.json | tr -d '"')
      model_file=$(find $DATA_PATH -type f -name "opusTC.$lang_code.*.spm")
    fi

    echo "model file $model_file"
    if [ ! -z "$model_file" ]
    then
      echo -e "\tParsing train data"
      dir=$DATA_PATH/opus-100-corpus/v1.0/supervised
      mkdir -p $DATA_PATH/supervised/$lp
      cd $SP_PATH
      ./spm_encode --model=$model_file \
                   < $dir/$lp/opus.$lp-train.$other_lang \
                   > $DATA_PATH/supervised/$lp/opus.$lp-train.$other_lang.sp
      ./spm_encode --model=$DATA_PATH/opusTC.eng.$VOCAB_SIZE_ENG.spm \
                   < $dir/$lp/opus.$lp-train.en \
                   > $DATA_PATH/supervised/$lp/opus.$lp-train.en.sp
      cd $CUR_DIR
      if [ -f $dir/$lp/opus.$lp-dev.$other_lang ]
      then
        echo -e "\tParsing dev data"
        cd $SP_PATH
        ./spm_encode --model=$model_file \
                     < $dir/$lp/opus.$lp-dev.$other_lang \
                     > $DATA_PATH/supervised/$lp/opus.$lp-dev.$other_lang.sp
        ./spm_encode --model=$DATA_PATH/opusTC.eng.$VOCAB_SIZE_ENG.spm \
                     < $dir/$lp/opus.$lp-dev.en \
                     > $DATA_PATH/supervised/$lp/opus.$lp-dev.en.sp
        cd $CUR_DIR
      else
        echo -e "\tDev data not found"
      fi
      if [ -f $dir/$lp/opus.$lp-test.$other_lang ]
      then
        echo -e "\tParsing test data"
        cd $SP_PATH
        ./spm_encode --model=$model_file \
                     < $dir/$lp/opus.$lp-test.$other_lang \
                     > $DATA_PATH/supervised/$lp/opus.$lp-test.$other_lang.sp
        ./spm_encode --model=$DATA_PATH/opusTC.eng.$VOCAB_SIZE_ENG.spm \
                     < $dir/$lp/opus.$lp-test.en \
                     > $DATA_PATH/supervised/$lp/opus.$lp-test.en.sp
        cd $CUR_DIR
      else
        echo -e "\tTest data not found"
      fi
    else
      echo "Sentencepiece model not found for $other_lang"
    fi
  done
fi

# Manually remove problematic lines from data files
# TODO: find a better way to handle problematic input files
# remove very long line in ar-en dev data at line 1067 (cause issues in translate.py)
sed -i.bak -e '1067d' $DATA_PATH/opus-100-corpus/v1.0/supervised/ar-en/opus.ar-en-dev.ar
sed -i.bak -e '1067d' $DATA_PATH/opus-100-corpus/v1.0/supervised/ar-en/opus.ar-en-dev.en
sed -i.bak -e '1067d' $DATA_PATH/supervised/ar-en/opus.ar-en-dev.ar.sp
sed -i.bak -e '1067d' $DATA_PATH/supervised/ar-en/opus.ar-en-dev.en.sp

# Parse the test sets for zero-shot translation directions
if true; then
  mkdir -p $DATA_PATH/zero-shot
  for dir in $DATA_PATH/opus-100-corpus/v1.0/zero-shot/*
  do
    lp=$(basename $dir)  # get name of dir from full path
    mkdir -p $DATA_PATH/zero-shot/$lp
    echo "Parsing $lp test data"
    IFS=- read sl tl <<< $lp
    for l in $sl $tl
    do
      l_code=$(jq -c ".${l} | .[\"639-2\"]" $DATA_PATH/iso_639-1.json | tr -d '"')
      model_file=$(find $DATA_PATH -type f -name "opusTC.$l_code.*.spm")
      cd $SP_PATH
      ./spm_encode --model=$model_file \
                 < $dir/opus.$lp-test.$l \
                 > $DATA_PATH/zero-shot/$lp/opus.$lp-test.$l.sp
      cd $CUR_DIR
    done
  done
fi
