#!/bin/bash

# get en-it language pair from the opus100 dataset
if [ ! -d opus100 ]; then
    mkdir opus100
fi

ONMT=`pwd`/..

OUTPUT_DIR=`pwd`/sample_data_opus100_test
mkdir -p $OUTPUT_DIR

for lang_pair in en-it de-en cs-en en-fi en-fr en-ru en-no en-es
#for lang_pair in de-en cs-en
do
  echo -e "\nLANG PAIR: $lang_pair\n"
  if [ ! -d opus100/${lang_pair} ]; then
    cd opus100
    scp -r "micheleb@mahti.csc.fi:/scratch/project_2005099/data/opus-100-corpus-master/opus-100-corpus/v1.0/supervised/$lang_pair" .
    cd ..
  fi

  DATADIR=`pwd`/opus100/${lang_pair}
  cd $OUTPUT_DIR

  # BPE encodings
  #pip install subword-nmt
  #RECALL: The original segmentation can be restored with the replacement:
  #            sed -r 's/(@@ )|(@@ ?$)//g'
  train_file=$DATADIR/opus.${lang_pair}-train
  bpe_trainfile=$OUTPUT_DIR/opus.${lang_pair}-train.bpe

  valid_file=$DATADIR/opus.${lang_pair}-dev
  bpe_validfile=$OUTPUT_DIR/opus.${lang_pair}-dev.bpe

  test_file=$DATADIR/opus.${lang_pair}-test
  bpe_testfile=$OUTPUT_DIR/opus.${lang_pair}-test.bpe

  num_operations='40000' # approx 20k per language
  codes_file=$OUTPUT_DIR/codecs4k.${lang_pair}
  vocab_file=$OUTPUT_DIR/bpe_vocab.${lang_pair}

  echo "Learning BPE on all the training text"

  arr_langs=(${lang_pair//-/ })
  lang_1=${arr_langs[0]}
  lang_2=${arr_langs[1]}

  # Learn BPE on all the training text, and get resulting vocabulary for each:
  subword-nmt learn-joint-bpe-and-vocab \
           --input ${train_file}.{"${lang_1}","${lang_2}"} \
           --num-workers 4 \
           -s ${num_operations} \
           -o ${codes_file} \
           --write-vocabulary ${vocab_file}.{"${lang_1}","${lang_2}"}

  for lang in "${arr_langs[@]}"
  do
    # re-apply byte pair encoding with vocabulary filter. # for test/dev data, re-use the same options.
    subword-nmt apply-bpe -c "${codes_file}" --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 1 < "${train_file}.${lang}" > "${bpe_trainfile}.${lang}"
    subword-nmt apply-bpe -c "${codes_file}" --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 1  < "${test_file}.${lang}"  > "${bpe_testfile}.${lang}"
    subword-nmt apply-bpe -c "${codes_file}" --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 1 < "${valid_file}.${lang}" > "${bpe_validfile}.${lang}"
  done

  # Preprocess with ONMT
  # here we use the vocabularies constructed with subword-nmt
  for src_lang in "${arr_langs[@]}"
  do
   for tgt_lang in "${arr_langs[@]}"
   do
     # no need for repeated data
     if [ ! -f $OUTPUT_DIR/op100.${tgt_lang}-${src_lang}_2.vocab.pt ]; then
       # preprocess
       echo -e "\n Preprocessing language pair $src_lang - $tgt_lang "
       SAVEDATA=$OUTPUT_DIR/op100.${src_lang}-${tgt_lang}
       src_train_file=${bpe_trainfile}.${src_lang}
       tgt_train_file=${bpe_trainfile}.${tgt_lang}
       src_valid_file=${bpe_validfile}.${src_lang}
       tgt_valid_file=${bpe_validfile}.${tgt_lang}
       python3 $ONMT/preprocess.py \
         -train_src $src_train_file \
         -train_tgt $tgt_train_file \
         -valid_src $src_valid_file \
         -valid_tgt $tgt_valid_file \
         -save_data ${SAVEDATA} \
         -src_vocab ${vocab_file}.${src_lang} \
         -tgt_vocab ${vocab_file}.${tgt_lang}
       fi
   done
  done

  cd ${ONMT}/data
done


# TODO: what should I do with this?
## ---------------------------------------------------------------------------------------------
##     CREATE A VOCAB. FOR EACH LANGUAGE, USING ALL OF THE DATASETS FOR THAT LANGUAGE.
##----------------------------------------------------------------------------------------------
#echo "Preprocessing vocab"
#python3 $ONMT/preprocess_build_vocab.py \
#   -train_dataset_prefixes $OUTPUT_DIR
#
## Fix vocabs
#echo "Fixing vocab"
##python3 $ONMT/fix_vocab.py -l en it de cs "fi" fr ru no es -b $OUTPUT_DIR -v op100 -n $OUTPUT_DIR
#python3 $ONMT/fix_vocab.py -l de en cs -b $OUTPUT_DIR -v op100 -n $OUTPUT_DIR
