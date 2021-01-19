# !/bin/bash 

ONMT=`pwd`
SAVE_PATH=$ONMT/model/demo

# Translate          -tgt data/sample_data/test_2016_flickr.lc.norm.tok.cs \          -verbose
python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_MULTI_step_100.pt \
         -src_lang fr \
         -src data/sample_data/test_2016_flickr.lc.norm.tok.bpe.fr \
         -tgt_lang cs \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_fr-cs.bpe.txt 
#         -gpu 0


python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_MULTI_step_100.pt \
         -src_lang en \
         -src data/sample_data/test_2016_flickr.lc.norm.tok.bpe.fr \
         -tgt_lang cs \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_en-cs.bpe.txt 
#         -gpu 0


python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_MULTI_step_100.pt \
         -src_lang de \
         -src data/sample_data/test_2016_flickr.lc.norm.tok.bpe.fr \
         -tgt_lang cs \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_de-cs.bpe.txt 
#         -gpu 0




testref=data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.cs
cat ${SAVE_PATH}/MULTILINGUAL_prediction_fr-cs.bpe.txt | sed -E 's/(@@ )|(@@ ?$)//g' | sacrebleu $testref



