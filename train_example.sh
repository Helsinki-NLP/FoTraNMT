# !/bin/bash 

# Train
ONMT=`pwd`
SAVE_PATH=$ONMT/model/demo
mkdir -p $SAVE_PATH

python train.py -data data/sample_data/m30k.en-cs \
                      data/sample_data/m30k.de-cs \
                      data/sample_data/m30k.fr-cs \
        -src_tgt       en-cs   de-cs   fr-cs \
        -lang_fam      fam1    fam1    fam2  \
        -batch_size    2048    2048    2048 \
        -batch_type    tokens  tokens  tokens \
        -normalization tokens  tokens  tokens \
        -save_model ${SAVE_PATH}/MULTILINGUAL_MULTI \
        -use_attention_bridge \
        -attention_heads 20 \
        -rnn_size 512 \
        -word_vec_size 512 \
        -transformer_ff 2048 \
        -heads 8 \
        -encoder_type transformer \
        -decoder_type transformer \
        -position_encoding \
        -enc_layers 3 \
        -dec_layers 3 \
        -dropout  0.1 \
        -train_steps 10000 \
        -valid_steps 5000 \
        -optim adam \
        -learning_rate 0.0002 \
        -save_checkpoint_steps 10000 \
        -model_type text \
        -world_size 1 -gpu_ranks 0



# IWSLT 2020 - EXAMPLE:

DATAPATH=/path/to/ONMTpreprocessed/data/

python train.py -data ${DATAPATH}/h5/all_partial/ENaudio_DEtext/data \
                      ${DATAPATH}/all/ENtext_DEtext/data     \
                      ${DATAPATH}/h5/all_partial/ENaudio_ENtext/data \
                      ${DATAPATH}/all/DEtext_ENtext/data \
                -src_tgt           en_a-de_t      en_t-de_t     en_a-en_t     de_t-en_t   \
                -model_type        audiotrf       text          audiotrf      text        \
                -audio_enc_pooling 1              1             1             1           \
                -batch_size        32             4096          32            4096        \
                -accum_count       8 \
                -batch_type        sents          tokens        sents         tokens      \
                -normalization     sents          tokens        sents         tokens      \
                -decoder_type      transformer    transformer   transformer   transformer \
                -cnn_kernel_width 3            \
                -save_model /scratch/project_2000945/iwslt19/models/audiotrf/$trainoption \
                -attention_heads 100            \
                -use_attention_bridge          \
                -init_decoder attention_matrix \
                -n_mels         40  \
                -n_stacked_mels  1  \
                -drop_audioafter 5500 \
        -enc_layers 3 -dec_layers 3 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer  -position_encoding \
        -max_generator_batches 2 -dropout 0.1 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 \
                -report_every            250  \
                -train_steps          350000  \
                -valid_steps           5000  \
                -save_checkpoint_steps 5000  \
                -keep_checkpoint           20  \
                -world_size   1  \
                -gpu_ranks    0  \
                -save_config /scratch/project_2000945/iwslt19/models/audiotrf/train-trf_$trainoption.config

python train.py -config /scratch/project_2000945/iwslt19/models/audiotrf/train-trf_$trainoption.config
