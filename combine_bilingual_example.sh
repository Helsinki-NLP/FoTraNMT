#!/bin/bash

ROOT=$(pwd)
SAVE_PATH=$ROOT/model/demo

python build_bilingual_model.py -src_lang en \
         -tgt_lang de \
         -encoder "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_encoder_1.pt \
         -decoder "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_decoder_2.pt \
         -bridge "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_bridge.pt \
         -generator "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_generator_2.pt \
         -model_frame "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_frame.pt \
         -output "$SAVE_PATH"/MULTILINGUAL_MULTI_step_10000_de-en.pt
