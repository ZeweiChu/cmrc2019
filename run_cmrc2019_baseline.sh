#!/bin/bash
source activate transformers

PATH_TO_BERT="../chinese_wwm_ext_L-12_H-768_A-12"

python run_cmrc2019_baseline.py \
    --vocab_file ./bert_weights_chinese/vocab.txt \
    --bert_config_file $PATH_TO_BERT/bert_config.json \
    --init_checkpoint $PATH_TO_BERT/pytorch_model.bin \
    --do_train \
    --do_predict \
    --train_file ./data/cmrc2019_train.json \
    --predict_file ./data/cmrc2019_trial.json \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --max_seq_length 512 \
    --output_dir ./output_model
