#!/bin/bash

# clean the old runs (if any)
rm -rf sample_run/elmo
rm -rf sample_run/parser

# Caution: Pre-trained word embedding file 'sample_run/cc.en.300.vec.sample' has to be replaced with the raw file extracted from https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.en.300.vec.gz to reproduce ELMoLex results. The default value for the hyper-parameters are optimal and used for our final submission.

if [ $1 == "check" ] ; then
# run for 1 epoch
# create ELMo features
python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/elmo --num_epochs 1
# train the parser
python train.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul --num_epochs 1  
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder sample_run/parser --system_tb sample_run/en_lines-ud-test_pred.conllu --gold_tb sample_run/gold --tb_out sample_run/en_lines-test-pred.conllu --lexicon sample_run/UDLex_English-Apertium.conllul --word_path sample_run/cc.en.300.vec.sample
fi

if [ $1 == "complete" ] ; then
# run for recommended no. of epochs
# create ELMo features
python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/elmo 
# train the parser
python train.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul  
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder sample_run/parser --system_tb sample_run/en_lines-ud-test_pred.conllu --gold_tb sample_run/gold --tb_out sample_run/en_lines-test-pred.conllu --lexicon sample_run/UDLex_English-Apertium.conllul --word_path sample_run/cc.en.300.vec.sample
fi
