#!/bin/bash

# clean the old runs (if any)
#rm -rf sample_run/elmo
#rm -rf sample_run/parser
source activate conll18
# Caution: Pre-trained word embedding file 'sample_run/cc.en.300.vec.sample' has to be replaced with the raw file extracted from https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.en.300.vec.gz to reproduce ELMoLex results. The default value for the hyper-parameters are optimal and used for our final submission.

if [ "$1" == "check" ] ; then
# run for 1 epoch
# create ELMo features
python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/elmo --num_epochs 1
# train the parser
python train.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul --num_epochs 1  
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder sample_run/parser --system_tb sample_run/en_lines-ud-test_pred.conllu --gold_tb sample_run/gold --tb_out sample_run/en_lines-test-pred.conllu --lexicon sample_run/UDLex_English-Apertium.conllul --word_path sample_run/cc.en.300.vec.sample
fi

if [ "$1" == "complete" ] ; then
# run for recommended no. of epochs
# create ELMo features
python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/elmo 
# train the parser
python train.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul  
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder sample_run/parser --system_tb sample_run/en_lines-ud-test_pred.conllu --gold_tb sample_run/gold --tb_out sample_run/en_lines-test-pred.conllu --lexicon sample_run/UDLex_English-Apertium.conllul --word_path sample_run/cc.en.300.vec.sample
fi

if [ "$1" == "elmo" ] ; then
echo RUN with elmo 
_id_run=$(uuidgen)
MODEL_ID=${_id_run:0:6}
#MODEL_NAME="$MODEL_ID-REAL_ELMO"
PROJECT_PATH="/home/benjamin/parsing/ELMolex_sosweet"


MODEL_NAME="cf6257-REAL_ELMO"

PARSER_PATH=$PROJECT_PATH/sosweet_run/parser_models/$MODEL_NAME-parser
test_data="cb2"
PREDICTION_PATH=$PARSER_PATH/predictions
DATA_PATH="$PROJECT_PATH/data/ud_tag-gold_dep"
DATA_SET_NAME="cb2+fqb+ftb"
TRAINING_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.train-ud_pred_tag_only-01-udpipe1.2.conllu"
DEV_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.dev-ud_pred_tag_only-01-udpipe1.2.conllu"
TEST_SET="${DATA_PATH}${TEST_FOLDER}/$DATA_SET_NAME.test-ud_pred_tag_only-01-udpipe1.2.conllu"
_SYSTEM_DATA_PRED_TAG="${DATA_PATH}$TEST_FOLDER/$test_data.test-ud_pred_tag_only-01-udpipe1.2.conllu"

GPU=1
ELMO=1
POS=1
#char_script="--char"
char_script=""
FAIR_VECTOR_PATH="/home/benjamin/parsing/NeuroTagger/word_embedding/fasttext_vector"
ELMO_PATH="$PROJECT_PATH/sosweet_run/elmo_models/$MODEL_NAME-elmo"
LEXICON=0
RANDOM_INIT=0
EPOCHS=1

#mkdir $PARSER_PATH
#mkdir $PREDICTION_PATH
#mkdir $ELMO_PATH
test_data="cb1"
#_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/$test_data.test.conll"
_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/cb2-w_hash.test.conll"
#_SYSTEM_DATA_PRED_TAG="${DATA_PATH}$TEST_FOLDER/$test_data.test-ud_pred_tag_only-01-udpipe1.2.conllu"
_SYSTEM_DATA_PRED_TAG="$_GOLD_DATA"

#python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path $TRAINING_SET --dev_path $DEV_SET --test_path $TEST_SET --dest_path sample_run/$MODEL_NAME_elmo 
#CUDA_VISIBLE_DEVICES=$GPU python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample \
#-train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path $ELMO_PATH --num_epochs 1
ELMO=0
python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample \
--train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path "$PARSER_PATH" \
--lexicon $LEXICON --random_init $RANDOM_INIT  --num_epochs $EPOCHS --elmo $ELMO --pos $POS  $char_script --prelstm_args "/home/benjamin/parsing/ELMolex_sosweet/elmo_sosweet/95391-deff507-ELMO-weights.hdf5" --batch_size 33


#CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample  --sosweet 1  > $PARSER_PATH/report_full.txt

#_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/cb1-1_2-w_hash.test.conll"
#_SYSTEM_DATA_PRED_TAG="$_GOLD_DATA"

#CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample  --sosweet 1  >> $PARSER_PATH/report_full.txt

_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/cb1-w_hash.test.conll"
_SYSTEM_DATA_PRED_TAG="$_GOLD_DATA"

#CUDA_VISIBLE_DEVICES=$GPU 
#python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample  --sosweet 1  --batch_size 32
#>> $PARSER_PATH/report_full.txt 

#>> $PARSER_PATH/report.txt 


#"/home/benjamin/parsing/ELMolex_sosweet/sosweet_run/elmo_models/TEST_ELMO_APPP-elmo/args.json"
#"/home/benjamin/parsing/ELMolex_sosweet/elmo_sosweet/95391-deff507-ELMO-weights.hdf5" \
#python train.py --word_path sample_run/cc.en.300.vec.sample --train_path sample_run/en_lines-ud-train.conllu --dev_path sample_run/en_lines-ud-dev.conllu --test_path sample_run/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul  
#python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$TEST_SET" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec.sample 
fi 

