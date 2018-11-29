#!/bin/bash

# clean the old runs (if any)
#rm -rf sample_run/elmo
#rm -rf sample_run/parser
source activate conll18
PROJECT_PATH="/home/benjamin/parsing/ELMolex_sosweet"
## Caution: Pre-trained word embedding file 'sample_run/cc.en.300.vec.sample' has to be replaced with the raw file extracted from https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.en.300.vec.gz to reproduce ELMoLex results. The default value for the hyper-parameters are optimal and used for our final submission.
#FAIR_VECTOR_PATH="$PROJECT_PATH/../NeuroTagger/word_embedding/fasttext_vector"
#DATA_PATH="$PROJECT_PATH/../data/release-2.2-st-train-dev-data-NORMALIZED-splits/direct/ud-fr_sequoia"
#TRAIN_FOLDER="" 
#TEST_FOLDER=""
DATA_PATH="$PROJECT_PATH/../data/sosweet"
# DEV assumed to be in train
#TEST_FOLDER="/test_2_conll"
#DATA_SET_NAME="cb2+fqb+ftb"
# GOLD DATA 
#GOLD_DATA="$DATA_PATH/fr_sequoia-ud-test.conllu"
#GOLD_DATA="${DATA_PATH}$TEST_FOLDER/cb1.test.conll"
# we work in gold tokenization and gold tagging for now 
# TODO : add pred tags from udpipe or from neurotagger
#SYSTEM_DATA=$GOLD_DATA
#SOSWEET_EVAL=1
_id_run=$(uuidgen)
MODEL_ID=$1
SUFFIX=$2
MODEL_NAME=$MODEL_ID-$SUFFIX

ELMO_PATH=$PROJECT_PATH/sosweet_run/elmo_models/$MODEL_NAME-elmo
PARSER_PATH=$PROJECT_PATH/sosweet_run/parser_models/$MODEL_NAME-parser

mkdir $ELMO_PATH
mkdir $PARSER_PATH

echo "$ELMO_PATH and $PARSER_PATH made"


TEST_DIR="/home/benjamin/parsing/ELMolex_sosweet/data/ud_output"

report="./reports/$MODEL_NAME-reports.txt"
echo "Writing to $report "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We train and test on gold tags for now" 
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET , final scoring on $GOLD_DATA"
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report
echo "We train and test on gold tags for now" >> $report
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET , final scoring on $GOLD_DATA" >> $report

# evaluating on cb1
for test_data in "cb1" "cb2" "$DATA_SET_NAME" ; 
do 
_GOLD_DATA="${DATA_PATH}$TEST_FOLDER/$test_data.test.conll"
_SYSTEM_DATA="${TEST_DIR}/$test_data-tag+parse.conllu"
# tweets preprocessing --> select only the id and the text --> one sentence per row 
# udpipe tokenizer and tagger
# add an argument for only prediction 
python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA" --gold_tb "$_GOLD_DATA" --tb_out "$PROJECT_PATH/sosweet_run/$test_data-test-pred-UD.conll" --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet $SOSWEET_EVAL >> $report 
done 
# then parallelize everything



