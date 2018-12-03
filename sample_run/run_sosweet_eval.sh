#!/bin/bash

# clean the old runs (if any)
#rm -rf sample_run/elmo
#rm -rf sample_run/parser
source activate conll18
PROJECT_PATH="/home/benjamin/parsing/ELMolex_sosweet"
# Caution: Pre-trained word embedding file 'sample_run/cc.en.300.vec.sample' has to be replaced with the raw file extracted from https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.en.300.vec.gz to reproduce ELMoLex results. The default value for the hyper-parameters are optimal and used for our final submission.
FAIR_VECTOR_PATH="$PROJECT_PATH/../NeuroTagger/word_embedding/fasttext_vector"
#DATA_PATH="$PROJECT_PATH/../data/release-2.2-st-train-dev-data-NORMALIZED-splits/direct/ud-fr_sequoia"
#TRAIN_FOLDER="" 
#TEST_FOLDER=""
DATA_PATH="$PROJECT_PATH/../data/sosweet"
# DEV assumed to be in train
TEST_FOLDER="/test_2_conll"
#DATA_SET_NAME="cb2+fqb+ftb"
# GOLD DATA 
#GOLD_DATA="$DATA_PATH/fr_sequoia-ud-test.conllu"
#GOLD_DATA="${DATA_PATH}$TEST_FOLDER/cb1.test.conll"
# we work in gold tokenization and gold tagging for now 
# TODO : add pred tags from udpipe or from neurotagger
#SYSTEM_DATA=$GOLD_DATA
#SOSWEET_EVAL=1
#_id_run=$(uuidgen)
#MODEL_ID=$1
#SUFFIX=$2
#MODEL_NAME=$MODEL_ID-$SUFFIX

#ELMO_PATH=$PROJECT_PATH/sosweett_run/elmo_models/$MODEL_NAME-elmo

# ELMO trained on UGC mixed + ELMO trained on SoSweet 
#MODEL_NAME="cf6257-REAL_ELMO"
#MODEL_BEST_TINY_ELMO="8f0a3a-9c9409-TEST-parser"

for MODEL_NAME in  "8f0a3a-9c9409-TEST" "cf6257-REAL_ELMO" ; do 
#for MODEL_NAME in  "cf6257-REAL_ELMO" ; do 

echo "START EVALUATING $MODEL_NAME"
PARSER_PATH="$PROJECT_PATH/sosweet_run/parser_models/$MODEL_NAME-parser"
PREDICTION_PATH="$PARSER_PATH/predictions"
#mkdir $ELMO_PATH
#mkdir $PARSER_PATH

##echo "$ELMO_PATH and $PARSER_PATH made"

trained_on="cb2+fqb+ftb"
#TEST_DIR="/home/benjamin/parsing/ELMolex_sosweet/data/ud_output"
TEST_DIR="/home/benjamin/parsing/ELMolex_sosweet/data/ud_tag-gold_dep"

EXTRA_LABEL="A"
report="$PROJECT_PATH/reports/$MODEL_NAME-reports-$EXTRA_LABEL.txt"
report_perf="$PROJECT_PATH/reports/$MODEL_NAME-performance-$EXTRA_LABEL.txt"

echo "WRITING to $report and $report_perf "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report

# model parameter 
if [ "$MODEL_NAME" == "8f0a3a-9c9409-TEST" ] ; then 
LEXICON="$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul"
fi 	
if [ "$MODEL_NAME" == "cf6257-REAL_ELMO" ] ; then 
LEXICON="0"
fi 

w2v_dir="$FAIR_VECTOR_PATH/cc.fr.300.vec"
GPU=0
# evaluating on cb1
#for test_data in "cb1" "cb1-1_2" "cb2" "$DATA_SET_NAME" "fqb+ftb" "fqb" "ftb"  ; 
for test_data in "cb1" "cb1-1_2" "cb1-2_2" "cb2"  "fqb+ftb" "ftb" "fqb" "cb2+fqb+ftb" ; do
_GOLD_DATA="${DATA_PATH}$TEST_FOLDER/$test_data.test.conll"
#_SYSTEM_DATA="${TEST_DIR}/$test_data-tag+parse.conllu"
_SYSTEM_DATA="${TEST_DIR}/$test_data.test-ud_pred_tag_only-01-udpipe1.2.conllu"
# gold tag

for data_to_test in "pred_tag" "gold_tag" ; do 
if [ "$data_to_test" == "pred_tag" ] ; then 
_TB="$PREDICTION_PATH/$test_data-test-pred_tag.conll"
_SYSTEM="$_SYSTEM_DATA"
fi 
if [ "$data_to_test" == "gold_tag" ] ; then  
_TB="$PREDICTION_PATH/$test_data-test-gold_tag.conll"
_SYSTEM="$_GOLD_DATA"
fi 
#_TB_OUT_GOLD_TAG="$PREDICTION_PATH/$test_data-test-gold_tag.conll"
#_TB_OUT_PRED_TAG="$PREDICTION_PATH/$test_data-test-pred_tag.conll"

#CUDA_VISIBLE_DEVICES=$GPU 
python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM" --gold_tb "$_GOLD_DATA" --tb_out $_TB --lexicon $LEXICON --word_path "$w2v_dir"  --sosweet 1 
# --batch_size 32
# >> $report 
#python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA" --gold_tb "$_GOLD_DATA" --tb_out "$PROJECT_PATH/sosweet_run/$test_data-test-pred-UD.conll" --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet $SOSWEET_EVAL >> $report 

# pred tag 
#CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out $_TB --lexicon $LEXICON --word_path "$w2v_dir"  --sosweet 1  >> $PARSER_PATH/report_full.txt

# evaluating on IN Domain test
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT ------------------------" >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$_TB"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$_TB"  >> $report.$test_data.$data_to_test
echo "SCORE ON $test_data - $data_to_test (of $_TB) " >> $report_perf
echo "WRITING to $report.$test_data.$data_to_test "
LAS=`grep  " Labeled   attachment score:" $report.$test_data.$data_to_test `
UAS=`grep  " Unlabeled attachment score:" $report.$test_data.$data_to_test`
echo "model:$MODEL_NAME,test:$test_data;pipe:$data_to_test;train:$trained_on;LAS:$LAS;UAS:$UAS" >> $report_perf
echo "WRITTEN to  $report_perf : Test on : $test_data $data_to_test trained on : $trained_on , LAS:$LAS,UAS:$UAS "
echo "WRITTEN report perf written $report_perf with $test_data test as well as $report.$test_data.$data_to_test report "
echo "WRITTEN full report written $report " 
done
done 

done 
