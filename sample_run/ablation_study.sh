#!/bin/bash

# clean the old runs (if any)
#rm -rf sample_run/o
#rm -rf sample_run/parser
# Environment variables 
source activate conll18
PROJECT_PATH="/home/benjamin/parsing/ELMolex_sosweet"
FAIR_VECTOR_PATH="/home/benjamin/parsing/NeuroTagger/word_embedding/fasttext_vector"
# Caution: Pre-trained word embedding file 'sample_run/cc.en.300.vec.sample' has to be replaced with the raw file extracted from https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.en.300.vec.gz to reproduce ELMoLex results. The default value for the hyper-parameters are optimal and used for our final submission.

_id_run=$(uuidgen)
MODEL_ID=${_id_run:0:6}


while getopts s:g:e:r:l:n:k:p:c: option
do
case "${option}"
in
s) SUFFIX=${OPTARG};;
g) GPU=${OPTARG};;
e) ELMO=${OPTARG};;
r) RANDOM_INIT=${OPTARG};;
l) LEXICON=${OPTARG};;
n) EPOCHS=${OPTARG};;
k) RUN_ID=${OPTARG};;
p) POS=${OPTARG};;
c) CHAR=${OPTARG};;
esac
done
# default parameters and assertions
if [ -z "$SUFFIX"  ]
then
        SUFFIX=TEST
        echo "ARGUMENTS : setting up SUFFIX to $SUFFIX"
fi
if [ -z "$GPU"  ]
then
        GPU=0
        echo "ARGUMENTS : setting up GPU to $GPU"
fi
if [ -z "$EPOCHS"  ]
then
        EPOCHS=1
        echo "ARGUMENTS : setting up EPOCHS to $EPOCHS"
fi
if [ -z "$LEXICON"  ]
then
    echo Missing arguments $LEXICON LEXICON
    exit n 
else 
    if [  "$LEXICON" ==  "0" ] 
    then
        lex_label=$LEXICON
    else 
        lex_label="1"
    fi
    echo lex_label is $lex_label 
fi
if [ -z "$POS"  ]
then
        POS=1
        echo "!! ARGUMENTS : setting up POS to $POS"
fi
if [ -z "$CHAR"  ]
then
        CHAR=1
        echo "!! ARGUMENTS : setting up CHAR to $CHAR"
fi
if [ -z "$ELMO"  ]
then
    echo Missing arguments $ELMO ELMO
    exit n 
fi
if [ -z "$RANDOM_INIT"  ]
then
    echo Missing arguments $RANDOM_INIT RANDOM_INIT
    exit n 
fi
if [ -z "$RUN_ID"  ]
then
	RUN_ID=0
    echo Run id set to 0
fi
# ARGUMENTS 
#SUFFIX=$1
#GPU=$2
#ELMO=0
#RANDOM_INIT=0
#LEXICON=0
PRED_TAG_EVAL="1"
#EPOCHS=1

MODEL_NAME=$RUN_ID-$MODEL_ID-$SUFFIX
ELMO_PATH=$PROJECT_PATH/sosweet_run/elmo_models/$MODEL_NAME-elmo
PARSER_PATH=$PROJECT_PATH/sosweet_run/parser_models/$MODEL_NAME-parser
PREDICTION_PATH=$PARSER_PATH/predictions

mkdir $ELMO_PATH
mkdir $PARSER_PATH
mkdir $PREDICTION_PATH

DATA_PATH="$PROJECT_PATH/data/ud_tag-gold_dep"
TRAIN_FOLDER="" 
TEST_FOLDER=""
DATA_SET_NAME="cb2+fqb+ftb"
TRAINING_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.train-ud_pred_tag_only-01-udpipe1.2.conllu"
DEV_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.dev-ud_pred_tag_only-01-udpipe1.2.conllu"
TEST_SET="${DATA_PATH}${TEST_FOLDER}/$DATA_SET_NAME.test-ud_pred_tag_only-01-udpipe1.2.conllu"

report_perf="$PROJECT_PATH/reports/$MODEL_NAME-performance.txt"
report="$PROJECT_PATH/reports/$MODEL_NAME-reports.txt"


echo "Writing to $report "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We train and test on gold tags for now" 
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET"
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report
echo "We train and test on gold tags for now" >> $report
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET" >> $report


# create ELMo features
#CUDA_VISIBLE_DEVICES=$GPU python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path $ELMO_PATH 
# train the parser
echo "DEBUG POS = $POS " 

if [ "$CHAR" == "0" ]; then
    char_script="--char"
elif [ "$CHAR" == "1" ]; then
    char_script=""
fi

if [ "$ELMO" == "1" ] ; then
echo "------------------------------------------   : TRAIN ELMO"
CUDA_VISIBLE_DEVICES=$GPU python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec \
--train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path $ELMO_PATH --num_epochs 40
echo "------------------------------------------ : TRAIN PARSER WITH ELMO  "
# --elmo ?? 
CUDA_VISIBLE_DEVICES=$GPU python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec \
--train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path "$PARSER_PATH" \
--lexicon $LEXICON --random_init $RANDOM_INIT  --num_epochs $EPOCHS --prelstm_args $ELMO_PATH/args.json \
--elmo $ELMO --pos $POS  $char_script 

elif [ "$ELMO" == "0" ]; then
echo "------------------------------------------   : TRAIN PARSER  no elmo "

CUDA_VISIBLE_DEVICES=$GPU python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec \
--train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path "$PARSER_PATH" \
--lexicon $LEXICON --random_init $RANDOM_INIT  --num_epochs $EPOCHS --elmo $ELMO --pos $POS $char_script 
fi
#--lexicon "$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul" \
# predict the parse tree and compute CONLL'18 scores
echo "MODEL_NAME : $MODEL_NAME training done" > $report_perf

# evaluating on cb1

for test_data in "cb2" "$DATA_SET_NAME" "cb1"  ; 
#for test_data in "cb1"
do 
#_GOLD_DATA="${DATA_PATH}$TEST_FOLDER/$test_data.test.conll"
_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/$test_data.test.conll"
_SYSTEM_DATA="${_GOLD_DATA}"
#_SYSTEM_DATA_PRED_TAG="${TEST_DIR}/$test_data-tag+parse.conllu"
_SYSTEM_DATA_PRED_TAG="${DATA_PATH}$TEST_FOLDER/$test_data.test-ud_pred_tag_only-01-udpipe1.2.conllu"

echo "Starting prediction on $test_data on gold tags"
CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet 1 >> $report 

#$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul 

if [ "$PRED_TAG_EVAL" == "1" ] ; then 
echo "SYSTEL ORED TAG $_SYSTEM_DATA_PRED_TAG"
echo "Starting evaluation on $test_data on pred tags"
CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred-UD.conll" --lexicon $LEXICON --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet 1 >> $report 
fi 

# evaluating on IN Domain test
echo "Starting evaluation on $test_data 1"
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT ------------------------" >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred.conll"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred.conll"  > $report.$test_data
if [ "$PRED_TAG_EVAL" == "1" ] ; then 
echo "Starting evaluation on $test_data 1"
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT PRED ------------------------" >> $report    
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred-UD.conll"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred-UD.conll"  > $report.$test_data.pred_tags 
fi 

echo "SCORE ON $test_data GOLD " >> $report_perf
grep  " Labeled   attachment score:" $report.$test_data >> $report_perf
grep  " Unlabeled attachment score:" $report.$test_data >> $report_perf

if [ "$PRED_TAG_EVAL" == "1" ] ; then 
echo "SCORE ON $test_data PRED " >> $report_perf
grep  " Labeled   attachment score:" $report.$test_data.pred_tags  >> $report_perf
grep  " Unlabeled attachment score:" $report.$test_data.pred_tags  >> $report_perf
fi 

echo "report perf written $report_perf with $test_data test as well as $report.$test_data report "
echo "full report written $report " 
done 

python misc/report2json.py --model_id $MODEL_ID --model_info $MODEL_NAME-${POS}_pos-${ELMO}_elmo-${RANDOM_INIT}_randominit-${CHAR}_char-${lex_label}_lex_label --training_info $DATA_SET_NAME --tag both --dir_report $report_perf --target_dir "$PROJECT_PATH/reports/$MODEL_NAME-performance.json"
echo  "$MODEL_NAME-${POS}_pos-${ELMO}_elmo-${RANDOM_INIT}_randominit-${CHAR}_char-${lex_label}_lex_label" ablation_study.sh completed >> $PROJECT_PATH/reports/$RUN_ID-completion_logs.txt


