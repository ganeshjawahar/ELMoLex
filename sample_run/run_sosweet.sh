
# clean the old runs (if any)
#rm -rf sample_run/o
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
TRAIN_FOLDER="/train_2_conll" 
TEST_FOLDER="/test_2_conll"
DATA_SET_NAME="cb2+fqb+ftb"
# GOLD DATA 
#GOLD_DATA="$DATA_PATH/fr_sequoia-ud-test.conllu"

# we work in gold tokenization and gold tagging for now 
# TODO : add pred tags from udpipe or from neurotagger
SOSWEET_EVAL=1
_id_run=$(uuidgen)
MODEL_ID=${_id_run:0:6}
SUFFIX=$2
GPU=$3
MODEL_NAME=$MODEL_ID-$SUFFIX

ELMO_PATH=$PROJECT_PATH/sosweet_run/elmo_models/$MODEL_NAME-elmo
PARSER_PATH=$PROJECT_PATH/sosweet_run/parser_models/$MODEL_NAME-parser
PREDICTION_PATH=$PARSER_PATH/predictions

mkdir $ELMO_PATH
mkdir $PARSER_PATH
mkdir $PREDICTION_PATH


echo "$ELMO_PATH and $PARSER_PATH made"

TRAINING_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.train.conll"
DEV_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.dev.conll"
TEST_SET="${DATA_PATH}${TEST_FOLDER}/$DATA_SET_NAME.test.conll"

report="./reports/$MODEL_NAME-reports.txt"


if [ $1 == "check" ] ; then

echo "Writing to $report "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We train and test on gold tags for now" 
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET"
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report
echo "We train and test on gold tags for now" >> $report
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET" >> $report

# run for 1 epoch
# create ELMo features
python nlm.py --word_path sample_run/cc.en.300.vec.sample --train_path data/en_lines-ud-train.conllu --dev_path data/en_lines-ud-dev.conllu --test_path data/en_lines-ud-test.conllu --dest_path sample_run/elmo --num_epochs 40
# train the parser
python train.py --word_path sample_run/cc.en.300.vec.sample --train_path data/en_lines-ud-train.conllu --dev_path data/en_lines-ud-dev.conllu --test_path data/en_lines-ud-test.conllu --dest_path sample_run/parser --prelstm_args sample_run/elmo/args.json --lexicon sample_run/UDLex_English-Apertium.conllul --num_epochs 150  
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder sample_run/parser --system_tb sample_run/en_lines-ud-test_pred.conllu --gold_tb sample_run/gold --tb_out sample_run/en_lines-test-pred.conllu --lexicon sample_run/UDLex_English-Apertium.conllul --word_path sample_run/cc.en.300.vec.sample
fi

if [ $1 == "complete" ] ; then
# run for recommended no. of epochs
# create ELMo features
echo "Writing to $report "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We train and test on gold tags for now" 
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET"
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report
echo "We train and test on gold tags for now" >> $report
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET" >> $report

CUDA_VISIBLE_DEVICES=1 python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path $ELMO_PATH 
# train the parser
CUDA_VISIBLE_DEVICES=1 python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path "$PARSER_PATH" --prelstm_args $ELMO_PATH/args.json --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul  
# predict the parse tree and compute CONLL'18 scores
echo "MODEL_NAME : $MODEL_NAME" > $report_perf

# evaluating on cb1
for test_data in "cb1" "cb2" "$DATA_SET_NAME" ; 
do 
_GOLD_DATA="${DATA_PATH}$TEST_FOLDER/$test_data.test.conll"
_SYSTEM_DATA="${_GOLD_DATA}"

python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA" --gold_tb "$_GOLD_DATA" --tb_out "$PROJECT_PATH/sosweet_run/$test_data-test-pred.conll" --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet $SOSWEET_EVAL >> $report 
# evaluating on IN Domain test
if [ "$SOSWEET_EVAL" == "1" ] ; then 
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT ------------------------" >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PROJECT_PATH/sosweet_run/$test_data-test-pred.conll"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PROJECT_PATH/sosweet_run/$test_data-test-pred.conll"  >> $report.$test_data
report_perf="./reports/$MODEL_NAME-performance.txt"
echo "SCORE ON $test_data " >> $report_perf
grep  " Labeled   attachment score:" $report.$test_data >> $report_perf
grep  " Unlabeled attachment score:" $report.$test_data >> $report_perf
echo "report perf written $report_perf with $test_data test as well as $report.$test_data report "
fi 
echo "full report written $report " 
done 
fi

TEST_DIR="/home/benjamin/parsing/ELMolex_sosweet/data/ud_output"
PRED_TAG="1"

if [ $1 == "base" ] ; then



# run for recommended no. of epochs
# create ELMo features
#CUDA_VISIBLE_DEVICES=1 python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path $ELMO_PATH 
# train the parser

DATA_PATH="$PROJECT_PATH/data/ud_tag-gold_dep"
TRAIN_FOLDER="" 
TEST_FOLDER=""
DATA_SET_NAME="cb2+fqb+ftb"
TRAINING_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.train-ud_pred_tag_only-01-udpipe1.2.conllu"
DEV_SET="${DATA_PATH}${TRAIN_FOLDER}/$DATA_SET_NAME.dev-ud_pred_tag_only-01-udpipe1.2.conllu"
TEST_SET="${DATA_PATH}${TEST_FOLDER}/$DATA_SET_NAME.test-ud_pred_tag_only-01-udpipe1.2.conllu"

report_perf="$PROJECT_PATH/reports/$MODEL_NAME-performance.txt"

echo "Writing to $report "
echo "MODEL_NAME : $MODEL_NAME" > $report
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" 
echo "We train and test on gold tags for now" 
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET"
echo "We work on gold tokenization (coming from ftb, fqb, cb2 spmrl ish transofmred to conll)" >> $report
echo "We train and test on gold tags for now" >> $report
echo "Training on $TRAINING_SET  validation on $DEV_SET , testing on $TEST_SET" >> $report


echo "------------------------------------------   : TRAIN"
CUDA_VISIBLE_DEVICES=$GPU python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --elmo 0  \
--train_path "$TRAINING_SET" --dev_path "$DEV_SET" --test_path "$TEST_SET" --dest_path "$PARSER_PATH" \
--lexicon 0 \
--random_init 1  --num_epochs 1

#--lexicon "$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul" \
# predict the parse tree and compute CONLL'18 scores
echo "MODEL_NAME : $MODEL_NAME (training done)" > $report_perf

# evaluating on cb1

for test_data in "cb1" "cb2" "$DATA_SET_NAME" ; 
#for test_data in "cb1"
do 
#_GOLD_DATA="${DATA_PATH}$TEST_FOLDER/$test_data.test.conll"
_GOLD_DATA="$PROJECT_PATH/../data/sosweet/test_2_conll/$test_data.test.conll"
_SYSTEM_DATA="${_GOLD_DATA}"
#_SYSTEM_DATA_PRED_TAG="${TEST_DIR}/$test_data-tag+parse.conllu"
_SYSTEM_DATA_PRED_TAG="${DATA_PATH}$TEST_FOLDER/$test_data.test-ud_pred_tag_only-01-udpipe1.2.conllu"

echo "Starting prediction on $test_data on gold tags"
CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred.conll" --lexicon 0 --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet $SOSWEET_EVAL >> $report 

#$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul 

if [ "$PRED_TAG" == "1" ] ; then 
	echo "Starting evaluation on $test_data on pred tags"
CUDA_VISIBLE_DEVICES=$GPU python test.py --pred_folder "$PARSER_PATH" --system_tb "$_SYSTEM_DATA_PRED_TAG" --gold_tb "$_GOLD_DATA" --tb_out "$PREDICTION_PATH/$test_data-test-pred-UD.conll" --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec  --sosweet $SOSWEET_EVAL >> $report 
fi 

# evaluating on IN Domain test
if [ "$SOSWEET_EVAL" == "1" ] ; then 
echo "Starting evaluation on $test_data o"
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT ------------------------" >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred.conll"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred.conll"  > $report.$test_data
if [ "$PRED_TAG" == "1" ] ; then 
evaluation_script="/home/benjamin/parsing/NeuroNLP2/evaluation/eval07.pl"
echo "------------------------------ ADDING REPORT PRED ------------------------" >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred-UD.conll"  >> $report 
perl $evaluation_script -g "$_GOLD_DATA" -s "$PREDICTION_PATH/$test_data-test-pred-UD.conll"  > $report.$test_data.pred_tags 
fi 

echo "SCORE ON $test_data GOLD " >> $report_perf
grep  " Labeled   attachment score:" $report.$test_data >> $report_perf
grep  " Unlabeled attachment score:" $report.$test_data >> $report_perf

if [ "$PRED_TAG" == "1" ] ; then 
echo "SCORE ON $test_data PRED " >> $report_perf
grep  " Labeled   attachment score:" $report.$test_data.pred_tags  >> $report_perf
grep  " Unlabeled attachment score:" $report.$test_data.pred_tags  >> $report_perf
fi 

echo "report perf written $report_perf with $test_data test as well as $report.$test_data report "
fi 
echo "full report written $report " 
done 
fi

python misc/report2json.py --model_id $MODEL_ID  --model_info $MODEL_NAME --training_info $DATA_SET_NAME --tag both --dir_report $report_perf --target_dir "$PROJECT_PATH/reports/$MODEL_NAME-performance.json"



