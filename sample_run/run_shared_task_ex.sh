python nlm.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "${DATA_PATH}${TRAIN_FOLDER}/fr_sequoia-ud-train.conllu" --dev_path "${DATA_PATH}${TRAIN_FOLDER}/fr_sequoia-ud-dev.conllu" --test_path "${DATA_PATH}/${TEST_FOLDER}/fr_sequoia-ud-test.conllu" --dest_path $ELMO_PATH --num_epochs 1
# train the parser
python train.py --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --train_path "${DATA_PATH}${TRAIN_FOLDER}/fr_sequoia-ud-train.conllu" --dev_path "$DATA_PATH${TRAIN_FOLDER}/fr_sequoia-ud-dev.conllu" --test_path "${DATA_PATH}${TEST_FOLDER}fr_sequoia-ud-test.conllu" --dest_path "$PARSER_PATH" --prelstm_args $ELMO_PATH/args.json --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul  --num_epochs 1
# predict the parse tree and compute CONLL'18 scores
python test.py --pred_folder "$PARSER_PATH" --system_tb $DATA_PATH/eval-ud.conllu --gold_tb "$GOLD_DATA" --tb_out "$PROJECT_PATH/sosweet_run/fr_sequoia-test-pred.conllu" --lexicon $PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul --word_path $FAIR_VECTOR_PATH/cc.fr.300.vec --epoch -1
fi
