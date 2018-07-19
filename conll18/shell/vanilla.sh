#upos+xpos
#rm -rf /home/ganesh/objects/neurogp/en/vanillauposxpos
#CUDA_VISIBLE_DEVICES=0 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/vanillauposxpos --num_epochs 1
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/vanillauposxpos

#nlminit10
#rm -rf /home/ganesh/objects/neurogp/en/nlminit10
#CUDA_VISIBLE_DEVICES=1 python nlm.py --dest_path /home/ganesh/objects/neurogp/en/nlminit10/lm --bptt 10 --num_epoch 1
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon None --prelstm_args /home/ganesh/objects/neurogp/en/nlminit10/lm/args.json --dest_path /home/ganesh/objects/neurogp/en/nlminit10
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/nlminit10

#elmohid150
#rm -rf /home/ganesh/objects/neurogp/en/elmohid150
#CUDA_VISIBLE_DEVICES=1 python nlm.py --dest_path /home/ganesh/objects/neurogp/en/elmohid150/lm --bptt 10 --hidden_size 150
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon None --prelstm_args /home/ganesh/objects/neurogp/en/elmohid150/lm/args.json --dest_path /home/ganesh/objects/neurogp/en/elmohid150 --elmo
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/elmohid150

#lex_hot
#rm -rf /home/ganesh/objects/neurogp/en/lex_hot_drop_fallback
#CUDA_VISIBLE_DEVICES=0 python train.py --lex_hot --dest_path /home/ganesh/objects/neurogp/en/lex_hot_drop_fallback
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/lex_hot_drop_fallback

#lex_embed_none
#rm -rf /home/ganesh/objects/neurogp/en/lex_embed_none_drop_fallback
#CUDA_VISIBLE_DEVICES=0 python train.py --lex_attn None --dest_path /home/ganesh/objects/neurogp/en/lex_embed_none_drop_fallback
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/lex_embed_none_drop_fallback

#lex_attn_specific
#rm -rf /home/ganesh/objects/neurogp/en/lex_attn_specific_drop_fallback
#CUDA_VISIBLE_DEVICES=0 python train.py --lex_attn Specific --dest_path /home/ganesh/objects/neurogp/en/lex_attn_specific_drop_fallback --num_epochs 1 --prelstm_args None.json
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/lex_attn_specific_drop_fallback

#lex_attn_group
#rm -rf /home/ganesh/objects/neurogp/en/lex_attn_group_drop_fallback
#CUDA_VISIBLE_DEVICES=0 python train.py --lex_attn Group --dest_path /home/ganesh/objects/neurogp/en/lex_attn_group_drop_fallback
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/lex_attn_group_drop_fallback

#mwe_sequoia
#rm -rf /home/ganesh/objects/neurogp/en/mwe_sequoia
#CUDA_VISIBLE_DEVICES=0 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/mwe_sequoia --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.fr.300.vec --train_path /home/ganesh/objects/conll18/udpipe-trained/direct/ud-fr_sequoia/fr_sequoia-ud-model-train.conllu --dev_path /home/ganesh/objects/conll18/udpipe-trained/direct/ud-fr_sequoia/fr_sequoia-ud-model-dev.conllu --test_path /home/ganesh/objects/conll18/udpipe-trained/direct/ud-fr_sequoia/fr_sequoia-ud-model-test.conllu --num_epochs 1
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/mwe_sequoia --system_tb /home/ganesh/objects/conll18/udpipe-trained/direct/ud-fr_sequoia/fr_sequoia-ud-eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/udpipe-trained/direct/ud-fr_sequoia/fr_sequoia-ud-eval-gold.conllu

#en_bsize_16
#rm -rf /home/ganesh/objects/neurogp/en/en_bsize_16
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/en_bsize_16 --batch_size 16
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/en_bsize_16

#en_bsize_8
#rm -rf /home/ganesh/objects/neurogp/en/en_bsize_8
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/en_bsize_8 --batch_size 8
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/en_bsize_8

#mwe_galician
#rm -rf /home/ganesh/objects/neurogp/en/mwe_galician
#CUDA_VISIBLE_DEVICES=0 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/mwe_galician --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --train_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-train.conllu --dev_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-dev.conllu --test_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-test.conllu --num_epochs 1
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/mwe_galician --system_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-gold.conllu

#search_no_attn
#rm -rf /home/ganesh/objects/neurogp/en/search_no_attn
#CUDA_VISIBLE_DEVICES=1 python nlm.py --dest_path /home/ganesh/objects/neurogp/en/search_no_attn/lm --bptt 10 --hidden_size 100 --num_epochs 1 --pos
#mkdir /home/ganesh/objects/neurogp/en/search_no_attn/search
#java -jar tmp/jars/indexer.jar /home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-train.conllu /home/ganesh/objects/neurogp/en/search_no_attn/search
#java -jar tmp/jars/searcher.jar /home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-train.conllu /home/ganesh/objects/neurogp/en/search_no_attn/search 101 /home/ganesh/objects/neurogp/en/search_no_attn/search/train_hits.txt
#java -jar tmp/jars/searcher.jar /home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-dev.conllu /home/ganesh/objects/neurogp/en/search_no_attn/search 100 /home/ganesh/objects/neurogp/en/search_no_attn/search/dev_hits.txt
#java -jar tmp/jars/searcher.jar /home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-test.conllu /home/ganesh/objects/neurogp/en/search_no_attn/search 100 /home/ganesh/objects/neurogp/en/search_no_attn/search/test_hits.txt
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon None --prelstm_args /home/ganesh/objects/neurogp/en/search_no_attn/lm/args.json --dest_path /home/ganesh/objects/neurogp/en/search_no_attn --search --num_epochs 1
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/search_no_attn

#elmo_lex
#rm -rf /home/ganesh/objects/neurogp/en/elmo_lex
#CUDA_VISIBLE_DEVICES=1 python nlm.py --dest_path /home/ganesh/objects/neurogp/en/elmo_lex/lm --bptt 10 --hidden_size 150 --num_epochs 1
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon /home/ganesh/data/conll/UDLexicons.0.2/UDLex_English-Apertium.conllul --lex_attn Specific --prelstm_args /home/ganesh/objects/neurogp/en/elmo_lex/lm/args.json --dest_path /home/ganesh/objects/neurogp/en/elmo_lex --elmo --num_epochs 1
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/elmo_lex

#test_lex_expand
#rm -rf /home/ganesh/objects/neurogp/en/test_lex_expand
#CUDA_VISIBLE_DEVICES=0 python train.py --lex_attn Specific --dest_path /home/ganesh/objects/neurogp/en/test_lex_expand --num_epochs 1 --prelstm_args None.json --lex_trim
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_lex_expand --lex_expand

#test_w2v_expansion
#rm -rf /home/ganesh/objects/neurogp/en/test_w2v_expansion
#CUDA_VISIBLE_DEVICES=0 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/test_w2v_expansion --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --train_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-train.conllu --dev_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-dev.conllu --test_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-test.conllu --num_epochs 1 --prelstm_args None.json --vocab_trim
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_w2v_expansion --vocab_expand --system_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-gold.conllu --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec

#test_w2v_transformation
#rm -rf /home/ganesh/objects/neurogp/en/test_w2v_transformation
#CUDA_VISIBLE_DEVICES=0 python train.py --lexicon None --dest_path /home/ganesh/objects/neurogp/en/test_w2v_transformation --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --train_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-train.conllu --dev_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-dev.conllu --test_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-test.conllu --num_epochs 150 --prelstm_args None.json --vocab_trim
#python ltrans.py --dest_path /home/ganesh/objects/neurogp/en/test_w2v_transformation --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --train_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-train.conllu #> out_ltrans
python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_w2v_transformation --system_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-gold.conllu --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec > out_test1
python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_w2v_transformation --system_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-gold.conllu --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --vocab_expand > out_test2
python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_w2v_transformation --system_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-ud.conllu --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/eval-gold.conllu --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --vocab_expand --linear_transform > out_test3

#test_upload
#rm -rf /home/ganesh/objects/neurogp/en/test_upload
#CUDA_VISIBLE_DEVICES=1 python nlm.py --dest_path /home/ganesh/objects/neurogp/en/test_upload/elmo --bptt 10 --hidden_size 150 --num_epochs 1
#CUDA_VISIBLE_DEVICES=1 python train.py --lexicon /home/ganesh/data/conll/UDLexicons.0.2/UDLex_English-Apertium.conllul --lex_attn Specific --prelstm_args /home/ganesh/objects/neurogp/en/test_upload/elmo/args.json --dest_path /home/ganesh/objects/neurogp/en/test_upload --elmo --num_epochs 1
#python ltrans.py --dest_path /home/ganesh/objects/neurogp/en/test_upload --word_path /home/ganesh/data/conll/fair_vectors_raw/cc.gl.300.vec --train_path /home/ganesh/objects/conll18/june16_splits/direct/Galician-CTG/model-train.conllu
#python test.py --pred_folder /home/ganesh/objects/neurogp/en/test_upload

#test_arabic
#python test.py --pred_folder /home/ganesh/objects/finale/upload_june29/Arabic-PADT --system_tb /home/benjamin/parsing/data/release-2.2-st-train-dev-data-NORMALIZED-splits/direct/ud-ar_padt/eval-ud.conllu.post --gold_tb /home/ganesh/objects/conll18/june16_splits/direct/Arabic-PADT/eval-gold.conllu




