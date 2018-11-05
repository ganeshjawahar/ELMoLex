
EPOCHS=$2
LABEL=$1
PROJECT_PATH="/home/benjamin/parsing/ELMolex_sosweet"
LEXICON_PATH="$PROJECT_PATH/sosweet_run/lexicons/UDLex_French-Lefff.conllul"
#lexicon=/home/benjamin/parsing/ELMolex_sosweet/sosweet_run/lexicons/UDLex_French-Lefff.conllul
_id_run=$(uuidgen)
RUN_ID=${_id_run:0:6}
for elmo in  0 1 ; do 
	for lexicon in  0 1 ; do 
	#for lexicon in  1 0; do 
		for pos in 0  ; do 
			for char in 0 1  ; do 
				for random_init in 0 1 ; do 
				#for random_init in 0 1; do 
					if [ "$lexicon" == "1" ] ; then 
						lexicon=$LEXICON_PATH
					fi 
					random=`shuf -i1-2 -n1`
					GPU=$(($random - 1))
					echo sh $PROJECT_PATH/sample_run/ablation_study.sh -s $1 -k $RUN_ID -l $lexicon -r $random_init -e $elmo -n $EPOCHS -g $GPU -p $pos -c $char >> $PROJECT_PATH/runs/$RUN_ID-$LABEL-run_$GPU.sh 
				done
			done 
		done
	done
done
echo $PROJECT_PATH/runs/$RUN_ID-$LABEL-run_0.sh , $PROJECT_PATH/runs/$RUN_ID-$LABEL-run_1.sh 