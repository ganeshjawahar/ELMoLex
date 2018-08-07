''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Instructions for creating training and testing scripts for the finale run
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Step-1:
=======
Set the following environment variables (if not done before):
export CL_TB_GOLD=/home/ganesh/data/conll/release-2.2-st-train-dev-data/release-2.2-st-train-dev-data/ud-treebanks-v2.2  (path to the http://ufal.mff.cuni.cz/~zeman/soubory/release-2.2-st-train-dev-data.zip)
export CL_TB_UD=/home/ganesh/data/conll/ud-2.2-conll18-crossfold-morphology (path to the http://ufal.mff.cuni.cz/~zeman/soubory/ud-2.2-conll18-crossfold-morphology.tar.xz)
export CL_UD_MODEL=/home/ganesh/data/conll/ud-2.2-conll18-baseline-models/models (path to the http://ufal.mff.cuni.cz/~zeman/soubory/ud-2.2-conll18-baseline-models.tar.xz)
export CL_WORD_VEC=/home/ganesh/data/conll/fair_vectors_raw (path to vectors downloaded from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
export CL_LEX_LAT=/home/ganesh/data/conll/UDLexicons.0.2 (path to the http://pauillac.inria.fr/~sagot/index.html#udlexicons)
export CL_HOME=/home/ganesh/objects/post_finale (path to an empty folder where all the intermediate files created during training would be stored.)

Step-2:
=======
Prepare the training and dev. data
Update conll18/resources/ben_tags.txt with the recent tags
cd ELMoLex/
python conll18/py/createFinaleData.py

Step-3:
=======
Create the finale training scripts
cd ELMoLex/
python conll18/py/createFinaleScripts.py

Step-4:
=======
Create the finale testing scripts
cd ELMoLex/
python conll18/py/createFinaleTest.py






