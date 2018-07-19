import sys
import glob
import os

RAW_TREEBANK="/home/ganesh/data/conll/release-2.2-st-train-dev-data-NORMALIZED"
WORD_EMBED="/home/ganesh/data/conll/word_embeddings"
FINA_WORD_EMBED="/home/ganesh/data/conll/fair_vectors_raw"

# get all langs
task_langs = {}
for folder in glob.glob(RAW_TREEBANK+"/ud-*"):
  fname = folder.split("/")[-1]
  lang = fname.split("-")[1].split('_')[0]
  task_langs[lang] = True

lang_done = ['ar', 'en', 'fr', 'vi']

# get langs for which word vectors are present
w2v_langs = {}
for folder in glob.glob(WORD_EMBED+"/cc.*.300.vec.gz"):
  fname = folder.split("/")[-1]
  lang = fname.split('.')[1]
  w2v_langs[lang] = True

# get langs for which word vectors are required
for lang in task_langs:
  if lang not in lang_done and lang not in w2v_langs:
    print(lang)

# gunzip for cc.*.vec.gz
w = open("gunzip_cc.sh", "w")
for lang in task_langs:
  if lang in w2v_langs and lang not in lang_done:
    assert(os.path.exists(WORD_EMBED+"/cc."+lang+".300.vec.gz"))
    w.write("gunzip < "+WORD_EMBED+"/cc."+lang+".300.vec.gz"+" > "+FINA_WORD_EMBED+"/cc."+lang+".300.vec\n")
w.close()






