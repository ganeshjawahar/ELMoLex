import os
import sys
import glob
from tqdm import tqdm

from misc.conll18_ud_eval import load_conllu_file, evaluate

def getFileFromFolder(folder, pattern):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if fname.endswith(pattern):
      return file_a
  return None

obj_folder = "/home/ganesh/objects/conll18/june16_splits"

w = open("tmp/results/udpipe-results.csv", "w")
header = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]
w.write("treebank")
for head in header:
  w.write(","+head)
w.write("\n")

num = 0
for tb in tqdm(glob.glob(obj_folder+"/direct/*")):
  ud_file = getFileFromFolder(tb, 'eval-ud.conllu')
  gold_file = getFileFromFolder(tb, 'eval-gold.conllu')
  assert(ud_file!=None)
  assert(gold_file!=None)
  tb = tb.split("/")[-1]
  ud_out = load_conllu_file(ud_file)
  gold_out = load_conllu_file(gold_file)
  ud_score = evaluate(gold_out, ud_out)
  w.write(tb)
  for head in header:
    w.write(",{:.2f}".format(100*ud_score[head].f1))
  w.write("\n")
  num = num + 1
w.close()


