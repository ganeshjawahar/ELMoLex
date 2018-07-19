import sys
import glob
import os

CL_TB_GOLD=os.environ['CL_TB_GOLD']
#CL_TB_UD=os.environ['CL_TB_UD']
#CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_UD_MODEL=os.environ['CL_UD_MODEL']
#CL_LEX_LAT=os.environ['CL_LEX_LAT']
#CL_RUN_SPLITS=os.environ['CL_RUN_SPLITS']

# get all gold treebanks with atleast train
gold_tbs = {}
num_td, num_t = 0, 0
for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
  tb = tb.split("/")[-1]
  is_train, is_dev = False, False
  for conllu_f in glob.glob(CL_TB_GOLD+"/"+tb+"/*.conllu"):
    conllu_f = conllu_f.split("/")[-1]
    if 'train' in conllu_f:
      is_train = True
    if 'dev' in conllu_f:
      is_dev = True
  if is_train and is_dev:
    num_td+=1
  elif is_train:
    num_t+=1
  if is_train:
    gold_tbs[tb[tb.find('_')+1:].lower()] = True
print('#td=%d; #t=%d'%(num_td, num_t))

ud_tbs = {}
for tb in glob.glob(CL_UD_MODEL+"/*udpipe"):
  tb = tb.split('/')[-1].split("-ud-2.2-conll18-180430.udpipe")[0]
  ud_tbs[tb.lower()] = True

for tb in gold_tbs:
  if tb not in ud_tbs:
    print(tb)





