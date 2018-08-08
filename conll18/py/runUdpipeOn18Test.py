# run the pre-trained udpipe model on CL_TB_18_RELEASE

import sys
import os
import glob
from tqdm import tqdm

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_UD_MODEL=os.environ['CL_UD_MODEL']
CL_TB_18_RELEASE=os.environ['CL_TB_18_RELEASE']
CL_HOME = os.environ['CL_HOME'] + "/tb18_test_udpipe_preds"
obj_folder = CL_HOME

if not os.path.exists(obj_folder):
  os.makedirs(obj_folder)

def getTreebankDetails(tb2size):
  tb_direct, tb_crossval, tb_delex = [], [], []
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    tb = tb.split("/")[-1]
    is_train, is_dev = False, False
    for conllu_f in glob.glob(CL_TB_GOLD+"/"+tb+"/*.conllu"):
      conllu_f = conllu_f.split("/")[-1]
      if 'train' in conllu_f:
        is_train = True
      if 'dev' in conllu_f:
        is_dev = True
    tb = tb[tb.find('_')+1:]
    if is_train and is_dev:
      tb_direct.append(tb)
    elif is_train:
      if tb2size[tb][0] > 50:
        tb_crossval.append(tb)
      else:
        tb_delex.append(tb)
    else:
      tb_delex.append(tb)
  return tb_direct, tb_crossval, tb_delex

def getTb2Size():
  tb2size = {}
  with open('conll18/resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size

def getFileFromFolder(folder, pattern):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if fname.endswith(pattern):
      return file_a
  return None

tb2size = getTb2Size()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)

w_f = open('conll18/shell/tb18_test_udpipe_preds.sh', 'w')
for tb in tqdm(tb_direct + tb_crossval + tb_delex):
  test_raw_txt = getFileFromFolder(CL_TB_18_RELEASE + "/UD_"+tb, 'test.txt')
  cur_ud_model_file = CL_UD_MODEL + "/" + tb.lower() + "-ud-2.2-conll18-180430.udpipe"
  assert(os.path.exists(test_raw_txt))
  if not os.path.exists(cur_ud_model_file):
    cur_ud_model_file = CL_UD_MODEL + "/mixed-ud-ud-2.2-conll18-180430.udpipe"
  assert(os.path.exists(cur_ud_model_file))
  w_f.write("cat "+test_raw_txt+" | udpipe --tokenize --tag --parse "+cur_ud_model_file+" --outfile="+obj_folder+"/"+test_raw_txt.split("/")[-1].split(".")[0]+"-pred.conllu\n")
w_f.close()

print('run the following:')
print('bash conll18/shell/parallelRun_ud.sh conll18/shell/tb18_test_udpipe_preds.sh 16')












