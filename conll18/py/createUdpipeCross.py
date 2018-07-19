import sys
import os
import glob
import codecs
from shutil import copyfile
from tqdm import tqdm

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_UD_MODEL=os.environ['CL_UD_MODEL']

obj_folder = "/home/ganesh/objects/conll18/june16_splits"

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

def getFileFromFolder(folder, pattern):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if fname.endswith(pattern):
      return file_a
  return None

def getTb2Size():
  tb2size = {}
  with open('../resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size

tb2size = getTb2Size()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)
print('# direct = %d; # crossval = %d; # delex = %d; total = %d;'%(len(tb_direct), len(tb_crossval), len(tb_delex), len(tb_direct)+len(tb_crossval)+len(tb_delex)))

num_fold = 5
print('creating data for treebanks which has to be cross-folded...')
num_treebanks_succ = 0
#w_f = open('../shell/ud_cross.sh', 'w')
w_f = open('../shell/ud_cross_test.sh', 'w')
#os.makedirs(obj_folder+"/fold")
for treebank in tqdm(tb_crossval):
  cur_tb_ud_folder = CL_TB_GOLD + "/UD_" + treebank
  cur_ud_model_file = CL_UD_MODEL + "/" + treebank.lower() + "-ud-2.2-conll18-180430.udpipe"
  assert(os.path.exists(cur_tb_ud_folder)==True)
  assert(os.path.exists(cur_ud_model_file)==True)
  cur_tb_ud_train_file = getFileFromFolder(cur_tb_ud_folder, 'train.conllu')
  assert(os.path.exists(cur_tb_ud_train_file)==True)
  cur_tb_txt_train_file = getFileFromFolder(cur_tb_ud_folder, 'train.txt')
  assert(os.path.exists(cur_tb_txt_train_file)==True)
  
  cur_tb_run = obj_folder+"/fold/"+treebank
  #os.makedirs(cur_tb_run)

  #w_f.write("cat "+cur_tb_txt_train_file+" | udpipe --tokenize --tag --parse "+cur_ud_model_file+" --outfile="+cur_tb_run+"/eval-ud.conllu\n")
  #copyfile(cur_tb_ud_train_file, cur_tb_run+"/eval-gold.conllu")

  pred_folder = "/home/ganesh/objects/finale/upload_system1_fcross_7/"+treebank
  assert(os.path.exists(pred_folder)==True)
  w_f.write("python test.py --pred_folder "+pred_folder+" --system_tb "+cur_tb_run+"/eval-ud.conllu --gold_tb "+cur_tb_run+"/eval-gold.conllu\n")

  num_treebanks_succ = num_treebanks_succ + 1
  #if num_treebanks_succ == 1:
  #  break
w_f.close()


















