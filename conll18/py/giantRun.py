import os
import sys
import glob
import operator

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_RUN_SPLITS=os.environ['CL_RUN_SPLITS']

obj_folder = CL_RUN_SPLITS # "/home/ganesh/objects/conll18/june16_splits"

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

def getTreebank2WordEmbedMapping():
  tb2wv = {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    if train_f!=None:
      lang = train_f.split("/")[-1].split("_")[0]
      tb = tb.split("/")[-1]
      tb2wv[tb[tb.find('_')+1:]] = lang
  return tb2wv

def getTb2Size():
  tb2size = {}
  with open('data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size

def ordertbs(tb2size):
  cur_tb2size = {}
  bsizes = [8, 16, 32, 64]
  cur_tb2bsize = {}
  for tb in glob.glob(obj_folder+"/direct/*"):
    tb = tb.split("/")[-1]
    cur_tb2size[tb] = tb2size[tb]
    bsize = None
    tb2size[tb] = int(tb2size[tb])
    if tb2size[tb] > 20000:
      bsize = 64
    elif tb2size[tb] > 10000:
      bsize = 32
    elif tb2size[tb] > 5000:
      bsize = 16
    else:
      bsize = 8
    cur_tb2bsize[tb] = bsize
  sorted_x = sorted(cur_tb2size.items(), key=operator.itemgetter(1))
  return sorted_x, cur_tb2bsize

tb2wv = getTreebank2WordEmbedMapping()
tb2size = getTb2Size()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)
print('# direct = %d; # crossval = %d; # delex = %d; total = %d;'%(len(tb_direct), len(tb_crossval), len(tb_delex), len(tb_direct)+len(tb_crossval)+len(tb_delex)))

'''
otb, o2bsize = ordertbs(readDatasize())
dest_folder = "/home/ganesh/objects/finale/june16/direct"
os.makedirs(dest_folder)
w_train = open("giant_train.sh", "w")
w_test = open("giant_test.sh", "w")
num = 0
for tb in otb:
  tb = obj_folder+"/direct/" + tb[0]
  model_train_file = getFileFromFolder(tb, 'model-train.conllu')
  model_dev_file = getFileFromFolder(tb, 'model-dev.conllu')
  model_test_file = getFileFromFolder(tb, 'model-test.conllu')
  eval_ud_file = getFileFromFolder(tb, 'eval-ud.conllu')
  eval_gold_file = getFileFromFolder(tb, 'eval-gold.conllu')
  tb = tb.split("/")[-1]
  w2v_file = CL_WORD_VEC + "/cc."+tb2wv[tb]+".300.vec"
  assert(os.path.exists(model_train_file)==True)
  assert(os.path.exists(model_dev_file)==True)
  assert(os.path.exists(model_test_file)==True)
  assert(os.path.exists(eval_ud_file)==True)
  assert(os.path.exists(eval_gold_file)==True)
  assert(os.path.exists(w2v_file)==True)
  dest_path = dest_folder + "/" + tb
  os.makedirs(dest_path)
  gpu_id = num%2
  w_train.write("CUDA_VISIBLE_DEVICES="+str(gpu_id)+" python train.py --lexicon None --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path "+model_test_file+" --batch_size "+str(o2bsize[tb.split("/")[-1]])+" > "+dest_path+"/out_train\n") 
  w_test.write("python test.py --pred_folder "+dest_path+" --system_tb "+eval_ud_file+" --gold_tb "+eval_gold_file+" > "+dest_path+"/out_test\n") 
  num+=1
w_train.close()
w_test.close()
'''

'''
#dest_folder = "/home/ganesh/objects/finale/june16/fold"
dest_folder = "/scratch/gjawahar/projects/objects/finale/june16/fold"
os.makedirs(dest_folder)
w_train = open("giant_train_fold1.sh", "w")
num = 0
for tb in tb_crossval:
  dest_path = dest_folder + "/" + tb
  os.makedirs(dest_path)
  w2v_file = CL_WORD_VEC + "/cc."+tb2wv[tb]+".300.vec"
  for i in range(5):
    cur_fold_tb = obj_folder+ "/fold/" + tb + "/fold" + str(i)
    model_train_file = getFileFromFolder(cur_fold_tb, 'model-train.conllu')
    model_dev_file = getFileFromFolder(cur_fold_tb, 'model-dev.conllu')
    assert(os.path.exists(model_train_file)==True)
    assert(os.path.exists(model_dev_file)==True)
    cur_fold_dest = dest_path + '/fold' + str(i)
    os.makedirs(cur_fold_dest)
    w_train.write("python train.py --lexicon None --dest_path "+cur_fold_dest+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path None --batch_size 8 > "+cur_fold_dest+"/out_train\n")
  num+=1
w_train.close()
'''

dest_folder = "/home/ganesh/objects/finale/june16/delex"
os.makedirs(dest_folder)
w_train_1 = open("giant_train_delex.sh", "w")
w_train_2 = open("giant_train_delex_2.sh", "w")
num = 0
for tb in tb_delex:
  tb = obj_folder+"/delex/" + tb
  model_train_file = getFileFromFolder(tb, 'model-train.conllu')
  model_dev_file = getFileFromFolder(tb, 'model-dev.conllu')
  model_test_file = getFileFromFolder(tb, 'model-test.conllu')
  if model_train_file!=None:
    tb = tb.split("/")[-1]
    w2v_file = CL_WORD_VEC + "/cc.en.300.vec"
    assert(os.path.exists(model_train_file)==True)
    assert(os.path.exists(model_dev_file)==True)
    assert(os.path.exists(model_test_file)==True)
    assert(os.path.exists(w2v_file)==True)
    dest_path = dest_folder + "/" + tb
    os.makedirs(dest_path)
    if num%2==0:
      w_train_1.write("CUDA_VISIBLE_DEVICES="+str(num%2)+" python train.py --delex --lexicon None --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path "+model_test_file+" --batch_size 16 > "+dest_path+"/out_train\n") 
    else:
      w_train_2.write("CUDA_VISIBLE_DEVICES="+str(num%2)+" python train.py --delex --lexicon None --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path "+model_test_file+" --batch_size 16 > "+dest_path+"/out_train\n") 
  num+=1
w_train_1.close()
w_train_2.close()









