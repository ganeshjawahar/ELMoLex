import sys
import os
import glob
import codecs
from shutil import copyfile
from tqdm import tqdm

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_UD_MODEL=os.environ['CL_UD_MODEL']

obj_folder = "/home/ganesh/objects/conll18/finale_splits"

def getDelexLinks():
  tb2sources = {}
  with open('../resources/delex.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      assert(len(content)==2)
      targ_tb = content[0][content[0].find('_')+1:]
      tb2sources[targ_tb] = []
      for item in content[1].split(','):
        source_tb = item if item=='mixed' else item[item.find('_')+1:]
        tb2sources[targ_tb].append(source_tb)
  return tb2sources

def getTb2Size():
  tb2size = {}
  with open('../resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size

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

def read_raw_sentences(train_conll_f):
  raw_sentences, conllu_sentences = [], []
  cur_conllu = ""
  with codecs.open(train_conll_f, mode="r", encoding="utf-8") as f:
    for line in f:
      content = line.strip()

      # add to raw_sentences
      if content.startswith("# text = "):
        sentence = content[len("# text = "):].strip()
        raw_sentences.append(sentence)

      # add to conllu inst.
      if len(content)==0:
        if len(cur_conllu)!=0:
          conllu_sentences.append(cur_conllu.strip())
        cur_conllu=""
      else:
        cur_conllu+=content+"\n"
  if cur_conllu!="":
    conllu_sentences.append(cur_conllu.strip())
  assert(len(raw_sentences)==len(conllu_sentences))
  return raw_sentences, conllu_sentences

def writeSentsJustOne(out_folder, cur_inst, cur_conllu, fname):
  w_f_conll, w_f_txt = codecs.open(out_folder+'/'+fname+'.conllu', 'w', "utf-8"), codecs.open(out_folder+'/'+fname+'.txt', 'w', "utf-8")
  cur_raw_txt = ""
  for si in range(len(cur_conllu)):
    inst = cur_inst[si]
    conllu = cur_conllu[si]
    w_f_conll.write(conllu+"\n\n")
    if si!=0 and "# newdoc" in conllu:
      w_f_txt.write(cur_raw_txt.strip()+"\n\n")
      cur_raw_txt=""
    cur_raw_txt = cur_raw_txt + inst + " "
  if cur_raw_txt!="":
    w_f_txt.write(cur_raw_txt.strip()+"\n\n")
  w_f_conll.close()
  w_f_txt.close()

def writeSents(out_folder, cur_test_inst, cur_test_conllu, cur_train_inst, cur_train_conllu, fname2):
  w_f_conll, w_f_txt = codecs.open(out_folder+'/model-train.conllu', 'w', "utf-8"), codecs.open(out_folder+'/model-train.txt', 'w', "utf-8")
  cur_raw_txt = ""
  for si in range(len(cur_train_conllu)):
    inst = cur_train_inst[si]
    conllu = cur_train_conllu[si]
    w_f_conll.write(conllu+"\n\n")
    if si!=0 and "# newdoc" in conllu:
      w_f_txt.write(cur_raw_txt.strip()+"\n\n")
      cur_raw_txt=""
    cur_raw_txt = cur_raw_txt + inst + " "
  if cur_raw_txt!="":
    w_f_txt.write(cur_raw_txt.strip()+"\n\n")
  w_f_conll.close()
  w_f_txt.close()
  w_f_conll, w_f_txt = codecs.open(out_folder+'/model-'+fname2+'.conllu', 'w', "utf-8"), codecs.open(out_folder+'/model-'+fname2+'.txt', 'w', "utf-8")
  cur_raw_txt = ""
  for si in range(len(cur_test_conllu)):
    inst = cur_test_inst[si]
    conllu = cur_test_conllu[si]
    w_f_conll.write(conllu+"\n\n")
    if si!=0 and "# newdoc" in conllu:
      w_f_txt.write(cur_raw_txt.strip()+"\n\n")
      cur_raw_txt=""
    cur_raw_txt = cur_raw_txt + inst + " "
  if cur_raw_txt!="":
    w_f_txt.write(cur_raw_txt.strip()+"\n\n")
  w_f_conll.close()
  w_f_txt.close()

tb2size = getTb2Size()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)
print('# direct = %d; # crossval = %d; # delex = %d; total = %d;'%(len(tb_direct), len(tb_crossval), len(tb_delex), len(tb_direct)+len(tb_crossval)+len(tb_delex)))

w_f = open('cdat.sh', 'w')
print('creating data for treebanks with train and dev...')
num_treebanks_succ = 0
os.makedirs(obj_folder+"/direct")
for treebank in tqdm(tb_direct):
  cur_tb_gold_folder = CL_TB_GOLD + "/UD_" + treebank
  cur_tb_ud_folder = CL_TB_UD + "/UD_" + treebank
  cur_ud_model_file = CL_UD_MODEL + "/" + treebank.lower() + "-ud-2.2-conll18-180430.udpipe"
  assert(os.path.exists(cur_tb_gold_folder)==True)
  assert(os.path.exists(cur_tb_ud_folder)==True)
  assert(os.path.exists(cur_ud_model_file)==True)
  cur_tb_ud_train_file = getFileFromFolder(cur_tb_ud_folder, 'train.conllu')
  cur_tb_ud_dev_file = getFileFromFolder(cur_tb_ud_folder, 'dev.conllu')
  assert(os.path.exists(cur_tb_ud_train_file)==True)
  assert(os.path.exists(cur_tb_ud_dev_file)==True)
  cur_tb_gold_dev_file = getFileFromFolder(cur_tb_gold_folder, 'dev.conllu')
  cur_tb_gold_dev_txt_file = getFileFromFolder(cur_tb_gold_folder, 'dev.txt')
  assert(os.path.exists(cur_tb_gold_dev_file)==True)
  assert(os.path.exists(cur_tb_gold_dev_txt_file)==True)

  cur_tb_run = obj_folder+"/direct/"+treebank
  os.makedirs(cur_tb_run)
  train_raw_sentences, train_conllu_sentences = read_raw_sentences(cur_tb_ud_train_file)
  cur_dev_inst, cur_dev_conllu, cur_train_inst, cur_train_conllu = [], [], [], []
  num_train = int(len(train_raw_sentences)*0.9)
  for si in range(len(train_raw_sentences)):
    if si < num_train:
      cur_train_inst.append(train_raw_sentences[si])
      cur_train_conllu.append(train_conllu_sentences[si])
    else:
      cur_dev_inst.append(train_raw_sentences[si])
      cur_dev_conllu.append(train_conllu_sentences[si])
  writeSents(cur_tb_run, cur_dev_inst, cur_dev_conllu, cur_train_inst, cur_train_conllu, 'dev')
  copyfile(cur_tb_ud_dev_file, cur_tb_run+"/model-test.conllu")
  w_f.write("cat "+cur_tb_gold_dev_txt_file+" | udpipe --tokenize --tag --parse "+cur_ud_model_file+" --outfile="+cur_tb_run+"/eval-ud.conllu\n")
  copyfile(cur_tb_gold_dev_file, cur_tb_run+"/eval-gold.conllu")

  num_treebanks_succ = num_treebanks_succ + 1
  #if num_treebanks_succ == 2:
  #  break
w_f.close()

'''
num_fold = 5
print('creating data for treebanks which has to be cross-folded...')
num_treebanks_succ = 0
os.makedirs(obj_folder+"/fold")
for treebank in tqdm(tb_crossval):
  cur_tb_ud_folder = CL_TB_UD + "/UD_" + treebank
  cur_ud_model_file = CL_UD_MODEL + "/" + treebank.lower() + "-ud-2.2-conll18-180430.udpipe"
  assert(os.path.exists(cur_tb_ud_folder)==True)
  assert(os.path.exists(cur_ud_model_file)==True)
  cur_tb_ud_train_file = getFileFromFolder(cur_tb_ud_folder, 'train.conllu')
  assert(os.path.exists(cur_tb_ud_train_file)==True)

  cur_tb_run = obj_folder+"/fold/"+treebank
  os.makedirs(cur_tb_run)
  raw_sentences, conllu_sentences = read_raw_sentences(cur_tb_ud_train_file)
  cur_test_inst, cur_test_conllu, cur_train_inst, cur_train_conllu = [], [], [], []
  num_train = int(len(raw_sentences)*0.9)
  for si in range(len(raw_sentences)):
    if si < num_train:
      cur_train_inst.append(raw_sentences[si])
      cur_train_conllu.append(conllu_sentences[si])
    else:
      cur_test_inst.append(raw_sentences[si])
      cur_test_conllu.append(conllu_sentences[si])
  writeSents(cur_tb_run, cur_test_inst, cur_test_conllu, cur_train_inst, cur_train_conllu, 'test')

  subset_size = int(len(cur_train_inst)/num_fold)
  for fi in range(num_fold):
    cur_fold_run = cur_tb_run+"/fold"+str(fi)
    os.makedirs(cur_fold_run)
    cur_fold_dev_inst, cur_fold_dev_conllu, cur_fold_train_inst, cur_fold_train_conllu = [], [], [], []
    i_start = fi*subset_size
    i_end = max((fi+1)*subset_size, len(cur_train_inst))-1 if fi==num_fold-1 else (fi+1)*subset_size-1 
    for si in range(len(cur_train_inst)):
      if i_start<=si and si<=i_end:
        cur_fold_dev_inst.append(cur_train_inst[si])
        cur_fold_dev_conllu.append(cur_train_conllu[si])
      else:
        cur_fold_train_inst.append(cur_train_inst[si])
        cur_fold_train_conllu.append(cur_train_conllu[si])
    writeSents(cur_fold_run, cur_fold_dev_inst, cur_fold_dev_conllu, cur_fold_train_inst, cur_fold_train_conllu, 'dev')
  num_treebanks_succ = num_treebanks_succ + 1
  #if num_treebanks_succ == 1:
  #  break
'''

w_f = open('../shell/cdat_delex.sh', 'w')
print('creating data for treebanks to be run in delexicalized fashion...')
tb2sources = getDelexLinks()
assert(len(tb2sources)==len(tb_delex))

print(tb2sources)
num_treebanks_succ = 0
os.makedirs(obj_folder+"/delex")
proc_mixed, delex2size, K = False, {}, 100
for targ_treebank in tqdm(tb_delex):
  cur_tb_gold_folder = CL_TB_GOLD + "/UD_" + targ_treebank
  assert(os.path.exists(cur_tb_gold_folder)==True)

  sources = tb2sources[targ_treebank]
  if sources[0] == 'mixed' and proc_mixed:
    continue

  if sources[0] == 'mixed':
    sources = tb_direct + tb_crossval
    proc_mixed = True

  cur_tb_run = obj_folder+"/delex/"+targ_treebank
  os.makedirs(cur_tb_run)

  # use train.conllu for evaluation
  gold_train_txt = getFileFromFolder(cur_tb_gold_folder, 'train.txt')
  gold_train_conllu = getFileFromFolder(cur_tb_gold_folder, 'train.conllu')
  if gold_train_txt!=None:
    cur_ud_model_file = CL_UD_MODEL + "/" + targ_treebank.lower() + "-ud-2.2-conll18-180430.udpipe"
    assert(os.path.exists(cur_ud_model_file)==True)
    w_f.write("cat "+gold_train_txt+" | udpipe --tokenize --tag --parse "+cur_ud_model_file+" --outfile="+cur_tb_run+"/eval-ud.conllu\n")
    copyfile(gold_train_conllu, cur_tb_run+"/eval-gold.conllu")

  cur_model_train_conllu, cur_model_dev_conllu, cur_model_test_conllu = [], [], []
  cur_model_train_txt, cur_model_dev_txt, cur_model_test_txt = [], [], []
  for src_treebank in sources:
    # create model files
    if src_treebank in tb_direct or src_treebank in tb_crossval:
      split_folder = 'direct' if src_treebank in tb_direct else 'fold'
      cur_src_tb_splits_folder = obj_folder + '/' + split_folder + '/' + src_treebank
      cur_src_tb_model_train_file = getFileFromFolder(cur_src_tb_splits_folder, 'train.conllu')
      cur_src_tb_model_dev_file = getFileFromFolder(cur_src_tb_splits_folder, 'dev.conllu')
      cur_src_tb_model_test_file = getFileFromFolder(cur_src_tb_splits_folder, 'test.conllu')
      assert(os.path.exists(cur_src_tb_model_train_file)==True)
      assert(os.path.exists(cur_src_tb_model_test_file)==True)
      raw_sentences, conllu_sentences = read_raw_sentences(cur_src_tb_model_train_file)
      cur_model_train_conllu += conllu_sentences[0:K] if sources[0] == 'mixed' else conllu_sentences
      cur_model_train_txt += raw_sentences[0:K] if sources[0] == 'mixed' else raw_sentences
      if cur_src_tb_model_dev_file!=None:
        raw_sentences, conllu_sentences = read_raw_sentences(cur_src_tb_model_dev_file)
        cur_model_dev_conllu += conllu_sentences[0:K//10] if sources[0] == 'mixed' else conllu_sentences
        cur_model_dev_txt += raw_sentences[0:K//10] if sources[0] == 'mixed' else raw_sentences
      raw_sentences, conllu_sentences = read_raw_sentences(cur_src_tb_model_test_file)
      cur_model_test_conllu += conllu_sentences[0:K//10] if sources[0] == 'mixed' else conllu_sentences
      cur_model_test_txt += raw_sentences[0:K//10] if sources[0] == 'mixed' else raw_sentences

  # write model files
  writeSentsJustOne(cur_tb_run, cur_model_train_txt, cur_model_train_conllu, 'model-train')
  writeSentsJustOne(cur_tb_run, cur_model_dev_txt, cur_model_dev_conllu, 'model-dev')
  writeSentsJustOne(cur_tb_run, cur_model_test_txt, cur_model_test_conllu, 'model-test')

  delex2size[targ_treebank] = [len(cur_model_train_conllu), len(cur_model_dev_conllu), len(cur_model_test_conllu)]

  num_treebanks_succ = num_treebanks_succ + 1
  #if num_treebanks_succ == 2:
  #  break
w_f.close()
print(delex2size)








