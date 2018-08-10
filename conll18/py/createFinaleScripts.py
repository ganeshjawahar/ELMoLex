import os
import sys
import glob
import operator
from tqdm import tqdm
import codecs

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_LEX_LAT=os.environ['CL_LEX_LAT']
CL_HOME=os.environ['CL_HOME']
CL_RUN_SPLITS=CL_HOME+"/data"
dry_run = len(sys.argv)>1
if dry_run:
  print('doing trail run...')

obj_folder = CL_RUN_SPLITS
dest_folder = CL_HOME + '/system1'
main_script_folder = CL_HOME + '/system1_scripts'
side_script_folder = main_script_folder + '/side'
if not os.path.exists(dest_folder):
  os.makedirs(dest_folder)
if not os.path.exists(main_script_folder):
  os.makedirs(main_script_folder)
if not os.path.exists(side_script_folder):
  os.makedirs(side_script_folder)

print('model outputs to be stored at %s'%(dest_folder))
print('training scripts to be stored at %s'%(main_script_folder))

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

def getFileFromFolder(folder, pattern, start=False):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if start:
      if fname.startswith(pattern):
        return file_a
    elif fname.endswith(pattern):
      return file_a
  return None

def getTb2Size():
  tb2size = {}
  with open('conll18/resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size

def getTreebank2WordEmbedMapping():
  tb2wv = {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    if train_f!=None:
      lang = train_f.split("/")[-1].split("_")[0]
      tb = tb.split("/")[-1]
      tb2wv[tb[tb.find('_')+1:]] = lang
  return tb2wv

def getBatchSize(size):
  if size<2500:
    return 8
  if size<5000:
    return 16
  if size<10000:
    return 32
  if size<40000:
    return 64
  if size<200000:
    return 128
  return 256

def getNumEpochsNLM(size):
  if size<2500:
    return 80
  if size<5000:
    return 60
  if size<10000:
    return 40
  if size<40000:
    return 20
  return 15

def getApertium(lis):
  for item in lis:
    if 'aper' in item.lower():
      return [item]
  lis.sort()
  return lis

def getLexicon():
  # get all lexicons
  lex_res = {}
  for lex in glob.glob(CL_LEX_LAT+"/*.conllul"):
    lex = lex.split("/")[-1]
    typ = lex[lex.find('-')+1:].split(".")[0].split("-")[0]
    lex = lex[lex.find('_')+1:lex.find('-')]
    if lex not in lex_res:
      lex_res[lex] = []
    lex_res[lex].append(typ)
  for lex in lex_res:
    lex_res[lex] = getApertium(lex_res[lex])
  return lex_res

def getConlluSize(file):
  num_spaces=0
  with codecs.open(file, 'r', 'utf-8') as f:
    for line in f:
      content = line.strip()
      if len(content)==0:
        num_spaces+=1
  return num_spaces

def getLexiconStr(tb, tb2sources, lex_res):
  sources = tb2sources[tb]
  lexicons = {}
  for source in sources:
    lang = source.split('-')[0]
    if lang not in lex_res:
      continue
    lexicon_file = getFileFromFolder(CL_LEX_LAT, 'UDLex_'+lang+'-'+lex_res[lang][0], True)
    lexicons[lexicon_file] = True
  if len(lexicons)==0:
    return 'None'
  lexs_arr = []
  for i, lex in enumerate(lexicons):
    if i==0:
      lexs_arr.append(lex)
    else:
      lexs_arr.append(lex.split('/')[-1])
  if tb=='Armenian-ArmTDP':
    lexs_arr.append('UDLex_Armenian-Apertium.conllul')
  elif tb=='Faroese-OFT':
    lexs_arr.append('UDLex_Faroese-Apertium.conllul')
  elif tb=='Kurmanji-MG':
    lexs_arr.append('UDLex_Kurmanji-Apertium.conllul')
  return ','.join(lexs_arr)

def getDelexLinks(lex_res):
  tb2sources = {}
  with open('conll18/resources/delex.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      assert(len(content)==2)
      targ_tb = content[0][content[0].find('_')+1:]
      tb2sources[targ_tb] = []
      for item in content[1].split(','):
        source_tb = item if item=='mixed' else item[item.find('_')+1:]
        tb2sources[targ_tb].append(source_tb)
      if tb2sources[targ_tb][0]=='mixed':
        items = []
        for lex in lex_res:
          items.append(lex)
        tb2sources[targ_tb] = items
  return tb2sources

tb2wv = getTreebank2WordEmbedMapping()
tb2size = getTb2Size()
lex_res = getLexicon()
tb2sources = getDelexLinks(lex_res)
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)
print('# direct = %d; # crossval = %d; # delex = %d; total = %d;'%(len(tb_direct), len(tb_crossval), len(tb_delex), len(tb_direct)+len(tb_crossval)+len(tb_delex)))

f_master = open(main_script_folder+"/master_script.sh", "w")

# direct
print('preparing direct scripts...')
num = 0
for tb in tqdm(tb_direct):
  tb = obj_folder+"/direct/" + tb
  model_train_file = getFileFromFolder(tb, 'model-train.conllu')
  model_dev_file = getFileFromFolder(tb, 'model-dev.conllu')  
  tb = tb.split("/")[-1]
  w2v_file = CL_WORD_VEC + "/cc."+tb2wv[tb]+".300.vec"
  assert(os.path.exists(model_train_file)==True)
  assert(os.path.exists(model_dev_file)==True)
  assert(os.path.exists(w2v_file)==True)
  cur_train_size = getConlluSize(model_train_file)

  cur_batch_size, cur_nlm_epochs = str(getBatchSize(cur_train_size)), str(getNumEpochsNLM(cur_train_size))
  dest_path = dest_folder + "/" + tb
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)
  
  cur_parser_epochs = '250'
  if dry_run:
    cur_nlm_epochs = '1'
    cur_parser_epochs = '1'

  side_name = str(num)+"_"+tb
  f_side = open(side_script_folder+"/"+side_name+".sh", 'w')
  lang = tb.split('-')[0]
  cmd1 = "mkdir "+dest_path+"/elmo"
  cmd2 = "python nlm.py --dest_path "+dest_path+"/elmo --word_path "+w2v_file+" --batch_size "+cur_batch_size+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path None --num_epochs "+cur_nlm_epochs+" > "+dest_path+"/out_elmo"
  f_side.write(cmd1+"\n")
  f_side.write(cmd2+"\n")
  cmd3 = None
  if lang in lex_res:
    # lex + elmo
    lexicon_file = getFileFromFolder(CL_LEX_LAT, 'UDLex_'+lang+'-'+lex_res[lang][0], True)
    assert(lexicon_file!=None)
    lex_trim_str = '' if tb not in ['Ancient_Greek-PROIEL', 'Ancient_Greek-Perseus', 'Czech-PDT'] else '--lex_trim ' # TODO: handle lex-trim within elmolex code
    cmd3 = "python train.py --lexicon "+lexicon_file+" --lex_attn Specific --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path None --batch_size "+cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+cur_parser_epochs+" "+lex_trim_str+"> "+dest_path+"/out_train"
  else:
    # just elmo
    cmd3 = "python train.py --lexicon None --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path None --batch_size "+cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+cur_parser_epochs+" > "+dest_path+"/out_train"
  f_side.write(cmd3+"\n")
  #cmd4 = "python ltrans.py --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" > "+dest_path+"/out_ltrans"
  #f_side.write(cmd4+"\n")
  f_side.close()
  f_master.write('bash '+side_script_folder+"/"+side_name+".sh\n")
  num+=1

print('preparing crossval scripts...')
for tb in tqdm(tb_crossval):
  dest_path = dest_folder + "/" + tb
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)
  w2v_file = CL_WORD_VEC + "/cc."+tb2wv[tb]+".300.vec"
  side_name = str(num)+"_"+tb
  f_side = open(side_script_folder+"/"+side_name+".sh", 'w')
  cmds = []

  model_train_file = getFileFromFolder(obj_folder+ "/fold/" + tb, 'model-train.conllu')
  cur_train_size = getConlluSize(model_train_file)
  cur_batch_size, cur_nlm_epochs = str(getBatchSize(cur_train_size)), str(getNumEpochsNLM(cur_train_size))
  lang = tb.split('-')[0]  

  cur_parser_epochs = '250'
  if dry_run:
    cur_nlm_epochs = '1'
    cur_parser_epochs = '1'

  cmd1 = "mkdir "+dest_path+"/elmo"
  cmd2 = "python nlm.py --dest_path "+dest_path+"/elmo --word_path "+w2v_file+" --batch_size "+cur_batch_size+" --train_path "+model_train_file+" --dev_path None --test_path None --num_epochs "+cur_nlm_epochs+" > "+dest_path+"/out_elmo"
  cmds.append(cmd1)
  cmds.append(cmd2)

  for i in range(5):
    cur_fold_tb = obj_folder+ "/fold/" + tb + "/fold" + str(i)
    fold_model_train_file = getFileFromFolder(cur_fold_tb, 'model-train.conllu')
    fold_model_dev_file = getFileFromFolder(cur_fold_tb, 'model-dev.conllu')
    assert(os.path.exists(fold_model_train_file)==True)
    assert(os.path.exists(fold_model_dev_file)==True)
    fold_cur_train_size = getConlluSize(fold_model_train_file)
    fold_cur_batch_size, fold_cur_nlm_epochs = str(getBatchSize(fold_cur_train_size)), str(getNumEpochsNLM(fold_cur_train_size))
    cur_fold_dest = dest_path + '/fold' + str(i)   
    if not os.path.exists(cur_fold_dest):
      os.makedirs(cur_fold_dest)
    cmd = None
    if lang in lex_res:
      # lex + elmo
      lexicon_file = getFileFromFolder(CL_LEX_LAT, 'UDLex_'+lang+'-'+lex_res[lang][0], True)
      assert(lexicon_file!=None)
      cmd = "python train.py --lexicon "+lexicon_file+" --lex_attn Specific --dest_path "+cur_fold_dest+" --word_path "+w2v_file+" --train_path "+fold_model_train_file+" --dev_path "+fold_model_dev_file+" --test_path None --batch_size "+fold_cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+cur_parser_epochs+" > "+cur_fold_dest+"/out_train"
    else:
      # just elmo
      cmd = "python train.py --lexicon None --dest_path "+cur_fold_dest+" --word_path "+w2v_file+" --train_path "+fold_model_train_file+" --dev_path "+fold_model_dev_file+" --test_path None --batch_size "+fold_cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+cur_parser_epochs+" > "+cur_fold_dest+"/out_train"
    cmds.append(cmd)
  cmd="epochs=`python conll18/py/foldgetAvgEpoch.py "+dest_path+"`"
  cmds.append(cmd)
  final_train_epochs = '$epochs'
  if dry_run:
    final_train_epochs = '1'
  cmd = None
  if lang in lex_res:
    # lex + elmo
    lexicon_file = getFileFromFolder(CL_LEX_LAT, 'UDLex_'+lang+'-'+lex_res[lang][0], True)
    assert(lexicon_file!=None)
    cmd = "python train.py --lexicon "+lexicon_file+" --lex_attn Specific --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path None --test_path None --batch_size "+cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+final_train_epochs+" > "+dest_path+"/out_train"
  else:
    cmd = "python train.py --lexicon None --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path None --test_path None --batch_size "+cur_batch_size+" --prelstm_args "+dest_path+"/elmo/args.json --elmo --num_epochs "+final_train_epochs+" > "+dest_path+"/out_train"
  cmds.append(cmd)
  #cmd4 = "python ltrans.py --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" > "+dest_path+"/out_ltrans"
  #cmds.append(cmd4)

  for cmd in cmds:
    f_side.write(cmd+"\n")
  f_side.close()
  f_master.write('bash '+side_script_folder+"/"+side_name+".sh\n")
  num+=1

print('preparing delex scripts...')
for tb in tqdm(tb_delex):
  model_train_file = getFileFromFolder(obj_folder+"/delex/" + tb, 'model-train.conllu')
  if not model_train_file:
    continue
  model_dev_file = getFileFromFolder(obj_folder+"/delex/" + tb, 'model-dev.conllu')
  side_name = str(num)+"_"+tb
  f_side = open(side_script_folder+"/"+side_name+".sh", 'w')
  w2v_file = CL_WORD_VEC + "/cc.en.300.vec"
  assert(os.path.exists(model_train_file)==True)
  assert(os.path.exists(model_dev_file)==True)
  assert(os.path.exists(w2v_file)==True)

  cur_train_size = getConlluSize(model_train_file)
  cur_batch_size, cur_nlm_epochs = str(getBatchSize(cur_train_size)), str(getNumEpochsNLM(cur_train_size))
  lang = tb.split('-')[0]
  dest_path = dest_folder + "/" + tb
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)

  cur_parser_epochs = '250'
  if dry_run:
    cur_parser_epochs = '1'

  lexicon_str = getLexiconStr(tb, tb2sources, lex_res)
  lex_trim_str = '' if tb not in ['Thai-PUD', 'Armenian-ArmTDP'] else '--lex_trim '
  train_cmd="python train.py --lex_attn Specific --delex --lexicon "+lexicon_str+" --dest_path "+dest_path+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path None --prelstm_args args.json --batch_size "+cur_batch_size+" --num_epochs "+cur_parser_epochs+" "+lex_trim_str+"> "+dest_path+"/out_train"
  f_side.write(train_cmd+"\n")
  f_side.close()
  f_master.write('bash '+side_script_folder+"/"+side_name+".sh\n")
  num+=1

f_master.close()























