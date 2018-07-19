import os
import sys
import glob

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_RUN_SPLITS=os.environ['CL_RUN_SPLITS']
CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_HOME=os.environ['CL_HOME']

def checkIsAllIsWell():
  dest_folder = CL_HOME + "/" + "handle_unicode"
  with open('../resources/encoding_issues.txt', 'r') as f:
    for line in f:
      tb = line.strip()
      cur_tb_dest = dest_folder+'/'+tb
      if os.path.exists(cur_tb_dest+"/out_train"):
        lines = [line.rstrip('\n') for line in open(cur_tb_dest+"/out_train")]
        if len(lines)<2 or "Time (mins):" not in lines[-1]:
          print(tb)
      if os.path.exists(cur_tb_dest+"/out_test"):
        lines = [line.rstrip('\n') for line in open(cur_tb_dest+"/out_test")]
        if len(lines)<2 or "BLEX" not in lines[-1]:
          print(tb)
checkIsAllIsWell()
sys.exit(0)

gpus = sys.argv[1].split(",")
gpu_ids = [int(gpu) for gpu in gpus]
num_gpus = len(gpu_ids)
obj_folder = CL_RUN_SPLITS
run_name = sys.argv[2]
dest_folder = CL_HOME+"/"+run_name

def getFileFromFolder(folder, pattern):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if fname.endswith(pattern):
      return file_a
  return 'None'

def getConlluSize(file):
  num_spaces=0
  with open(file, 'r') as f:
    for line in f:
      content = line.strip()
      if len(content)==0:
        num_spaces+=1
  return num_spaces

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

def getTb2Size():
  tb2size = {}
  with open('../resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size
def getTreebankDetails():
  tb2size = getTb2Size()
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

tb2wv = getTreebank2WordEmbedMapping()
tb_direct, tb_crossval, tb_delex = getTreebankDetails()

w=open('../shell/handleUnicode.sh', 'w')
num=0
num_cross, num_dir, num_delex = 0, 0, 0
with open('../resources/encoding_issues.txt', 'r') as f:
  for line in f:
    tb = line.strip()
    tb_type = ''
    if tb in tb_direct:
      tb_type = 'direct'
    elif tb in tb_crossval:
      tb_type = 'fold'
    else:
      tb_type = 'delex'
    cur_tb_dest = dest_folder + "/"+ tb
    os.makedirs(cur_tb_dest)
    gpu_id = str(gpu_ids[num%num_gpus])

    lang_code = tb2wv[tb] if tb not in tb_delex else 'en'
    w2v_file = CL_WORD_VEC + "/cc."+lang_code+".300.vec"
    assert(os.path.exists(w2v_file)==True)
    
    model_train_file = getFileFromFolder(obj_folder + '/'+tb_type+'/' + tb, 'model-train.conllu')
    model_dev_file = getFileFromFolder(obj_folder + '/'+tb_type+'/'+ tb, 'model-dev.conllu')
    model_test_file = getFileFromFolder(obj_folder + '/'+tb_type+'/'+ tb, 'model-test.conllu')
    eval_ud_file = getFileFromFolder(obj_folder + '/'+tb_type+'/'+ tb, 'eval-ud.conllu')
    eval_gold_file = getFileFromFolder(obj_folder + '/'+tb_type+'/'+ tb, 'eval-gold.conllu')

    bsize=getBatchSize(getConlluSize(model_train_file))

    cmd = ''
    cmd+="CUDA_VISIBLE_DEVICES="+gpu_id+" python train.py --lexicon None --dest_path "+cur_tb_dest+" --word_path "+w2v_file+" --train_path "+model_train_file+" --dev_path "+model_dev_file+" --test_path "+model_test_file+" --batch_size "+str(bsize)+" --num_epochs 1 > "+cur_tb_dest+"/out_train\n"
    if eval_ud_file!='None' and eval_gold_file!='None':
      cmd+="python test.py --pred_folder "+cur_tb_dest+" --system_tb "+eval_ud_file+" --gold_tb "+eval_gold_file+" > "+cur_tb_dest+"/out_test\n"
    num+=1
    w.write(cmd)
w.close()

