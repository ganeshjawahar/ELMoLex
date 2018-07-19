import os
import sys
import glob
import operator
from tqdm import tqdm
import json

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_LEX_LAT=os.environ['CL_LEX_LAT']
CL_HOME=os.environ['CL_HOME']
CL_TB_17="/home/ganesh/objects/conll18/conll17-ud-test-2017-05-09"
TEST_MODEL_DIR=CL_HOME+"/fair_outputs/upload_june30"
TEST_OUTPUT_DIR=CL_HOME+"/testFinale"

run_name = sys.argv[1]
TEST_OUTPUT_DIR+="/"+run_name
os.makedirs(TEST_OUTPUT_DIR)
vocab_expand = int(sys.argv[2])==1
lex_expand = int(sys.argv[3])==1
use_ben = int(sys.argv[4])==1

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

def getFileFromFolder(folder, pattern, start=False):
  for file_a in glob.glob(folder+"/*"):
    fname = file_a.split("/")[-1]
    if start:
      if fname.startswith(pattern):
        return file_a
    elif fname.endswith(pattern):
      return file_a
  return None

def getBenTags():
  ltcodes_bentags = {}
  with open('../resources/ben_tags.txt') as f:
    for line in f:
      items = line.strip().split()
      ben_info = {}
      ltcode, ben_info["path"], ben_info["train_file"], ben_info["dev_file"], ben_info["test_file"] = items
      ltcodes_bentags[ltcode] = ben_info
  return ltcodes_bentags

def getTreebank2NormalizedName():
  tb2ltcodes, ltcodes2tb = {}, {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    #if train_f!=None:
    #  norm = train_f.split("/")[-1].split("-")[0]
    #  tb2ltcodes[tb[tb.find('_')+1:]] = norm
    #  ltcodes2tb[norm] = tb[tb.find('_')+1:]
    #else:
    if train_f==None:
      norm = tb.split("/")[-1].split("-")[-1]
      ltcodes2tb[norm] = tb[tb.find('_')+1:]
      tb2ltcodes[tb[tb.find('_')+1:]] = norm
  return tb2ltcodes, ltcodes2tb

def getWordVectors():
  tb_wordvectors = {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    if train_f!=None:
      lang = train_f.split("/")[-1].split("_")[0]
      tb = tb.split("/")[-1]
      tb_wordvectors[tb[tb.find('_')+1:]] = lang
  return tb_wordvectors

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

def getLexiconStr(tb):
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
  return ','.join(lexs_arr)

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
      if tb2sources[targ_tb][0]=='mixed':
        items = []
        for lex in lex_res:
          items.append(lex)
        tb2sources[targ_tb] = items
  return tb2sources

def getMoldelInfo():
  ltcode2modelinfo = {}
  for tb in glob.glob(TEST_MODEL_DIR+"/*"):
    tb = tb.split("/")[-1]
    model_info = {}
    
    if tb not in tb2ltcodes:
      continue

    ltcode = tb2ltcodes[tb]

    # find pred folder
    model_info['pred_folder'] = TEST_MODEL_DIR + "/" + tb
    assert(os.path.exists(model_info['pred_folder']))

    # find word vecs
    model_info['word_path'] = CL_WORD_VEC + "/cc." +tb_wordvectors[tb]+".300.vec"
    assert(os.path.exists(model_info['word_path']))

    # find lexicons
    lexicon_str = "None"
    if tb not in tb_delex:
      # direct or cross
      lang = tb.split('-')[0]
      if lang in lex_res:
        lexicon_str = getFileFromFolder(CL_LEX_LAT, 'UDLex_'+lang+'-'+lex_res[lang][0], True)
        assert(lexicon_str!=None)
    else:
      # delex
      lexicon_str = getLexiconStr(tb)
    model_info['lexicon'] = lexicon_str

    ltcode2modelinfo[ltcode] = model_info
  return ltcode2modelinfo

tb2ltcodes, ltcodes2tb = getTreebank2NormalizedName()
print(tb2ltcodes)
print(len(tb2ltcodes))
sys.exit(0)
ltcodes_bentags = getBenTags()
tb_wordvectors = getWordVectors()
lex_res = getLexicon()
tb2sources = getDelexLinks()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(getTb2Size())
ltcodes_modelinfo = getMoldelInfo()

f_test = open('../shell/testFinale.sh', 'w')

# read json file
json_file = os.path.join(CL_TB_17, 'metadata.json')
json_cont = json.load(open(json_file))

# run over tbs
discarded_ltcodes = []
for tb_item in json_cont:
  lcode, tcode, ltcode = tb_item['lcode'], tb_item['tcode'], tb_item['ltcode']
  txt_f, ud_f, out_f, gold_f = tb_item['rawfile'], tb_item['psegmorfile'], tb_item['outfile'], tb_item['goldfile']

  if ltcode not in ltcodes_modelinfo:
    # pre-trained model unavailable
    discarded_ltcodes.append(ltcode)
    continue
  train_info = ltcodes_modelinfo[ltcode]

  if use_ben and ltcode not in ltcodes_bentags:
    # ben-tags not available
    discarded_ltcodes.append(ltcode)
    continue

  eval_ud_f = os.path.join(CL_TB_17, ud_f) if not use_ben else os.path.join(ltcodes_bentags[ltcode]["path"], ltcodes_bentags[ltcode]["test_file"])
  eval_gold_f = os.path.join(os.environ['CL_TB_17'], gold_f)
  assert(os.path.exists(eval_ud_f)==True)
  assert(os.path.exists(eval_gold_f)==True)

  cmd = "python test.py --word_path "+train_info['word_path']+" --lexicon "+train_info['lexicon']
  cmd += " --system_tb "+eval_ud_f+" --gold_tb "+eval_gold_f
  cmd += " --pred_folder "+train_info['pred_folder']
  if vocab_expand:
    cmd += " --vocab_expand"
  if lex_expand:
    cmd += " --lex_expand"
  cmd += " > "+TEST_OUTPUT_DIR+"/out_"+ltcode
  
  f_test.write(cmd+'\n')

f_test.close()
print(discarded_ltcodes)


