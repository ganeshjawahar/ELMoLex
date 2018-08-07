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
CL_TB_18_RELEASE=os.environ['CL_TB_18_RELEASE']
TEST_MODEL_DIR=CL_HOME+"/system1"
TEST_OUTPUT_DIR=CL_HOME+"/testFinale"

if not os.path.exists(TEST_OUTPUT_DIR):
  os.makedirs(TEST_OUTPUT_DIR)

vocab_expand = True
lex_expand = True
use_ben = True

def getTb2Size():
  tb2size = {}
  with open('conll18/resources/data_size.tsv', 'r') as f:
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
  ben_tags = {}
  tb2ltcodes, ltcodes2tb = getTreebank2NormalizedName()
  with open('conll18/resources/ben_tags.txt', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb_name = ltcodes2tb[content[0]]
      pred_test = content[1]+'/'+content[4]
      assert(os.path.exists(pred_test))
      ben_tags[tb_name] = pred_test
  return ben_tags

def getTreebank2NormalizedName():
  tb2ltcodes, ltcodes2tb = {}, {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    if train_f!=None:
      norm = train_f.split("/")[-1].split("-")[0]
      tb2ltcodes[tb[tb.find('_')+1:]] = norm
      ltcodes2tb[norm] = tb[tb.find('_')+1:]
  
  lcds = ['pcm','en','th','ja','br','fo','fi','sv','cs']
  tcds = ['nsc','pud','pud','modern','keb','oft','pud','pud','pud']
  unnorm = ['Naija-NSC','English-PUD','Thai-PUD','Japanese-Modern','Breton-KEB','Faroese-OFT','Finnish-PUD','Swedish-PUD','Czech-PUD']

  for l, t, u in zip(lcds, tcds, unnorm):
    lt = l+"_"+t
    ltcodes2tb[lt] = u
    tb2ltcodes[u] = lt

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
    model_info['word_path'] = "None"
    if tb in tb_wordvectors:
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
      lexicon_str = getLexiconStr(tb, tb2sources, lex_res)
    model_info['lexicon'] = lexicon_str

    ltcode2modelinfo[ltcode] = model_info
  print('model information recovered for %d treebanks'%(len(ltcode2modelinfo)))
  return ltcode2modelinfo

tb2ltcodes, ltcodes2tb = getTreebank2NormalizedName()
ltcodes_bentags = getBenTags()
tb_wordvectors = getWordVectors()
lex_res = getLexicon()
tb2sources = getDelexLinks(lex_res)
tb_direct, tb_crossval, tb_delex = getTreebankDetails(getTb2Size())
ltcodes_modelinfo = getMoldelInfo()

print('writing test master script at %s'%(CL_HOME + '/system1_scripts/testFinale.sh'))
f_test = open(CL_HOME + '/system1_scripts/testFinale.sh', 'w')

# run over tbs
discarded_ltcodes = []
for tb_item in glob.glob(CL_TB_18_RELEASE+"/*"):
  tb = tb_item.split("/")[-1][3:]
  
  if tb not in tb2ltcodes:
    continue

  ltcode = tb2ltcodes[tb]
  
  if tb not in ltcodes_bentags:
    # pre-trained model unavailable
    discarded_ltcodes.append(ltcode)
    continue

  train_info = ltcodes_modelinfo[ltcode]
  eval_ud_f = ltcodes_bentags[tb]
  eval_gold_f = getFileFromFolder(os.path.join(CL_TB_18_RELEASE, 'UD_'+tb), 'test.conllu')
  assert(eval_gold_f!=None)

  cmd = "python test.py --word_path "+train_info['word_path']+" --lexicon "+train_info['lexicon']
  cmd += " --system_tb "+eval_ud_f+" --gold_tb "+eval_gold_f
  cmd += " --pred_folder "+train_info['pred_folder']
  cmd += " --vocab_expand"
  cmd += " --lex_expand"
  cmd += " > "+TEST_OUTPUT_DIR+"/out_"+ltcode
  
  f_test.write(cmd+'\n')

f_test.close()

print('trained model or testing file not avaliable for following %d treebanks:'%(len(discarded_ltcodes)))
print(discarded_ltcodes)


