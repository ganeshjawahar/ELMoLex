import os
import glob
import json

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_TB_UD=os.environ['CL_TB_UD']
CL_WORD_VEC=os.environ['CL_WORD_VEC']
CL_LEX_LAT=os.environ['CL_LEX_LAT']

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

tb2ltcodes, ltcodes2tb = getTreebank2NormalizedName()

'''
with open('../resources/delex.tsv') as f:
  for line in f:
    content = line.strip().split()
    target, source = content
    res = tb2ltcodes[target[3:]].replace("_","\_")+" & "
    st = ''
    for src in source.split(','):
      if src[3:] not in tb2ltcodes:
        st+= src+","
      else:
        st+= tb2ltcodes[src[3:]].replace("_","\_")+", "
    st = st[0:-2]
    res += st + " \\\\ \hline"
    print(res)
'''

las_scores, mlas_scores, blex_scores = [], [], []
def fill_scores(fname):
  scores = []
  with open('../results/finale_'+fname) as f:
    for line in f:
      content = line.strip()
      if "ParisNLP (Paris)" in content:
        scores.append([content.split()[0][0:-1],content.split()[-1]])
  return scores
las_scores = fill_scores('las')
mlas_scores = fill_scores('mlas')
blex_scores = fill_scores('blex')

las_comb, las_rem = las_scores[0:5], las_scores[5:]
mlas_comb, mlas_rem = mlas_scores[0:5], mlas_scores[5:]
blex_comb, blex_rem = blex_scores[0:5], blex_scores[5:]

table = []

'''
i = 0
for tb in sorted(ltcodes2tb):
  row = []
  row.append(tb)
  row.append(las_rem[i])
  row.append(mlas_rem[i])
  row.append(blex_rem[i])
  i+=1
  table.append(row)

for i in range(41):
  res = table[i][0].replace("_","\_")+' & '
  res += table[i][1][1]+" ("+table[i][1][0]+") & "
  res += table[i][2][1]+" ("+table[i][2][0]+") & "
  res += table[i][3][1]+" ("+table[i][3][0]+") & "
  res += table[41+i][0].replace("_","\_")+' & '
  res += table[41+i][1][1]+" ("+table[41+i][1][0]+") & "
  res += table[41+i][2][1]+" ("+table[41+i][2][0]+") & "
  res += table[41+i][3][1]+" ("+table[41+i][3][0]+") & "
  res = res[0:-2]
  res += " \\\\"
  print(res)
'''

'''
for i in range(len(las_comb)):
  row = []
  row.append(las_comb[i])
  row.append(mlas_comb[i])
  row.append(blex_comb[i])
  table.append(row)

for i in range(len(table)):
  res = table[i][0][1]+" ("+table[i][0][0]+") & "
  res += table[i][1][1]+" ("+table[i][1][0]+") & "
  res += table[i][2][1]+" ("+table[i][2][0]+") & "
  res = res[0:-2]
  res += " \\\\"
  print(res)
'''


import xlwt
from xlrd import open_workbook

res_path = '../resources/result_master.xls'

rb = open_workbook(res_path, formatting_info=True)
r_sheet = rb.sheet_by_index(5)

def is_valid_cell(cell_str):
  return '0'<=cell_str[0] and cell_str[0]<='9'

for ri in range(1, 83):
  all_valid = True
  scores = []
  for ci in [1, 2, 3, 4, 5, 10]:
    if not is_valid_cell(r_sheet.cell(ri, ci).value):
      all_valid = False
    scores.append([r_sheet.cell(ri, ci).value.split()[0], r_sheet.cell(ri, ci).value.split(',')[-1][0:-1]])
  scores = scores[1:]
  if all_valid:
    res = tb2ltcodes[r_sheet.cell(ri, 0).value].replace("_","\_")
    for score in scores:
      res += ' & '+score[0]+' ('+score[1].split('.')[0]+') '
    res += ' \\\\ \hline'
    print(res)





