# creates the finale report 

import sys
import os
import glob
def getTb2Size():
  tb2size = {}
  with open('conll18/resources/data_size.tsv', 'r') as f:
    for line in f:
      content = line.strip().split()
      tb = content[0][content[0].find('_')+1:]
      tb2size[tb] = [int(content[1]), int(content[2])]
  return tb2size
def getTreebankDetails(tb2size):
  CL_TB_GOLD = os.environ['CL_TB_GOLD']
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

src = "/home/ganesh/objects/post_finale/system1"
not_done = []
for source_folder in glob.glob(src+"/*"):
  last_line = None
  with open(source_folder+"/out_train") as f:
    for line in f:
      last_line = line.strip()
  if last_line==None or not last_line.startswith("Time (mins)"):
    not_done.append(source_folder.split("/")[-1])

print(not_done)
print(len(not_done))
'''
tb2size = getTb2Size()
tb_direct, tb_crossval, tb_delex = getTreebankDetails(tb2size)

r = open("/tmp/conll2.sh", 'r')
lmap = {}
for line in r:
  line = line.strip()
  lmap[line.split()[1].split("/")[-1].split("_")[1]]=line
r.close()

w = open('/tmp/conll.sh', 'w')
for done in not_done:
  if done+".sh" in lmap:
    w.write(lmap[done+".sh"]+"\n")
w.close()
'''


