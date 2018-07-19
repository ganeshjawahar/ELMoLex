import glob
import os
import sys

CL_TB_GOLD=os.environ['CL_TB_GOLD']

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
  tb2norm = {}
  for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
    train_f = getFileFromFolder(tb, 'train.conllu')
    if train_f!=None:
      norm = train_f.split("/")[-1].split("-")[0]
      tb2norm[tb[tb.find('_')+1:]] = norm
  return tb2norm

tb2norm = getTreebank2NormalizedName()

i = 1
for tb in sorted(tb2norm):
  print(str(i)+"\t"+tb+"\t"+tb2norm[tb])
  i = i + 1


