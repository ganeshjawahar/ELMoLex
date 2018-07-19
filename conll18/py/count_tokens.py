import os
import sys
import glob
from tqdm import tqdm
import operator

lang_max = {}
lang_count = {}

for folder in tqdm(glob.glob('/home/ganesh/data/conll/ud-2.2-conll18-crossfold-morphology-NORMALIZED/*')):
  mx = -1
  count = 0
  for file in glob.glob(folder+'/*'):
    file = file.split('/')[-1]
    if '.conllu' in file:
      cur_count = 0
      seen_root = False
      with open(folder+'/'+file, 'r') as f:
        for line in f:
          line = line.strip()
          if len(line.strip())==0:
            if mx < cur_count:
              mx = cur_count
            if cur_count > 140:
              count+=1
            cur_count = 0
            seen_root = False
            continue
          if line[0] == '#':
            continue
          cur_count+=1
          tokens = line.strip().split('\t')
          if tokens[-3] == 'root':
            assert(seen_root==False)
            seen_root = True
      if mx < cur_count:
        mx = cur_count
        if cur_count > 140:
          count+=1
  lang_max[folder.split('/')[-1]] = mx 
  lang_count[folder.split('/')[-1]] = count 


sorted_x = sorted(lang_max.items(), key=operator.itemgetter(1), reverse=True)
#print(sorted_x)

sorted_x = sorted(lang_count.items(), key=operator.itemgetter(1), reverse=True)
#print(sorted_x)



