# creates the finale report 

import sys
import os
import glob

CL_HOME = os.environ['CL_HOME']
obj_folder = CL_HOME + "/testFinale"

las_f = obj_folder + "/finale_las.tsv"
mlas_f = obj_folder + "/finale_mlas.tsv"
blex_f = obj_folder + "/finale_blex.tsv"

ltcode2results = {}
for out_file in glob.glob(obj_folder+"/out_*"):
  fname = out_file.split("/")[-1]
  ltcode = fname[fname.find('_')+1:fname.rfind('_')]
  las_score, mlas_score, blex_score = None, None, None
  with open(out_file, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith("LAS F1"):
        las_score = line
      if line.startswith("MLAS"):
        mlas_score = line
      if line.startswith("BLEX"):
        blex_score = line
  if las_score and mlas_score and blex_score:
    if ltcode not in ltcode2results:
      ltcode2results[ltcode] = {}
    if fname.endswith('udtags'):
      ltcode2results[ltcode]['udpipe'] = {}
      for name, score in [('las', las_score), ('mlas', mlas_score), ('blex', blex_score)]:
        ltcode2results[ltcode]['udpipe'][name] = score.split()[3] if name == 'las' else score.split()[2]
      ltcode2results[ltcode]['elmolex-udtags'] = {}
      for name, score in [('las', las_score), ('mlas', mlas_score), ('blex', blex_score)]:
        ltcode2results[ltcode]['elmolex-udtags'][name] = score.split()[5] if name == 'las' else score.split()[4]
    if fname.endswith('bentags'):
      ltcode2results[ltcode]['elmolex-bentags'] = {}
      for name, score in [('las', las_score), ('mlas', mlas_score), ('blex', blex_score)]:
        ltcode2results[ltcode]['elmolex-bentags'][name] = score.split()[5] if name == 'las' else score.split()[4]

def get_map(mp, fst, scnd):
  if fst in mp:
    if scnd in mp[fst]:
      return mp[fst][scnd]
  return 'None'

las_w = open(las_f, 'w')
mlas_w = open(mlas_f, 'w')
blex_w = open(blex_f, 'w')
header = 'tb\tudpipe\telmolex-udtags\telmolex-bentags\n'
las_w.write(header)
mlas_w.write(header)
blex_w.write(header)
for ltcode in sorted(ltcode2results):
  for name, w in [('las', las_w), ('mlas', mlas_w), ('blex', blex_w)]:
    udpipe = get_map(ltcode2results[ltcode], 'udpipe', name)
    udtags = get_map(ltcode2results[ltcode], 'elmolex-udtags', name)
    bentags = get_map(ltcode2results[ltcode], 'elmolex-bentags', name)
    w.write(ltcode+"\t"+udpipe+"\t"+udtags+"\t"+bentags+"\n")
las_w.close()
mlas_w.close()
blex_w.close()

print('las, mlas and blex results respecitvely can be fetched from the following files:')
print(las_f)
print(mlas_f)
print(blex_f)

