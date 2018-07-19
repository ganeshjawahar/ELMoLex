import sys
import glob
import os
import xlwt

CL_TB_GOLD=os.environ['CL_TB_GOLD']
CL_HOME=os.environ['CL_HOME']
OBJ_FOLDER = CL_HOME + "/" + sys.argv[1]
run_name = sys.argv[1]

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

def complete_train_direct(file):
  if os.path.exists(file)==False:
    return 'Not_Started'
  lines = [line.rstrip('\n') for line in open(file)]
  if len(lines) < 2:
    return 'In_Progress'
  if "Time (mins):" in lines[-1]:
    content = lines[-1].split()
    return [content[2], content[4]]
  return 'In_Progress'

def get_score_from_train(file):
  lines = [line.rstrip('\n') for line in open(file)]
  score_line = '9999'
  for line in lines:
    if line.startswith('las:'):
      score_line = line
  score = 100.0*float(score_line.split()[3])
  score = '%.2f'%score
  return score

def complete_train_others(file):
  if os.path.exists(file)==False:
    return 'Not_Started'
  lines = [line.rstrip('\n') for line in open(file)]
  if len(lines) < 2:
    return 'In_Progress'
  if "Time (mins):" in lines[-1]:
    time_content = lines[-1].split()
    score_content = get_score_from_train(file)
    return [time_content[2], time_content[4], score_content]
  return 'In_Progress'

def complete_test(file):
  if os.path.exists(file)==False:
    return 'Not_Started'
  lines = [line.rstrip('\n') for line in open(file)]
  if len(lines) < 4:
    return 'In_Progress'
  if "LAS" in lines[-3] and "MLAS" in lines[-2] and "BLEX" in lines[-1]:
    las_res = lines[-3].split()
    las_res = [las_res[3], las_res[5], las_res[8], las_res[9]]
    mlas_res = lines[-2].split()
    mlas_res = [mlas_res[2], mlas_res[4], mlas_res[7], mlas_res[8]]
    blex_res = lines[-1].split()
    blex_res = [blex_res[2], blex_res[4], blex_res[7], blex_res[8]]
    return [ las_res, mlas_res, blex_res ] 
  return 'In_Progress'

tb_direct, tb_crossval, tb_delex = getTreebankDetails(getTb2Size())

result_info = {}
for tb in glob.glob(CL_TB_GOLD+"/UD_*"):
  tb = tb.split("/")[-1]
  tb = tb[tb.find('_')+1:]
  cur_res_folder = OBJ_FOLDER+"/"+tb
  train_out = getFileFromFolder(cur_res_folder, 'out_train')
  test_out = getFileFromFolder(cur_res_folder, 'out_test')
  if train_out!=None and test_out!=None:
    result_info[tb] = [ complete_train_direct(cur_res_folder+"/out_train"), complete_test(cur_res_folder+"/out_test") ]
  else:
    result_info[tb] = [ complete_train_others(cur_res_folder+"/out_train"), complete_test(cur_res_folder+"/out_test") ]
    
wb = xlwt.Workbook()

# write score against SOTA
scores = ['LAS-SOTA']
headers = ['Treebank', 'SOTA', 'GP_Vanilla']
def readSota():
  tblang2sotalas = {}
  with open('../resources/las_sota.txt', 'r') as f:
    li = 0
    for line in f:
      li = li + 1
      if li<=2:
        continue
      content = line.strip().split()
      lang_code = content[1]
      tb_code = content[2] if content[2].isalpha() else '<none>'
      score = content[2] if tb_code is '<none>' else content[3]
      if lang_code not in tblang2sotalas:
        tblang2sotalas[lang_code] = {}
      tblang2sotalas[lang_code][tb_code] = score
  tb2norm = getTreebank2NormalizedName()
  tb2sotalas = {}
  for tb in tb2norm:
    norm = tb2norm[tb]
    lang_code, tb_code = norm.split('_')
    score = None if lang_code not in tblang2sotalas else tblang2sotalas[lang_code]
    if score:
      score = None if tb_code not in tblang2sotalas[lang_code] else tblang2sotalas[lang_code][tb_code]
      if not score:
        score = tblang2sotalas[lang_code]['<none>'] if '<none>' in tblang2sotalas[lang_code] else None
    tb2sotalas[tb] = score
  return tb2sotalas
tb2sotalas = readSota()

def runsota(is_delta, tb2sotalas):
  sname = '-full' if not is_delta else '-delta'
  for si in range(len(scores)):
    ws = wb.add_sheet(scores[si]+sname)
    for i, header in enumerate(headers):
      ws.write(0, i, header)
    ri = 1
    for tb in sorted(result_info):
      ws.write(ri, 0, tb)
      train_result, test_result = result_info[tb]

      # get our result
      our_res = None
      our_style = None
      sota_score = tb2sotalas[tb] if tb in tb2sotalas else None
      if type(test_result) is str or type(train_result) is str:
        if type(test_result) is not str:
          pred_scores = test_result[si]
          if not is_delta:
            our_res = pred_scores[1]
          else:
            if not sota_score:
              our_res = 'inf'
            else:
              our_res = float(pred_scores[1]) - float(sota_score)
              our_res = '%.2f'%our_res
        else:  
          our_res = None
          if type(train_result) is str or len(train_result)==2:
            our_res = test_result
          else:
            assert(len(train_result)==3)           
            if not is_delta:
              our_res = train_result[-1]  + " (-1,-1,"+train_result[1]+")"
            else:
              if not sota_score:
                our_res = 'inf'
              else:
                our_res = float(train_result[-1]) - float(sota_score)
                our_res = '%.2f'%our_res
      else:
        pred_scores = test_result[si]
        if not is_delta:
          our_res = pred_scores[1]
        else:
          if not sota_score:
            our_res = 'inf'
          else:
            our_res = float(pred_scores[1]) - float(sota_score)
            our_res = '%.2f'%our_res
        if sota_score and float(sota_score) < float(pred_scores[1]):
          udpipe_style = xlwt.XFStyle()
          pattern = xlwt.Pattern()
          pattern.pattern = xlwt.Pattern.SOLID_PATTERN
          pattern.pattern_fore_colour = xlwt.Style.colour_map['green']
          udpipe_style.pattern = pattern
      if our_style!=None:
        ws.write(ri, 2, our_res, our_style)
      else:
        ws.write(ri, 2, our_res)

      # get sota result
      sota_score = 'inf' if not sota_score else sota_score
      ws.write(ri, 1, sota_score)

      ri = ri + 1
runsota(True, tb2sotalas)
runsota(False, tb2sotalas)

# write scores against ud
scores = ['LAS', 'MLAS', 'BLEX']
headers = ['Treebank', 'Udpipe', 'GP_Vanilla']
def runud(is_delta):
  sname = '-full' if not is_delta else '-delta'
  for si in range(len(scores)):
    ws = wb.add_sheet(scores[si]+sname)
    for i, header in enumerate(headers):
      ws.write(0, i, header)
    ri = 1
    for tb in sorted(result_info):
      ws.write(ri, 0, tb)
      train_result, test_result = result_info[tb]

      # get our result
      our_res = None
      udpipe_style = None
      if type(test_result) is str or type(train_result) is str:
        if type(test_result) is not str:
          pred_scores = test_result[si]
          our_res = pred_scores[1] + " (" + pred_scores[2]+"," + pred_scores[3] + ",-1)" if not is_delta else pred_scores[2]
        else:  
          our_res = None
          if type(train_result) is str or len(train_result)==2:
            our_res = test_result
          else:
            assert(len(train_result)==3)
            if not is_delta:
              our_res = train_result[-1] + " (-1,-1,"+train_result[1]+")"
            else:
              our_res = train_result[-1]
      else:
        pred_scores = test_result[si]
        train_time = train_result[1]
        our_res = pred_scores[1] + " (" + pred_scores[2]+"," + pred_scores[3] + "," + train_time + ")"  if not is_delta else pred_scores[2]
        if float(pred_scores[0]) > float(pred_scores[1]):
          udpipe_style = xlwt.XFStyle()
          pattern = xlwt.Pattern()
          pattern.pattern = xlwt.Pattern.SOLID_PATTERN
          pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
          udpipe_style.pattern = pattern
      ws.write(ri, 2, our_res)

      # get udpipe result
      udpipe_res = None
      if type(test_result) is str:
        udpipe_res = test_result
      else:
        pred_scores = test_result[si]
        udpipe_res = pred_scores[0]
      if udpipe_style!=None:
        ws.write(ri, 1, udpipe_res, udpipe_style)
      else:
        ws.write(ri, 1, udpipe_res)

      ri = ri + 1
runud(True)
runud(False)

wb.save('../results/result_'+run_name.replace("/","_")+'.xls')
print('results are saved in: '+os.path.abspath('../results/result_'+run_name.replace("/","_")+'.xls'))




