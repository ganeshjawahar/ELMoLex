import sys
import glob
import os
import xlwt
from xlrd import open_workbook

GOOG_CONLL_RES = os.environ['GOOG_CONLL_RES']
NUM_SHEETS = 8
NUM_ROWS = 83
NUM_COLS = 3

res_path = '../results/result_master.xls'

# mention order
col1 = ['serpentes/result_june16_direct.xls', 'ibm/result_othersj19.xls', 'limsi/result_limsi_direct.xls', 'ibm/result_direct_limsi_unicode.xls', 'ibm/result_delex_vanilla.xls']
col2 = ['ibm/result_nlm.xls']
col3 = ['ibm/result_elmo.xls']
col4 = ['ibm/result_lexj19.xls']
col5 = ['serpentes/result_ben_tag_vanilla.xls']
col6 = ['serpentes/result_uxpos_cat.xls']
col7 = ['ibm/result_delex_nlm.xls']
col8 = ['ibm/result_delex_lex.xls']
col9 = ['ibm/result_elmolex.xls']
cols = [col1, col2, col3, col4, col5, col6, col7, col8, col9]
models = ['Vanilla-GP', 'NLM Init', 'ELMO', 'Lexicons', 'NeuralTagger', 'UXPOS-Concat', 'Delex-NLM', 'Delex-LEX', 'Elmo-Lex']

# get sheet names
def getSheetNames():
  rb = open_workbook(GOOG_CONLL_RES+'/ibm/result_nlm.xls', formatting_info=True)
  sheet_names = []
  for si in range(NUM_SHEETS):
    r_sheet = rb.sheet_by_index(si)
    sheet_names.append(r_sheet.name)
  return sheet_names

# extract the results from workbook
def readWorkBook(path):
  sheets = []
  rb = open_workbook(GOOG_CONLL_RES+'/'+path, formatting_info=True)
  for si in range(NUM_SHEETS):
    r_sheet = rb.sheet_by_index(si)
    sheet = []
    for ri in range(NUM_ROWS):
      row = []
      for ci in range(NUM_COLS):
        row.append(r_sheet.cell(ri, ci).value)
      sheet.append(row)
    sheets.append(sheet)
  return sheets

# merge rows in 2 workbook
def is_proper_cell(content):
  start_chars = ['-', '+', '.']
  for i in range(10):
    start_chars.append(str(i))
  return content[0] in start_chars
def mergeRows(wb1, wb2):
  wb3 = []
  for sheet1, sheet2 in zip(wb1, wb2):
    sheet3 = []
    ri = 0
    for row1, row2 in zip(sheet1, sheet2):
      if ri == 0:
        sheet3.append(row1)
      else:
        cell1 = row1[0]
        cell2, cell3 = row1[1], row1[2]
        if not is_proper_cell(cell3) and is_proper_cell(row2[2]):
          cell2, cell3 = row2[1], row2[2]
        sheet3.append([cell1, cell2, cell3])
      ri = ri + 1
    wb3.append(sheet3)
  return wb3

# merge columns in 2 workbook
def mergeColumns(wb1, wb2):
  wb3 = []
  for sheet1, sheet2 in zip(wb1, wb2):
    sheet3 = []
    for row1, row2 in zip(sheet1, sheet2):
      cells = row1 + [row2[2]]
      sheet3.append(cells)
    wb3.append(sheet3)
  return wb3

# create workbook
def createWorkbook(wb, destpath):
  res_master = xlwt.Workbook()
  sheet_names = getSheetNames()
  for sni in range(len(sheet_names)):
    sname = sheet_names[sni]
    ws = res_master.add_sheet(sname)
    assert(len(wb[sni][0])==(2+len(models)))
    for row in range(len(wb[sni])):
      row_info = wb[sni][row]
      if row==0:
        for ci in range(len(row_info)):
          if ci<2:
            ws.write(row, ci, row_info[ci])
          else:
            ws.write(row, ci, models[ci-2])
      else:
        for ci in range(len(row_info)):
          ws.write(row, ci, row_info[ci])
  res_master.save(destpath)

final_results = None
for model in cols:
  model_items = [readWorkBook(model_item) for model_item in model]
  mitem = None
  if len(model_items)>1:
    mitem = model_items[0]
    for mi in range(1, len(model_items)):
      mitem = mergeRows(mitem, model_items[mi])
  else:
    mitem = model_items[0]
  final_results = mitem if not final_results else mergeColumns(final_results, mitem)

createWorkbook(final_results, res_path)
print('results are stored in '+os.path.abspath(res_path))



