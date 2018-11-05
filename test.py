 # -*- coding: utf-8 -*-

'''
ELMoLex parser - testing
'''

import sys
import os
import glob
import pickle

from misc import args as test_args
from dat.dictionary import Dictionary
from dat import conllu_data, ioutils
from dat.conllu_writer import CoNLLWriter
from dat.constants import NUM_SYMBOLIC_TAGS

import torch

from misc.conll18_ud_eval import load_conllu_file, evaluate

# TODO (very ugly) so meant to be cleaned ! 
ONLY_PRED = True 
print("WARNING : ONLY_PRED IS {}".format(ONLY_PRED))
args = test_args.parse_test_args()

pred_trees_path = os.path.join(args.pred_folder, 'pred_trees')
if not os.path.exists(pred_trees_path):
  os.makedirs(pred_trees_path)

tb_out_path = os.path.join(args.pred_folder, 'pred_trees', args.tb_out) if args.tb_out == 'pred_tree.conllu' else args.tb_out
if os.path.exists(tb_out_path):
  print('%s already exists. So over-writing it.'%(tb_out_path))

use_gpu = False
print('loading dictionaries...')
dict_folder = os.path.join(args.pred_folder, 'dict')
word_dictionary = Dictionary('word', default_value=True, singleton=True)
word_dictionary.load(dict_folder, 'word')
char_dictionary = Dictionary('character', default_value=True)
char_dictionary.load(dict_folder, 'character')
pos_dictionary = Dictionary('pos', default_value=True)
pos_dictionary.load(dict_folder, 'pos')
xpos_dictionary = Dictionary('xpos', default_value=True)
xpos_dictionary.load(dict_folder, 'xpos')
type_dictionary = Dictionary('type', default_value=True)
type_dictionary.load(dict_folder, 'type')
print('words = '+str(word_dictionary.size())+'; chars = '+str(char_dictionary.size())+'; pos = '+str(pos_dictionary.size())+'; xpos = '+str(xpos_dictionary.size())+'; type = '+str(type_dictionary.size())+';')

train_args = test_args.load_train_args(args.pred_folder)
assert(train_args['num_words']==word_dictionary.size())
assert(train_args['num_chars']==char_dictionary.size())
assert(train_args['num_pos']==pos_dictionary.size())
assert(train_args['num_xpos']==xpos_dictionary.size())
assert(train_args['num_types']==type_dictionary.size())

data_reader, lexicon = conllu_data, None
if 'conllul' in train_args['lexicon']:
  from dat import lattice_data
  from dat.lattice_reader import Lattice
  data_reader = lattice_data
  lexicon = pickle.load(open(os.path.join(args.pred_folder, 'dict', 'lexicon.pkl'), 'rb'))
  #TODO : set it to False 
  if not ONLY_PRED:
    if args.lex_expand:
      oov_test_words = ioutils.getOOVWords(word_dictionary, args.gold_tb)
      print('adding lexical info for %d oov words'%(len(oov_test_words)))
      lexicon[0].addFeaturesForOOVWords(oov_test_words, args.lexicon)

oov_embed_dict = None
if not ONLY_PRED:
  if args.vocab_expand:
    # TODO = offset for elmolex_sosweet prediction
    oov_test_words = ioutils.getOOVWords(word_dictionary, args.system_tb)
    print('adding word embedding info for %d oov words'%(len(oov_test_words)))
    oov_embed_dict, embed_dim = ioutils.load_word_embeddings(args.word_path, False, None, oov_test_words)
    for oov_word in oov_embed_dict:
      oov_embed_dict[oov_word] = torch.from_numpy(oov_embed_dict[oov_word])
    assert(train_args['word_dim']==embed_dim)
    print('word embeddings for %d/%d oov words fetched'%(len(oov_embed_dict), len(oov_test_words)))
data_test = data_reader.read_data_to_variable(args.system_tb, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, use_gpu=use_gpu, volatile=True, symbolic_root=True, dry_run=False, lattice=lexicon)
num_test_data = sum(data_test[1])
print('No. of sentences (system_tb) = %d'%(num_test_data))

if 'json' in train_args['prelstm_args']:
  from models.modules.elmo_gp import ElmoGP
  network = ElmoGP(train_args['word_dim'], word_dictionary.size(), train_args['char_dim'], char_dictionary.size(), train_args['pos_dim'], pos_dictionary.size(), xpos_dictionary.size(), train_args['num_filters'], train_args['window'], train_args['hidden_size'], train_args['num_layers'], type_dictionary.size(), train_args['arc_space'], train_args['type_space'], embed_word=None, pos=train_args['use_pos'], char=train_args['use_char'], init_emb=False, prelstm_args=train_args['prelstm_args'], elmo=train_args['elmo'], lattice=lexicon, delex=train_args['delex'])
else:
  from models.modules.parser import BiRecurrentConvBiAffine
  network = BiRecurrentConvBiAffine(train_args['word_dim'], word_dictionary.size(), train_args['char_dim'], char_dictionary.size(), train_args['pos_dim'], pos_dictionary.size(), train_args['num_filters'], train_args['window'], train_args['hidden_size'], train_args['num_layers'], type_dictionary.size(), train_args['arc_space'], train_args['type_space'], embed_word=None, pos=train_args['use_pos'], char=train_args['use_char'])
if use_gpu:
  network.cuda()
model_id = None
if args.epoch !=-1:
  model_id = args.epoch
else:
  model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(args.pred_folder, 'model', '*'))])
  model_id = str(model_id)
model_path = os.path.join(args.pred_folder, 'model', 'model_epoch_'+str(model_id)+'.pt')

network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('loaded model %s successfully'%(model_path))

print('predicting parse tree...')
network.eval()
pred_writer = CoNLLWriter(word_dictionary, char_dictionary, pos_dictionary, type_dictionary)
pred_writer.start(tb_out_path)
g_lcorr, g_total = 0.0, 0
with torch.no_grad():
  for batch in data_reader.iterate_batch_variable(data_test, train_args['batch_size']):
    if lexicon!=None:
      word, char, pos, xpos, heads, types, masks, lengths, order_ids, raw_words, morph, raw_lines = batch
      heads_pred, types_pred = network.decode(word, char, pos, xpos, mask=masks, length=lengths, leading_symbolic=NUM_SYMBOLIC_TAGS, decode=train_args['decode'], input_morph=morph, vocab_expand=[oov_embed_dict, raw_words])
    else:
      word, char, pos, xpos, heads, types, masks, lengths, order_ids, raw_words, raw_lines = batch
      heads_pred, types_pred = network.decode(word, char, pos, xpos, mask=masks, length=lengths, leading_symbolic=NUM_SYMBOLIC_TAGS, decode=train_args['decode'], vocab_expand=[oov_embed_dict, raw_words])
    word = word.data.cpu().numpy()
    pos = pos.data.cpu().numpy()
    lengths = lengths.cpu().numpy()
    heads = heads.data.cpu().numpy()
    types = types.data.cpu().numpy()
    if not ONLY_PRED:
      stats, num_inst = network.compute_las(word, pos, heads_pred, types_pred, heads, types, word_dictionary, pos_dictionary, lengths, punct_set=train_args['punct_set'], symbolic_root=True)
      ucorr, lcorr, total, ucm, lcm = stats
      g_lcorr+=lcorr
      g_total+=total
    pred_writer.store_buffer(word, pos, heads_pred, types_pred, lengths, order_ids, raw_words, raw_lines, symbolic_root=True)
pred_writer.write_buffer()
pred_writer.close()  
if not ONLY_PRED:
  print('test las: %.4f'%(g_lcorr/g_total))
  conllu_18_eval = True
  if args.sosweet is not None: 
    if args.sosweet:
      conllu_18_eval = False
  if conllu_18_eval:
    print('computing CONLL-18 scores...')
    gold_out = load_conllu_file(args.gold_tb)
    our_out = load_conllu_file(tb_out_path)
    ud_out = load_conllu_file(args.system_tb)
    our_score = evaluate(gold_out, our_out)
    ud_score = evaluate(gold_out, ud_out)
    def per_diff(num1, num2):
      if num1==0.0:
        return num1
      return ((num2-num1)/num1)
    print("LAS F1 Score: {:.2f} (ud), {:.2f} (ours), diff {:.2f} {:.2f}%".format(100*ud_score["LAS"].f1, 100*our_score["LAS"].f1, 100*(our_score["LAS"].f1-ud_score["LAS"].f1), 100*per_diff(ud_score["LAS"].f1, our_score["LAS"].f1)))
    print("MLAS Score: {:.2f} (ud), {:.2f} (ours), diff {:.2f} {:.2f}%".format(100*ud_score["MLAS"].f1, 100*our_score["MLAS"].f1, 100*(our_score["MLAS"].f1-ud_score["MLAS"].f1), 100*per_diff(ud_score["MLAS"].f1, our_score["MLAS"].f1)))
    print("BLEX Score: {:.2f} (ud), {:.2f} (ours), diff {:.2f} {:.2f}%".format(100*ud_score["BLEX"].f1, 100*our_score["BLEX"].f1, 100*(our_score["BLEX"].f1-ud_score["BLEX"].f1), 100*per_diff(ud_score["BLEX"].f1, our_score["BLEX"].f1)))

  else:
    print("Adding evaluation eval07")

print("PREDICTION DONE {} pred_folder {} tb_out ".format(args.pred_folder,args.tb_out ))
open("/scratch/bemuller/parsing/sosweet/processing/logs/catching_errors.txt","a").write("PREDICTION DONE {} pred_folder {} tb_out \n ".format(args.pred_folder,args.tb_out))
