 # -*- coding: utf-8 -*-

'''
ELMoLex parser - training
'''

import sys
import os
from tqdm import tqdm
import pickle
import time
import numpy as np 
from dat import ioutils, conllu_data
from misc import args as train_args
from dat.constants import NUM_SYMBOLIC_TAGS

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
torch.manual_seed(123)

args = train_args.parse_train_args()
print("ELMO set to ", args.elmo)
print("RANDOM INIT set to ", args.random_init)
print("LEXICON set to ", args.lexicon)
print("POS set to ", args.pos)
print("CHAR set to ", str(args.char))





use_gpu = torch.cuda.is_available()
print("GPU found: "+str(use_gpu))
print('storing everything in '+str(args.dest_path))
dict_path = os.path.join(args.dest_path, 'dict')
models_path = os.path.join(args.dest_path, 'model')
if not os.path.exists(args.dest_path):
  os.makedirs(args.dest_path)
if not os.path.exists(dict_path):
  os.makedirs(dict_path)
if not os.path.exists(models_path):
  os.makedirs(models_path)

print('reading pre-trained word embedding...')
word_embed, word_dim = ioutils.load_word_embeddings(args.word_path, args.dry_run, [args.train_path, args.dev_path, args.test_path])

print('creating dictionaries...')
word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path, args.train_path, args.dev_path, args.test_path, word_embed, args.dry_run, vocab_trim=args.vocab_trim)
print('words = '+str(word_dictionary.size())+'; chars = '+str(char_dictionary.size())+'; pos = '+str(pos_dictionary.size())+'; xpos = '+str(xpos_dictionary.size())+'; type = '+str(type_dictionary.size())+';')

data_reader= conllu_data
lexicon = args.lexicon
if 'conllul' in args.lexicon:
  from dat import lattice_data
  from dat.lattice_reader import Lattice
  data_reader = lattice_data
  lexicon = [Lattice(args.lexicon, dict_path, trim=[args.lex_trim, [args.train_path, args.dev_path, args.test_path]]), args.lex_hot, args.lex_embed, args.lex_attn]
  pickle.dump(lexicon, open(os.path.join(dict_path, 'lexicon.pkl'), 'wb'))

data_train = data_reader.read_data_to_variable(args.train_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, use_gpu=use_gpu, symbolic_root=True, dry_run=args.dry_run, lattice=lexicon)
num_train_data = sum(data_train[1])
print('# sents in train = %d'%num_train_data)

use_dev = os.path.exists(args.dev_path)
if use_dev:
  data_dev = data_reader.read_data_to_variable(args.dev_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, use_gpu=use_gpu, volatile=True, symbolic_root=True, dry_run=args.dry_run, lattice=lexicon)
  num_dev_data = sum(data_dev[1])
  print('# sents in dev = %d'%num_dev_data)
use_test = os.path.exists(args.test_path)
if use_test:
  data_test = data_reader.read_data_to_variable(args.test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, use_gpu=use_gpu, volatile=True, symbolic_root=True, dry_run=args.dry_run, lattice=lexicon)
  num_test_data = sum(data_test[1])
  print('# sents in test = %d'%num_test_data)
punct_set = args.punctuation
print("random init of word vectors set to {} ".format(args.random_init))
word_table = ioutils.construct_word_embedding_table(word_dim, word_dictionary, word_embed, random_init=args.random_init)
print('defining model...')
window = 3
print("Warning : forced to use it ")
from models.modules.elmo_gp import ElmoGP
_lexicon = False if lexicon == "0" else lexicon
print("- elmo gp  ")
#network = ElmoGP(word_dim, word_dictionary.size(), args.char_dim, char_dictionary.size(), args.pos_dim, pos_dictionary.size(), xpos_dictionary.size(), args.num_filters, window, args.hidden_size, args.num_layers, type_dictionary.size(), args.arc_space, args.type_space, embed_word=word_table, pos=args.pos, char=args.char, init_emb=True, prelstm_args=args.prelstm_args, elmo=args.elmo, lattice=_lexicon, delex=args.delex)
network = ElmoGP(word_dim, word_dictionary.size(), args.char_dim, char_dictionary.size(), args.pos_dim, pos_dictionary.size(), xpos_dictionary.size(), args.num_filters, window, args.hidden_size, args.num_layers, type_dictionary.size(), args.arc_space, args.type_space, embed_word=word_table, pos=args.pos, char=args.char, init_emb=True, prelstm_args=args.prelstm_args, elmo=args.elmo, lattice=lexicon, delex=args.delex, word_dictionary=word_dictionary, char_dictionary=char_dictionary, use_gpu=use_gpu)
if use_gpu:
  network.cuda()
#count parameters 
model_parameters = filter(lambda p: p.requires_grad, network.parameters())
params = int(sum([np.prod(p.size()) for p in model_parameters]))
print("Number of trainable parameters {} ".format(params))

train_args.save_train_args(args.dest_path, args, [word_dim, word_dictionary.size(), char_dictionary.size(), pos_dictionary.size(), window, type_dictionary.size(), punct_set, args.decode, args.batch_size, xpos_dictionary.size(), params])

def gen_optim(lr, params):
  params = filter(lambda param: param.requires_grad, params)
  return Adam(params, lr=lr, betas=(0.9, 0.9), weight_decay=args.gamma, eps=1e-4)

def train():
  network.train()
  train_err, train_err_arc, train_err_type, train_total = 0.0, 0.0, 0.0, 0
  for batch in tqdm(range(1, num_train_batches+1)):
    optimizer.zero_grad()
    if lexicon!="0":
      word, char, pos, xpos, heads, types, masks, lengths, morph = data_reader.get_batch_variable(data_train, args.batch_size, unk_replace=args.unk_replace)
      loss_arc, loss_type = network.loss(word, char, pos, xpos, heads, types, mask=masks, length=lengths, input_morph=morph)
    else:
      #print("--> masks " , masks)
      word, char, pos, xpos, heads, types, masks, lengths, order_inputs = data_reader.get_batch_variable(data_train, args.batch_size, unk_replace=args.unk_replace)
      loss_arc, loss_type = network.loss(word, char, pos, xpos, heads=heads, types=types, mask=masks, length=lengths)
    loss = loss_arc + loss_type
    loss.backward()
    clip_grad_norm(network.parameters(), clip)
    optimizer.step()

    num_inst = masks.data.sum() - word.size(0)
    train_err += loss.data.item() * num_inst
    train_err_arc += loss_arc.data.item() * num_inst
    train_err_type += loss_type.data.item() * num_inst
    train_total += num_inst
  print('Loss: %.4f, Arc: %.4f, Type: %.4f' % (train_err/train_total, train_err_arc/train_total, train_err_type/train_total))

def evaluate_parser(dname, data):
  print('evaluating on %s'%(dname))
  num_data = sum(data[1])
  network.eval()
  g_lcorr, g_total = 0.0, 0
  with torch.no_grad():
    for batch in data_reader.iterate_batch_variable(data, args.batch_size):
      if lexicon!="0":
        word, char, pos, xpos, heads, types, masks, lengths, _, _, morph, _ = batch
        heads_pred, types_pred = network.decode(word, char, pos, xpos, mask=masks, length=lengths, leading_symbolic=NUM_SYMBOLIC_TAGS, decode=args.decode, input_morph=morph)
      else:
        word, char, pos, xpos, heads, types, masks, lengths, order_ids, _, _ = batch
        
        heads_pred, types_pred = network.decode(word, char, pos, xpos, mask=masks, length=lengths, leading_symbolic=NUM_SYMBOLIC_TAGS, decode=args.decode)
      word = word.data.cpu().numpy()
      pos = pos.data.cpu().numpy()
      lengths = lengths.cpu().numpy()
      heads = heads.data.cpu().numpy()
      types = types.data.cpu().numpy()
      stats, num_inst = network.compute_las(word, pos, heads_pred, types_pred, heads, types, word_dictionary, pos_dictionary, lengths, punct_set=punct_set, symbolic_root=True)
      ucorr, lcorr, total, ucm, lcm = stats
      g_lcorr+=lcorr
      g_total+=total
  return g_lcorr/g_total

print('training...')
num_train_batches = num_train_data // args.batch_size + 1
clip, best_dev_las, best_test_las, best_epoch = args.clip, -1, -1, -1
patient, decay, max_decay, double_schedule_decay, lr, schedule = 0, 0, 9, 5, args.learning_rate, args.schedule
optimizer = gen_optim(lr, network.parameters())
avg_epoch_time, train_start_time, num_epochs = 0, time.time(), 0
for epoch in range(1, args.num_epochs+1):
  print('epoch %d ...' % epoch)
  num_epochs+=1

  # training part
  cur_train_start = time.time()
  train()
  cur_train_end = time.time()
  avg_epoch_time += cur_train_end-cur_train_start

  if use_dev:
    # evaluation part
    cur_dev_las = evaluate_parser('dev', data_dev)
    if cur_dev_las>best_dev_las:
      best_dev_las = cur_dev_las
      best_epoch = epoch
      if use_test:
        best_test_las = evaluate_parser('test', data_test)
      torch.save(network.state_dict(), os.path.join(models_path, 'model_epoch_'+str(best_epoch)+'.pt'))
    else:
      if cur_dev_las * 100 < (best_dev_las * 100 - 5) or patient >= schedule:
        network.load_state_dict(torch.load(os.path.join(models_path, 'model_epoch_'+str(best_epoch)+'.pt')))
        lr = lr * args.decay_rate
        optimizer = gen_optim(lr, network.parameters())
        patient = 0
        decay += 1
        if decay%double_schedule_decay == 0:
          schedule *= 2
      else:
        patient+=1
    if decay == max_decay:
      print("breaking training because decay {} reached max_decay ".format(decay))
      break
  elif use_test:
    best_test_las = evaluate_parser('test', data_test)
  if not use_dev and not use_test:
    torch.save(network.state_dict(), os.path.join(models_path, 'model_epoch_'+str(best_epoch)+'.pt'))

  print('las: %.4f (dev), %.4f (test); %d (best epoch)'%(best_dev_las, best_test_las, best_epoch))
print('Time (mins): %.2f (avg-epoch), %.2f (total-train)'%((avg_epoch_time/num_epochs)/60.0, (time.time()-train_start_time)/60.0))


