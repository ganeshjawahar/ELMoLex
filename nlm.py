'''
Neural Language Modeling for ELMo features - ELMoLex
'''

import sys
import os
import math
from tqdm import tqdm

from dat import ioutils, nlm_data
from models.modules.nlm import NeuralLangModel
from misc import nlm_args
args = nlm_args.parse_train_args()

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
torch.manual_seed(args.seed)

args = nlm_args.parse_train_args()

use_gpu = torch.cuda.is_available()
print("GPU found: "+str(use_gpu))

print('storing everything in '+str(args.dest_path))
if not os.path.exists(args.dest_path):
  os.makedirs(args.dest_path)

print('reading pre-trained word embedding...')
word_embed, word_dim = ioutils.load_word_embeddings(args.word_path, args.dry_run, [args.train_path, args.dev_path, args.test_path])

print('creating dictionaries...')
nlm_data.init_seed(args.seed)
word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary = nlm_data.create_dict(args.train_path, args.dev_path, args.test_path, word_embed, args.dry_run)
print('words = %d; chars = %d; pos = %d; xpos = %d;'%(word_dictionary.size(),char_dictionary.size(),pos_dictionary.size(),xpos_dictionary.size()))

data_train = nlm_data.read_data_to_variable(args.train_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, args.bptt, use_gpu=use_gpu, symbolic_root=True, dry_run=args.dry_run)
num_train_data = data_train[0].size(0)
print('# sents in train = %d'%num_train_data)
use_dev = os.path.exists(args.dev_path)
if use_dev:
  data_dev = nlm_data.read_data_to_variable(args.dev_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, args.bptt, use_gpu=use_gpu, volatile=True, symbolic_root=True, dry_run=args.dry_run)
  num_dev_data = data_dev[0].size(0)
  print('# sents in dev = %d'%num_dev_data)
use_test = os.path.exists(args.test_path)
if use_test:
  data_test = nlm_data.read_data_to_variable(args.test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, args.bptt, use_gpu=use_gpu, volatile=True, symbolic_root=True, dry_run=args.dry_run)
  num_test_data = data_test[0].size(0)
  print('# sents in test = %d'%num_test_data)

word_table = ioutils.construct_word_embedding_table(word_dim, word_dictionary, word_embed)

print('defining model...')
window = 3
network = NeuralLangModel(word_dim, word_dictionary.size(), args.char_dim, char_dictionary.size(), args.pos_dim, pos_dictionary.size(), xpos_dictionary.size(), args.num_filters, window, args.hidden_size, args.num_layers, embed_word=word_table, pos=args.pos, char=args.char, init_emb=True, delex=args.delex)
if use_gpu:
  network.cuda()

nlm_args.save_train_args(args.dest_path, args, [word_dim, word_dictionary.size(), char_dictionary.size(), pos_dictionary.size(), window, args.batch_size, xpos_dictionary.size()])

def gen_optim(lr, params):
  params = filter(lambda param: param.requires_grad, params)
  return Adam(params, lr=lr, betas=(0.9, 0.9), weight_decay=args.gamma, eps=1e-4)

def train():
  network.train()
  total_loss = 0.0
  for batch in tqdm(range(1, num_train_batches+1)):
    word, char, pos, targets, xpos = nlm_data.get_batch_variable(data_train, args.batch_size)
    optimizer.zero_grad()
    #network.zero_grad()
    output = network(word, char, pos, xpos)
    loss = criterion(output.view(-1, word_dictionary.size()), targets.view(-1))
    loss.backward()
    clip_grad_norm(network.parameters(), clip)
    optimizer.step()
    #for p in network.parameters():
    #  p.data.add_(-lr, p.grad.data)
    total_loss += loss.item()
    #break
  print('Loss: %.4f' % (total_loss/num_train_batches))

def evaluate(dname, data):
  print('evaluating on %s'%(dname))
  num_data = data[0].size(0)
  network.eval()
  total_loss, num_words = 0., 0
  with torch.no_grad():
    for batch in nlm_data.iterate_batch_variable(data, args.batch_size):
      word, char, pos, targets, xpos = batch
      output = network(word, char, pos, xpos)
      total_loss += len(word) * criterion(output.view(-1, word_dictionary.size()), targets.view(-1)).item()
      num_words += len(word)
      #break
  return total_loss/num_words

print('training...')
num_train_batches = num_train_data // args.batch_size + 1
clip, best_dev_perp, best_test_perp, best_epoch = args.clip, sys.maxsize, sys.maxsize, -1
patient, decay, max_decay, double_schedule_decay, lr, schedule = 0, 0, 9, 5, args.learning_rate, args.schedule
optimizer = gen_optim(lr, network.parameters())
#lr = args.learning_rate
criterion = nn.CrossEntropyLoss()
for epoch in range(1, args.num_epochs+1):
  print('epoch %d ...' % epoch)

  # training part
  train()

  cur_dev_perp = sys.maxsize
  if use_dev:
    # evaluation part
    cur_dev_perp = math.exp(evaluate('dev', data_dev))
    if cur_dev_perp<best_dev_perp:
      best_dev_perp = cur_dev_perp
      best_epoch = epoch
      if use_test:
        best_test_perp = math.exp(evaluate('test', data_test))
      torch.save(network.state_dict(), os.path.join(args.dest_path, 'model_epoch_'+str(best_epoch)+'.pt'))
    else:
      if patient >= schedule:
        network.load_state_dict(torch.load(os.path.join(args.dest_path, 'model_epoch_'+str(best_epoch)+'.pt')))
        lr = lr * args.decay_rate
        optimizer = gen_optim(lr, network.parameters())
        patient = 0
        decay += 1
        if decay%double_schedule_decay == 0:
          schedule *= 2
      else:
        patient+=1
      #lr /= 4.0
  elif use_test:
    best_test_perp = math.exp(evaluate('test', data_test))
  if not use_dev and not use_test:
    torch.save(network.state_dict(), os.path.join(args.dest_path, 'model_epoch_'+str(epoch)+'.pt'))
  #break

  print('perp: %.4f (cur_dev), %.4f (dev), %.4f (test); %d (best epoch)'%(cur_dev_perp, best_dev_perp, best_test_perp, best_epoch))




