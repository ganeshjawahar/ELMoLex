import sys
import codecs
import os

import numpy as np
import torch
from torch.autograd import Variable

from .constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE
from .conllu_reader import CoNLLReader
from .dictionary import Dictionary

def init_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)

def create_dict(train_path, dev_path, test_path, word_embed_dict, dry_run):
  word_dictionary = Dictionary('word', default_value=True, singleton=True)
  char_dictionary = Dictionary('character', default_value=True)
  pos_dictionary = Dictionary('pos', default_value=True)
  type_dictionary = Dictionary('type', default_value=True)
  xpos_dictionary = Dictionary('xpos', default_value=True)

  char_dictionary.add(PAD_CHAR)
  pos_dictionary.add(PAD_POS)
  xpos_dictionary.add(PAD_POS)
  type_dictionary.add(PAD_TYPE)

  char_dictionary.add(ROOT_CHAR)
  pos_dictionary.add(ROOT_POS)
  xpos_dictionary.add(ROOT_POS)
  type_dictionary.add(ROOT_TYPE)

  char_dictionary.add(END_CHAR)
  pos_dictionary.add(END_POS)
  xpos_dictionary.add(END_POS)
  type_dictionary.add(END_TYPE)

  vocab = dict()
  with codecs.open(train_path, 'r', 'utf-8', errors='ignore') as file:
    li = 0
    for line in file:
      line = line.strip()
      if len(line) == 0 or line[0]=='#':
        continue

      tokens = line.split('\t')
      if '-' in tokens[0] or '.' in tokens[0]:
        continue

      for char in tokens[1]:
        char_dictionary.add(char)

      word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      pos =  tokens[3] if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
      xpos = tokens[4]
      typ = tokens[7]

      pos_dictionary.add(pos)
      xpos_dictionary.add(xpos)
      type_dictionary.add(typ)

      if word in vocab:
        vocab[word] += 1
      else:
        vocab[word] = 1

      li = li + 1
      if dry_run and li == 100:
        break

  # collect singletons
  min_occurence = 1
  singletons = set([word for word, count in vocab.items() if count <= min_occurence])

  # if a singleton is in pretrained embedding dict, set the count to min_occur + c
  for word in vocab.keys():
    if word in word_embed_dict or word.lower() in word_embed_dict:
      vocab[word] += 1

  vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
  vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]

  max_vocabulary_size = 50000
  if len(vocab_list) > max_vocabulary_size:
    vocab_list = vocab_list[:max_vocabulary_size]

  def expand_vocab(data_paths):
    vocab_set = set(vocab_list)
    for data_path in data_paths:
      if os.path.exists(data_path):
        with codecs.open(data_path, 'r', 'utf-8', errors='ignore') as file:
          li = 0
          for line in file:
            line = line.strip()
            if len(line) == 0 or line[0]=='#':
              continue

            tokens = line.split('\t')
            if '-' in tokens[0] or '.' in tokens[0]:
              continue

            for char in tokens[1]:
              char_dictionary.add(char)

            word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
            pos =  tokens[3] if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
            typ = tokens[7]
            xpos = tokens[4]

            pos_dictionary.add(pos)
            type_dictionary.add(typ)
            xpos_dictionary.add(xpos)

            if word not in vocab_set and (word in word_embed_dict or word.lower() in word_embed_dict):
              vocab_set.add(word)
              vocab_list.append(word)
            li = li + 1
            if dry_run and li==100:
              break
  expand_vocab([dev_path, test_path])

  for word in vocab_list:
    word_dictionary.add(word)
    if word in singletons:
      word_dictionary.add_singleton(word_dictionary.get_index(word))

  word_dictionary.close()
  char_dictionary.close()
  pos_dictionary.close()
  xpos_dictionary.close()
  type_dictionary.close()
  return word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary

def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, bptt, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, dry_run=False):
  max_char_length = 0
  print('Reading data from %s' % source_path)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary, None)
  inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
  data = []
  while inst is not None and (not dry_run or counter < 100):
    inst_size = inst.length()
    sent = inst.sentence
    if len(sent.words) > bptt:
      # generate seqeuences
      num_sequences = len(sent.words) - bptt
      for seq_no in range(num_sequences):
        word_ids, char_id_seqs, pos_ids, xpos_ids, tar_ids = [], [], [], [], []
        for i in range(bptt):
          word_ids.append(sent.word_ids[seq_no+i])
          tar_ids.append(sent.word_ids[seq_no+i+1])
          char_id_seqs.append(sent.char_id_seqs[seq_no+i])
          pos_ids.append(inst.pos_ids[seq_no+i])
          xpos_ids.append(inst.xpos_ids[seq_no+i])
        data.append([word_ids, char_id_seqs, pos_ids, tar_ids, xpos_ids])
      max_len = max([len(char_seq) for char_seq in sent.char_seqs])
      max_char_length = max(max_len, max_char_length)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    counter += 1
  reader.close()
  return data, max_char_length

def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, bptt, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_gpu=False, volatile=False, dry_run=False):
  data, max_char_length = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, bptt, max_size=max_size, normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, dry_run=dry_run)

  wid_inputs = np.empty([len(data), bptt], dtype=np.int64)
  cid_inputs = np.empty([len(data), bptt, max_char_length], dtype=np.int64)
  pid_inputs = np.empty([len(data), bptt], dtype=np.int64)
  xpid_inputs = np.empty([len(data), bptt], dtype=np.int64)
  wid_outputs = np.empty([len(data), bptt], dtype=np.int64)

  for di in range(len(data)):
    word_ids, char_id_seqs, pos_ids, tar_wid, xpos_ids = data[di]

    wid_inputs[di, :] = word_ids
    for c, cids in enumerate(char_id_seqs):
      cid_inputs[di, c, :len(cids)] = cids
      cid_inputs[di, c, len(cids):] = PAD_ID_CHAR
    pid_inputs[di, :] = pos_ids
    xpid_inputs[di, :] = xpos_ids
    wid_outputs[di, :] = tar_wid

  words = Variable(torch.from_numpy(wid_inputs), requires_grad=False)
  chars = Variable(torch.from_numpy(cid_inputs), requires_grad=False)
  poss = Variable(torch.from_numpy(pid_inputs), requires_grad=False)
  xposs = Variable(torch.from_numpy(xpid_inputs), requires_grad=False)
  targets = Variable(torch.from_numpy(wid_outputs), requires_grad=False)
  if use_gpu:
    words = words.cuda()
    chars = chars.cuda()
    poss = poss.cuda()
    targets = targets.cuda()
    xposs = xposs.cuda()

  return words, chars, poss, targets, xposs

def get_batch_variable(data, batch_size):
  words, chars, poss, targets, xposs = data
  index = torch.randperm(words.size(0)).long()[:batch_size]
  if words.is_cuda:
    index = index.cuda()
  return words[index], chars[index], poss[index], targets[index], xposs[index]

def iterate_batch_variable(data, batch_size):
  words, chars, poss, targets, xposs = data
  index = torch.arange(0, words.size(0), dtype=torch.long)
  if words.is_cuda:
    index = index.cuda()
  num_batches = words.size(0) // batch_size
  for bi in range(num_batches):
    idx = index[bi * batch_size: (bi+1)*batch_size]
    yield words[idx], chars[idx], poss[idx], targets[idx], xposs[idx]

