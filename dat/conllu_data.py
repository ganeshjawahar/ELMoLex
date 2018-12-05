import sys
import codecs
import os

from .constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS,\
  ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE, CHAR_START_ID, CHAR_START, CHAR_END_ID
from .conllu_reader import CoNLLReader
from .dictionary import Dictionary
import numpy as np
np.random.seed(123)
import torch
torch.manual_seed(123)
from torch.autograd import Variable


def create_dict(dict_path, train_path, dev_path, test_path, word_embed_dict, dry_run, vocab_trim=False, add_start_char=0):
  """
  Given train, dev, test treebanks and a word embedding matrix :
  - basic mode : create key_value instanes for each CHAR, WORD, U|X-POS , Relation with special cases for Roots, Padding and End symbols
  - expanding is done on dev set (we assume that dev set is accessible)
  - if vocab_trim == False : we also perform expansion on test set
  check TODOs
  """
  word_dictionary = Dictionary('word', default_value=True, singleton=True)
  char_dictionary = Dictionary('character', default_value=True)
  pos_dictionary = Dictionary('pos', default_value=True)
  xpos_dictionary = Dictionary('xpos', default_value=True)
  type_dictionary = Dictionary('type', default_value=True)

  char_dictionary.add(PAD_CHAR)
  if add_start_char:
    char_dictionary.add(CHAR_START)
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

  vocab = dict() # what is it for ? TODO : cleaning
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
      pos = tokens[3] #if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
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
      vocab[word] += 1 # TODO : are you sure ? Why not + min_occurence ??

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
            pos = tokens[3] # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
            xpos = tokens[4]
            typ = tokens[7]

            pos_dictionary.add(pos)
            xpos_dictionary.add(xpos)
            type_dictionary.add(typ)

            if word not in vocab_set and (word in word_embed_dict or word.lower() in word_embed_dict):
              vocab_set.add(word)
              vocab_list.append(word)
            li = li + 1
            if dry_run and li==100:
              break
  expand_vocab([dev_path])
  if not vocab_trim:
    expand_vocab([test_path])

  for word in vocab_list:
    word_dictionary.add(word)
    if word in singletons:
      word_dictionary.add_singleton(word_dictionary.get_index(word))

  word_dictionary.save(dict_path)
  char_dictionary.save(dict_path)
  pos_dictionary.save(dict_path)
  xpos_dictionary.save(dict_path)
  type_dictionary.save(dict_path)
  word_dictionary.close()
  char_dictionary.close()
  pos_dictionary.close()
  xpos_dictionary.close()
  type_dictionary.close()

  return word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary


def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,
              max_size=None, normalize_digits=True,
              symbolic_root=False, symbolic_end=False, dry_run=False, verbose=0):
  """
  Given vocabularies , data_file :
  - creates a  list of bucket
  - each bucket is a list of unicode encoded worrds, character, pos tags, relations, ... based on DependancyInstances() and Sentence() objects
  """
  _buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, -1]

  last_bucket_id = len(_buckets) - 1
  data = [[] for _ in _buckets]
  max_char_length = [0 for _ in _buckets]
  if verbose>=1:
    print('Reading data from %s' % source_path)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary, None)
  inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)

  while inst is not None and (not dry_run or counter < 100):
    inst_size = inst.length()
    sent = inst.sentence
    for bucket_id, bucket_size in enumerate(_buckets):
      if inst_size < bucket_size or bucket_id == last_bucket_id:
        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, counter, sent.words, sent.raw_lines, inst.xpos_ids])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length[bucket_id] < max_len:
          max_char_length[bucket_id] = max_len
        if bucket_id == last_bucket_id and _buckets[last_bucket_id]<len(sent.word_ids):
          _buckets[last_bucket_id] = len(sent.word_ids)
        break
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    counter += 1
  reader.close()

  return data, max_char_length, _buckets


def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                          type_dictionary, max_size=None, normalize_digits=True, symbolic_root=False,
                          symbolic_end=False, use_gpu=False, volatile=False, dry_run=False, lattice=None,verbose=0,
                          add_end_char=0, add_start_char=0):
  """
  Given data ovject form read_variable creates array-like  variables for character, word, pos, relation, heads ready to be fed to a network
  """
  data, max_char_length, _buckets = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, verbose=verbose, max_size=max_size, normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, dry_run=dry_run)
  bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

  data_variable = []

  ss = [0] * len(_buckets)
  ss1 = [0] * len(_buckets)
  for bucket_id in range(len(_buckets)):
    bucket_size = bucket_sizes[bucket_id]
    if bucket_size == 0:
      data_variable.append((1, 1))
      continue
    bucket_length = _buckets[bucket_id]
    char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD+add_end_char)
    wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    xpid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

    masks_inputs = np.zeros([bucket_size, bucket_length], dtype=np.float32)
    single_inputs = np.zeros([bucket_size, bucket_length], dtype=np.int64)

    lengths_inputs = np.empty(bucket_size, dtype=np.int64)
    
    order_inputs = np.empty(bucket_size, dtype=np.int64)
    raw_word_inputs, raw_lines = [], []

    for i, inst in enumerate(data[bucket_id]):
      ss[bucket_id]+=1
      ss1[bucket_id]=bucket_length
      wids, cid_seqs, pids, hids, tids, orderid, word_raw, lines, xpids = inst
      inst_size = len(wids)
      lengths_inputs[i] = inst_size
      order_inputs[i] = orderid
      raw_word_inputs.append(word_raw)
      # word ids
      wid_inputs[i, :inst_size] = wids
      wid_inputs[i, inst_size:] = PAD_ID_WORD

      if add_start_char:
        shift = 1
      else:
        shift = 0
      if add_end_char:
        shift_end = 1
      else:
        shift_end = 0
      for c, cids in enumerate(cid_seqs):
        if add_start_char:
          cid_inputs[i, c, 0] = CHAR_START_ID
        cid_inputs[i, c, shift:len(cids)+shift] = cids
        if add_end_char:
          cid_inputs[i, c, len(cids)+shift+shift_end] = CHAR_END_ID

        cid_inputs[i, c, shift+len(cids)+shift_end:] = PAD_ID_CHAR
      #  cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
      cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
      # cid_inputs is batch_size, sent_len padded, word lenths padded
      # --
      # pos ids
      pid_inputs[i, :inst_size] = pids
      pid_inputs[i, inst_size:] = PAD_ID_TAG
      # xpos ids
      xpid_inputs[i, :inst_size] = xpids
      xpid_inputs[i, inst_size:] = PAD_ID_TAG
      # type ids
      tid_inputs[i, :inst_size] = tids
      tid_inputs[i, inst_size:] = PAD_ID_TAG
      # heads
      ONLY_PRED = False
      #print("WARNING : ONLY_PRED set to {} ".format(ONLY_PRED))
      if not ONLY_PRED:
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
      #else:
      #  hid_inputs[i, :0] = None
      #  hid_inputs[i, inst_size:] = None
      # masks
      masks_inputs[i, :inst_size] = 1.0
      for j, wid in enumerate(wids):
          if word_dictionary.is_singleton(wid):
              single_inputs[i, j] = 1
      raw_lines.append(lines)

    words = Variable(torch.from_numpy(wid_inputs), requires_grad=False)
    chars = Variable(torch.from_numpy(cid_inputs), requires_grad=False)
    pos = Variable(torch.from_numpy(pid_inputs), requires_grad=False)
    xpos = Variable(torch.from_numpy(xpid_inputs), requires_grad=False)
    heads = Variable(torch.from_numpy(hid_inputs), requires_grad=False)
    types = Variable(torch.from_numpy(tid_inputs), requires_grad=False)
    masks = Variable(torch.from_numpy(masks_inputs), requires_grad=False)
    single = Variable(torch.from_numpy(single_inputs), requires_grad=False)
    lengths = torch.from_numpy(lengths_inputs)
    if use_gpu:
      words = words.cuda()
      chars = chars.cuda()
      pos = pos.cuda()
      xpos = xpos.cuda()
      heads = heads.cuda()
      types = types.cuda()
      masks = masks.cuda()
      single = single.cuda()
      lengths = lengths.cuda()
    data_variable.append((words, chars, pos, xpos, heads, types, masks, single, lengths, order_inputs, raw_word_inputs, raw_lines))
  return data_variable, bucket_sizes, _buckets


def get_batch_variable(data, batch_size, unk_replace=0., lattice=None):
  """
  Given read_data_to_variable() get a random batch
  """
  data_variable, bucket_sizes, _buckets = data
  total_size = float(sum(bucket_sizes))
  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

  # Choose a bucket according to data distribution. We pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number = np.random.random_sample()
  bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
  bucket_length = _buckets[bucket_id]

  words, chars, pos, xpos, heads, types, masks, single, lengths, order_inputs, _, _ = data_variable[bucket_id]
  bucket_size = bucket_sizes[bucket_id]
  #print("INFO : BUCKET SIZE {}  BATCH SIZE {} (in conllu_data)".format(bucket_size, batch_size))

  batch_size = min(bucket_size, batch_size)
  index = torch.randperm(bucket_size).long()[:batch_size]
  if words.is_cuda:
    index = index.cuda()

  words = words[index]

  # discarding singleton
  if unk_replace:
    ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
    noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
    words = words * (ones - single[index] * noise)

  return words, chars[index], pos[index], xpos[index], heads[index], types[index], masks[index], lengths[index], order_inputs[index]

def iterate_batch_variable(data, batch_size, unk_replace=0., lattice=None):
  """
  Iterate over the dataset based on read_data_to_variable() object (used a evaluation)
  """
  data_variable, bucket_sizes, _buckets = data
  bucket_indices = np.arange(len(_buckets))

  for bucket_id in bucket_indices:
    bucket_size = bucket_sizes[bucket_id]
    bucket_length = _buckets[bucket_id]
    if bucket_size == 0:
      continue

    words, chars, pos, xpos, heads, types, masks, single, lengths, order_ids, raw_word_inputs, raw_lines = data_variable[bucket_id]
    if unk_replace:
      ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
      noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
      words = words * (ones - single * noise)
    
    for start_idx in range(0, bucket_size, batch_size):
      excerpt = slice(start_idx, start_idx + batch_size)
      yield words[excerpt], chars[excerpt], pos[excerpt], xpos[excerpt], heads[excerpt], types[excerpt], masks[excerpt], lengths[excerpt], order_ids[excerpt], raw_word_inputs[excerpt], raw_lines[excerpt]





