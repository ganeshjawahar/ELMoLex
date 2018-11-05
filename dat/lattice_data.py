import sys
import codecs

from .constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE, PAD_ID_MORPH, ROOT_ID_MORPH, UNK_ID
from .conllu_reader import CoNLLReader
from .dictionary import Dictionary

import numpy as np
np.random.seed(123)
import torch
torch.manual_seed(123)
from torch.autograd import Variable

def _get_lexicial_feature(lexicon, word, pos):
  raw_token = word
  lower_token = word.lower()
  raw_morph_token = raw_token + '$$$' + pos
  lower_morph_token = lower_token + '$$$' + pos

  if raw_morph_token in lexicon.wordpos2feats:
    return lexicon.wordpos2feats[raw_morph_token]
  if lower_morph_token in lexicon.wordpos2feats:
    return lexicon.wordpos2feats[lower_morph_token]
  if raw_token in lexicon.wordpos2feats:
    return lexicon.wordpos2feats[raw_token]
  if lower_token in lexicon.wordpos2feats:
    return lexicon.wordpos2feats[lower_token]
  return [ [ UNK_ID, UNK_ID ] ]

def get_lexical_features(lexicon, words, poss):
  feats = []
  for i in range(len(words)):
    word, pos = words[i], poss[i]
    if i == 0:
      feats.append([ [ROOT_ID_MORPH, ROOT_ID_MORPH] ])
    else:
      feats.append(_get_lexicial_feature(lexicon, word, pos))
  return feats

def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, dry_run=False, lattice=None):
  _buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, -1]
  last_bucket_id = len(_buckets) - 1
  data = [[] for _ in _buckets]
  max_char_length = [0 for _ in _buckets]
  max_morpho_sent = -1
  print('Reading data from %s' % source_path)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary, None)
  inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
  while inst is not None and (not dry_run or counter < 100):
    inst_size = inst.length()
    sent = inst.sentence
    for bucket_id, bucket_size in enumerate(_buckets):
      if inst_size < bucket_size or bucket_id == last_bucket_id:
        lexicon_feats = get_lexical_features(lattice[0], sent.words, inst.postags)
        max_len = max([len(feat) for feat in lexicon_feats if feat!=None]+[-1])
        max_morpho_sent = max(max_len, max_morpho_sent)
        data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, counter, sent.words, lexicon_feats, sent.raw_lines, inst.xpos_ids])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length[bucket_id] < max_len:
          max_char_length[bucket_id] = max_len
        if bucket_id == last_bucket_id and _buckets[last_bucket_id]<len(sent.word_ids):
          _buckets[last_bucket_id] = len(sent.word_ids)
        break
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    counter += 1
  reader.close()
  return data, max_char_length, _buckets, max_morpho_sent

def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, max_size=None, normalize_digits=True, symbolic_root=False, symbolic_end=False, use_gpu=False, volatile=False, dry_run=False, lattice=None):
  data, max_char_length, _buckets, max_morpho_sent = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, max_size=max_size, normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, dry_run=dry_run, lattice=lattice)
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
    char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
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

    morph_inputs = None
    if lattice:
      if lattice[1]: 
        # multi-hot
        morph_inputs = np.zeros([bucket_size, bucket_length, lattice[0].morph_value_dictionary.size()], dtype=np.float32)
      if lattice[2]!=-1:
        # embedding
        morph_specific = np.full((bucket_size, bucket_length, max_morpho_sent), PAD_ID_MORPH, dtype=np.int64)
        morph_group = np.full((bucket_size, bucket_length, max_morpho_sent), PAD_ID_MORPH, dtype=np.int64)
        morph_offsets = np.full((bucket_size, bucket_length), 1, dtype=np.int64)

    for i, inst in enumerate(data[bucket_id]):
      ss[bucket_id]+=1
      ss1[bucket_id]=bucket_length
      wids, cid_seqs, pids, hids, tids, orderid, word_raw, lexicon_feats, lines, xpids = inst
      inst_size = len(wids)
      lengths_inputs[i] = inst_size
      order_inputs[i] = orderid
      raw_word_inputs.append(word_raw)
      # word ids
      wid_inputs[i, :inst_size] = wids
      wid_inputs[i, inst_size:] = PAD_ID_WORD
      for c, cids in enumerate(cid_seqs):
        cid_inputs[i, c, :len(cids)] = cids
        cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
      cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
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
      if True: 
      # WAD FOUND FALSE when got back !! 
      ## REALY ??
      ## TURN TO TRUE for ablation_study again (end of October )
      # we do that in the ONLY_PRED==True because 
        #print("DEBUG--> is true ")
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
      #else:
        #print("DEBUG --> hids set to False ")
      # masks
      masks_inputs[i, :inst_size] = 1.0
      for j, wid in enumerate(wids):
          if word_dictionary.is_singleton(wid):
              single_inputs[i, j] = 1
      # lexicon feats
      if lattice:
        if lattice[1]:
          for wi, word_feat in enumerate(lexicon_feats):
            if word_feat != None:
              for morpho_feat in word_feat:
                morph_inputs[i, wi, morpho_feat[1]] = 1
        if lattice[2]!=-1:
          for wi, word_feat in enumerate(lexicon_feats):
            if word_feat!=None:
              for mi, morph_feat in enumerate(word_feat):
                morph_specific[i][wi][mi] = morph_feat[1]
                morph_group[i][wi][mi] = morph_feat[0]
              morph_offsets[i][wi] = len(word_feat)
      raw_lines.append(lines)

    words = Variable(torch.from_numpy(wid_inputs), requires_grad=False)
    chars = Variable(torch.from_numpy(cid_inputs), requires_grad=False)
    pos = Variable(torch.from_numpy(pid_inputs), requires_grad=False)
    xpos = Variable(torch.from_numpy(xpid_inputs), requires_grad=False)
    heads = Variable(torch.from_numpy(hid_inputs), requires_grad=False)
    types = Variable(torch.from_numpy(tid_inputs), requires_grad=False)
    masks = Variable(torch.from_numpy(masks_inputs), requires_grad=False)
    single = Variable(torch.from_numpy(single_inputs), requires_grad=False)
    if lattice:
      if lattice[2]!=-1:
        morph_specific = Variable(torch.from_numpy(morph_specific), requires_grad=False)
        morph_group = Variable(torch.from_numpy(morph_group), requires_grad=False)
        morph_offsets = Variable(torch.from_numpy(morph_offsets), requires_grad=False)
      if lattice[1]:
        morph_inputs = Variable(torch.from_numpy(morph_inputs), requires_grad=False)
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
      if lattice:
        if lattice[2]!=-1:
          morph_specific = morph_specific.cuda()
          # morph_group = morph_group.cuda()
          morph_offsets = morph_offsets.cuda()
        if lattice[1]:
          morph_inputs = morph_inputs.cuda()
    morph_out = None
    if lattice:
      if lattice[2]!=-1:
        morph_out = [morph_specific, morph_group, morph_offsets]
      if lattice[1]:
        morph_out = morph_inputs
    data_variable.append((words, chars, pos, xpos, heads, types, masks, single, lengths, order_inputs, raw_word_inputs, morph_out, raw_lines))
  return data_variable, bucket_sizes, _buckets

def get_batch_variable(data, batch_size, unk_replace=0.):
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

  words, chars, pos, xpos, heads, types, masks, single, lengths, _, _, morph_inputs, _ = data_variable[bucket_id]
  bucket_size = bucket_sizes[bucket_id]
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

  minp = None
  if morph_inputs is not None:
    if type(morph_inputs) is list:
      minp = [morph_inputs[0][index], morph_inputs[1][index], morph_inputs[2][index]]
    else:
      minp = morph_inputs[index]
  return words, chars[index], pos[index], xpos[index], heads[index], types[index], masks[index], lengths[index], minp

def iterate_batch_variable(data, batch_size, unk_replace=0.):
  data_variable, bucket_sizes, _buckets = data
  bucket_indices = np.arange(len(_buckets))

  for bucket_id in bucket_indices:
    bucket_size = bucket_sizes[bucket_id]
    bucket_length = _buckets[bucket_id]
    if bucket_size == 0:
      continue

    words, chars, pos, xpos, heads, types, masks, single, lengths, order_ids, raw_word_inputs, morph_inputs, raw_lines = data_variable[bucket_id]
    if unk_replace:
      ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
      noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
      words = words * (ones - single * noise)

    for start_idx in range(0, bucket_size, batch_size):
      excerpt = slice(start_idx, start_idx + batch_size)
      minp = None
      if morph_inputs is not None:
        if type(morph_inputs) is list:
          minp = [morph_inputs[0][excerpt], morph_inputs[1][excerpt], morph_inputs[2][excerpt]]
        else:
          minp = morph_inputs[excerpt]
      yield words[excerpt], chars[excerpt], pos[excerpt], xpos[excerpt], heads[excerpt], types[excerpt], masks[excerpt], lengths[excerpt], order_ids[excerpt], raw_word_inputs[excerpt], minp, raw_lines[excerpt]





