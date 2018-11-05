import gzip
import codecs
from tqdm import tqdm
import os

import numpy as np
np.random.seed(123)
import torch

from .constants import UNK_ID, DIGIT_RE

def construct_word_embedding_table(word_dim, word_dictionary, word_embed, random_init=False):
  scale = np.sqrt(3.0 / word_dim)
  table = np.empty([word_dictionary.size(), word_dim], dtype=np.float32)
  table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
  oov = 0
  for word, index in word_dictionary.items():
    if word in word_embed and not random_init:
      embedding = word_embed[word]
    elif word.lower() in word_embed and not random_init:
      embedding = word_embed[word.lower()]
    else:
      embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
      oov += 1
    table[index, :] = embedding
  print('word OOV: %d/%d' % (oov, word_dictionary.size()))
  return torch.from_numpy(table)

def load_word_embeddings(path, dry_run, content_arr, useful_words=None):
  if not useful_words:
    useful_words = getWordsToBeLoaded(content_arr)
  embed_dim = -1
  embed_dict = dict()
  pbar = None
  with codecs.open(path, 'r', 'utf-8', errors='ignore') as file:
    li = 0
    for line in file:
      line = line.strip()
      if len(line) == 0:
        continue
      tokens = line.split()
      if len(tokens) < 3:
        pbar=tqdm(total=int(tokens[0]) if not dry_run else 100)
        embed_dim = int(tokens[1])
        continue
      ## --
      if len(tokens)-1==embed_dim:
        word = DIGIT_RE.sub(b"0", str.encode(tokens[0])).decode()
        if word in useful_words:
          embed = np.empty([1, embed_dim], dtype=np.float32)
          embed[:] = tokens[1:]
          embed_dict[word] = embed
      ## --- 
      li = li + 1
      if li%5==0:
        pbar.update(5)
      if dry_run and li==100:
        break
  pbar.close()
  return embed_dict, embed_dim

def getWordsToBeLoaded(content_arr):
  words = {}
  for file in content_arr:
    if os.path.exists(file):
      with codecs.open(file, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if len(line) == 0 or line[0]=='#':
            continue
          tokens = line.split('\t')
          if '-' in tokens[0] or '.' in tokens[0]:
            continue
          word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
          if word not in words:
            words[word] = True
            words[word.lower()] = True
  return words

def getOOVWords(word_dictionary, test_path):
  oov_words = {}
  with codecs.open(test_path, 'r', 'utf-8', errors='ignore') as f:
    for line in f:
      line = line.strip()
      if len(line) == 0 or line[0]=='#':
        continue
      tokens = line.split('\t')
      if '-' in tokens[0] or '.' in tokens[0]:
        continue
      word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      if word not in oov_words:
        for cand_word in [word, word.lower()]:
          if word_dictionary.get_index(cand_word)==0:
            oov_words[cand_word] = True
  return oov_words

class Sentence(object):
  def __init__(self, words, word_ids, char_seqs, char_id_seqs, lines):
    self.words = words
    self.word_ids = word_ids
    self.char_seqs = char_seqs
    self.char_id_seqs = char_id_seqs
    self.raw_lines = lines

  def length(self):
    return len(self.words)

class DependencyInstance(object):
  def __init__(self, sentence, postags, pos_ids, xpostags, xpos_ids, lemmas, lemma_ids, heads, types, type_ids):
    self.sentence = sentence
    self.postags = postags
    self.pos_ids = pos_ids
    self.xpostags = xpostags
    self.xpos_ids = xpos_ids
    self.heads = heads
    self.types = types
    self.type_ids = type_ids
    self.lemmas = lemmas
    self.lemma_ids = lemma_ids

  def length(self):
    return self.sentence.length()


  