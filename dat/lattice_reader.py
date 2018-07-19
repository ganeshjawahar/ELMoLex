import codecs

from .dictionary import Dictionary
from .constants import PAD_ID_MORPH, ROOT_ID_MORPH
from .ioutils import getWordsToBeLoaded

class Lattice(object):
  def __init__(self, file_path, dict_path, trim=None):
    self.morph_token_dictionary = Dictionary('morph-token', default_value=True)
    self.morph_token_dictionary.add(PAD_ID_MORPH)
    self.morph_token_dictionary.add(ROOT_ID_MORPH)

    self.morph_class_dictionary = Dictionary('morph-class', default_value=True)
    self.morph_class_dictionary.add(PAD_ID_MORPH)
    self.morph_class_dictionary.add(ROOT_ID_MORPH)

    self.morph_value_dictionary = Dictionary('morph-value', default_value=True)
    self.morph_value_dictionary.add(PAD_ID_MORPH)
    self.morph_value_dictionary.add(ROOT_ID_MORPH)

    self.wordpos2feats = {}

    useful_words = None if not trim[0] else getWordsToBeLoaded(trim[1])

    file_paths = file_path.split(",")
    for i, fpath in enumerate(file_paths):
      if i!=0:
        fpath = file_paths[0][0:file_paths[0].rfind('/')]+'/'+fpath
      print('read lexicon: '+fpath)
      with codecs.open(fpath, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
          content = line.strip().split('\t')
          morph_feat = content[6]
          if morph_feat != '_':
            token = content[2]
            if useful_words and token not in useful_words:
              continue
            morph_token = content[2]+'$$$'+content[4]
            self.morph_token_dictionary.add(morph_token)
            word_feats = []
            for feat in morph_feat.split('|'):
              morph_class, morph_val = feat.split('=')
              self.morph_class_dictionary.add(morph_class)
              self.morph_value_dictionary.add(morph_class+'='+morph_val)
              word_feats.append([self.morph_class_dictionary.get_index(morph_class), self.morph_value_dictionary.get_index(morph_class+'='+morph_val)])

            if morph_token not in self.wordpos2feats:
              self.wordpos2feats[morph_token] = []
            self.wordpos2feats[morph_token]+=word_feats

            if token not in self.wordpos2feats:
              self.wordpos2feats[token] = []
            self.wordpos2feats[token]+=word_feats

    print('lexicon: #tokens=%d; #class=%d; #value=%d;'%(self.morph_token_dictionary.size(), self.morph_class_dictionary.size(), self.morph_value_dictionary.size()))

    if trim[0]:
      print('trimming num of features per lexical item...')
      for token in self.wordpos2feats:
        self.wordpos2feats[token] = self.wordpos2feats[token][0:15]

    self.morph_token_dictionary.save(dict_path)
    self.morph_class_dictionary.save(dict_path)
    self.morph_value_dictionary.save(dict_path)
  
  def addFeaturesForOOVWords(self, oov_words, lex_path):
    succ_tokens = {}
    file_paths = lex_path.split(",")
    for i, fpath in enumerate(file_paths):
      if i!=0:
        fpath = file_paths[0][0:file_paths[0].rfind('/')]+'/'+fpath
      print('read lexicon: '+fpath)
      with codecs.open(fpath, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
          content = line.strip().split('\t')
          morph_feat = content[6]
          if morph_feat != '_':
            token = content[2]
            if token not in oov_words:
              continue
            succ_tokens[token] = True
            morph_token = content[2]+'$$$'+content[4]
            self.morph_token_dictionary.add(morph_token)
            word_feats = []
            for feat in morph_feat.split('|'):
              morph_class, morph_val = feat.split('=')
              mclass_idx = self.morph_class_dictionary.get_index(morph_class)
              mval_idx = self.morph_value_dictionary.get_index(morph_class+'='+morph_val)
              if mclass_idx!=0 and  mval_idx!=0:
                word_feats.append([mclass_idx, mval_idx])

            if morph_token not in self.wordpos2feats:
              self.wordpos2feats[morph_token] = []
            self.wordpos2feats[morph_token]+=word_feats

            if token not in self.wordpos2feats:
              self.wordpos2feats[token] = []
            self.wordpos2feats[token]+=word_feats
    print('lexicon features for %d/%d oov words fetched'%(len(succ_tokens), len(oov_words)))



