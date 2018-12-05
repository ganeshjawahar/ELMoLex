import codecs
import sys
from .ioutils import DependencyInstance, Sentence
from .constants import DIGIT_RE, MAX_CHAR_LENGTH, NUM_CHAR_PAD, ROOT, ROOT_CHAR, ROOT_POS, ROOT_TYPE, PAD


class CoNLLReader(object):

  def __init__(self, file_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary, lemma_dictionary):
    self.__source_file = codecs.open(file_path, 'r', 'utf-8', errors='ignore')
    self.__file_path = file_path
    self.__word_dictionary = word_dictionary
    self.__char_dictionary = char_dictionary
    self.__lemma_dictionary = lemma_dictionary

    self.__pos_dictionary = pos_dictionary
    self.__xpos_dictionary = xpos_dictionary

    self.__type_dictionary = type_dictionary

  def close(self):
    self.__source_file.close()

  def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
    line = self.__source_file.readline()

    # skip multiple blank lines.
    raw_text = []
    while len(line) > 0 and (len(line.strip()) == 0 or line.strip()[0] == '#'):
      if line.strip()[0] == '#':
        raw_text.append(line)
      line = self.__source_file.readline()
    
    if len(line) == 0:
      return None

    lines = []
    while len(line.strip()) > 0:
      line = line.strip()
      lines.append(line.split('\t'))
      line = self.__source_file.readline()

    length = len(lines)
    if length == 0:
      return None

    words = []
    word_ids = []
    char_seqs = []
    char_id_seqs = []
    lemmas = []
    lemma_ids = []
    
    postags = []
    pos_ids = []
    xpostags = []
    xpos_ids = []

    types = []
    type_ids = []
    heads = []

    if symbolic_root:
      words.append(ROOT)
      word_ids.append(self.__word_dictionary.get_index(ROOT))
      char_seqs.append([ROOT_CHAR, ])
      char_id_seqs.append([self.__char_dictionary.get_index(ROOT_CHAR), ])
      lemmas.append(ROOT)
      #lemma_ids.append(self.__lemma_dictionary.get_index(ROOT))

      postags.append(ROOT_POS)
      pos_ids.append(self.__pos_dictionary.get_index(ROOT_POS))
      xpostags.append(ROOT_POS)
      xpos_ids.append(self.__xpos_dictionary.get_index(ROOT_POS))

      types.append(ROOT_TYPE)
      type_ids.append(self.__type_dictionary.get_index(ROOT_TYPE))
      heads.append(0)

    for tokens in lines:
      if '-' in tokens[0] or '.' in tokens[0]:
        continue
      if len(tokens)<10:
        sys.stderr.write("Sentence broken for unkwown reasons \n".format(lines))
        open("/scratch/bemuller/parsing/sosweet/processing/logs/catching_errors.txt","a").write("Line broken {} because of tokens {} from {} file \n ".format(lines, tokens,self.__file_path))        
        continue

      chars = []
      char_ids = [] 
      #sys.stderr.write("DEBUG --> tokens ERROR {} \n".format(tokens))
      #sys.stderr.write("DEBUG --> lines ERROR {} \n".format(lines))
      for char in tokens[1]:
        chars.append(char)
        char_ids.append(self.__char_dictionary.get_index(char))
      if len(chars) > MAX_CHAR_LENGTH:
        chars = chars[:MAX_CHAR_LENGTH]
        char_ids = char_ids[:MAX_CHAR_LENGTH]
      char_seqs.append(chars)
      char_id_seqs.append(char_ids)

      #sys.stderr.write("CHAR FILLED \n")

      words.append(tokens[1])
      lemmas.append(tokens[2])
      #sys.stderr.write("LEMMAS  FILLED \n")
      

      word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      word_ids.append(self.__word_dictionary.get_index(word))
      #lemma_ids.append(self.__lemma_dictionary.get_index(tokens[2]))

      pos = tokens[3] # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
      xpos = tokens[4]
      postags.append(pos)
      xpostags.append(xpos)
      pos_ids.append(self.__pos_dictionary.get_index(pos))
      xpos_ids.append(self.__xpos_dictionary.get_index(xpos))

      head = tokens[6]
      type = tokens[7]
      types.append(type)
      type_ids.append(self.__type_dictionary.get_index(type))
      heads.append(head)

    if symbolic_end:
      words.append(END)
      word_ids.append(self.__word_dictionary.get_index(END))
      char_seqs.append([END_CHAR, ])
      char_id_seqs.append([self.__char_dictionary.get_index(END_CHAR), ])
      postags.append(END_POS)
      pos_ids.append(self.__pos_dictionary.get_index(END_POS))
      types.append(END_TYPE)
      type_ids.append(self.__type_dictionary.get_index(END_TYPE))
      heads.append(0)
    return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, [lines, raw_text]), postags, pos_ids, xpostags, xpos_ids, lemmas, lemma_ids, heads, types, type_ids)



