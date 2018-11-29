import codecs
import sys
class CoNLLWriter(object):
  def __init__(self, word_dictionary, char_dictionary, pos_dictionary, type_dictionary):
    self.__source_file = None
    self.__word_dictionary = word_dictionary
    self.__char_dictionary = char_dictionary
    self.__pos_dictionary = pos_dictionary
    self.__type_dictionary = type_dictionary
    self.__out_data = {}

  def start(self, file_path):
    self.__source_file = codecs.open(file_path, 'w', 'utf-8')
    self.__file_path = file_path


  def close(self):
    self.__source_file.close()

  def store_buffer(self, word, pos, head, type, lengths, order_ids, raw_words, raw_lines, symbolic_root=False, symbolic_end=False):
    batch_size, _ = word.shape
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0
    for i in range(batch_size):
      sent_tokens = []
      for j in range(start, lengths[i] - end):
        w = raw_words[i][j]
        p = self.__pos_dictionary.get_instance(pos[i, j])
        t = self.__type_dictionary.get_instance(type[i, j])
        h = head[i, j]
        sent_tokens.append([w, h, t])
      self.__out_data[order_ids[i]] = [sent_tokens, raw_lines[i]]


  def write_buffer(self):
    for seq_no in range(len(self.__out_data)):
      sent_tokens, raw_lines = self.__out_data[seq_no]
      cur_ti = 0
      is_id = 0
      for ind, raw_row in enumerate(raw_lines[1]):
        if raw_row.startswith('# text = "id":'):
          is_id = 1
          ind_id = ind 

      if is_id:
        tweet_id_written = "# tweet_id ="+ raw_lines[1][ind_id][8:].strip()+"\n"
        self.__source_file.write(tweet_id_written)
        #self.__source_file.write(raw_lines[1][ind_id].strip()+"\n")
        continue
      else:
        for ind, raw_row in enumerate(raw_lines[1]):
          self.__source_file.write(raw_row.strip()+"\n")

      for ud_tokens in raw_lines[0]:
        if len(ud_tokens)<=9:
          try:
            catch_error = "/scratch/bemuller/parsing/sosweet/processing/logs/catching_errors.txt"
            open(catch_error,"a").write("Line broken {} on raw_lines {} of writted file  {} \n ".format(ud_tokens, raw_lines[1],self.__file_path))        
          except:
            print("WARNING ; coulnd not write to {} ".format(catch_error))
            print("Line broken {} on raw_lines {} of writted file  {} \n ".format(ud_tokens, raw_lines[1],self.__file_path))        
          continue 
        idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc = ud_tokens  

        if '-' not in idi and '.' not in idi:
        #if '-' not in idi:
          cur_model_tokens = sent_tokens[cur_ti]
          head, deprel = str(cur_model_tokens[1]), cur_model_tokens[2]
          cur_ti+=1
        self.__source_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc))
      self.__source_file.write("\n")
    


