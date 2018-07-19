import argparse
import json
import os

def parse_test_args():
  args_parser = argparse.ArgumentParser(description="ELMoLex Parser - Testing")

  args_parser.add_argument('--pred_folder', type=str, default='/home/ganesh/objects/neurogp/run1', help='folder containing the trained models, dictionaries and settings')

  args_parser.add_argument('--system_tb', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-eval-ud.conllu', help='Name of the CoNLL-U file with the predicted data.')
  args_parser.add_argument('--gold_tb', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-eval-gold.conllu', help='Name of the CoNLL-U file with the gold data.')
  args_parser.add_argument('--tb_out', type=str, default='pred_tree.conllu', help='file name for writing the predicted parse tree')

  # aux
  args_parser.add_argument('--lex_expand', action='store_false', help='should try to expand lexical information from UDLexicons for oov words?')
  args_parser.add_argument('--lexicon', type=str, default='/home/ganesh/data/conll/UDLexicons.0.2/UDLex_English-Apertium.conllul', help='path to the lexicon')
  args_parser.add_argument('--vocab_expand', action='store_false', help='should try to expand word vocabulary from pre-trained word embeddings for oov words?')
  args_parser.add_argument('--word_path', type=str, default='/home/ganesh/data/conll/fair_vectors_raw/cc.en.300.vec', help='path for word embedding dict')

  args_parser.add_argument('--epoch', type=int, default=-1, help='which model_epoch_<int> you want to use? Leave it to -1 for choosing the latest model.')

  args =  args_parser.parse_args()
  return args

def parse_train_args():
  args_parser = argparse.ArgumentParser(description="ELMoLex Parser - Training")

  args_parser.add_argument('--pos', action='store_false', help='use part-of-speech embedding.')
  args_parser.add_argument('--char', action='store_false', help='use character embedding and CNN.')
  args_parser.add_argument('--dry_run', action='store_true', help='run in small scale.')
  args_parser.add_argument('--decode', default='mst', choices=['mst', 'greedy'], help='decoding algorithm')

  args_parser.add_argument('--hidden_size', type=int, default=512, help='Number of hidden units in RNN')
  args_parser.add_argument('--arc_space', type=int, default=512, help='Dimension of tag space')
  args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
  args_parser.add_argument('--num_layers', type=int, default=3, help='Number of layers of RNN')
  args_parser.add_argument('--num_filters', type=int, default=100, help='Number of filters in CNN')
  args_parser.add_argument('--pos_dim', type=int, default=100, help='Dimension of POS embeddings')
  args_parser.add_argument('--char_dim', type=int, default=100, help='Dimension of Character embeddings')

  args_parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
  args_parser.add_argument('--batch_size', type=int, default=32, help='Number of sentences in each batch')
  args_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
  args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
  args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
  args_parser.add_argument('--p_rnn', nargs=2, type=float, help='dropout rate for RNN')
  args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
  args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
  args_parser.add_argument('--schedule', type=int, default=10, help='schedule for learning rate decay')

  args_parser.add_argument('--unk_replace', type=float, default=0.5, help='The rate to replace a singleton word with UNK')
  args_parser.add_argument('--punctuation', nargs='+', default="'.' '``' "''" ':' ','", type=str, help='List of punctuations')
  args_parser.add_argument('--word_path', type=str, default='/home/ganesh/data/conll/fair_vectors_raw/cc.en.300.vec', help='path for word embedding dict')

  args_parser.add_argument('--train_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-train.conllu', help='train conllu file')
  args_parser.add_argument('--dev_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-dev.conllu', help='dev conllu file')
  args_parser.add_argument('--test_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-test.conllu', help='test conllu file')

  args_parser.add_argument('--dest_path', type=str, default='/home/ganesh/objects/neurogp/run1', help='path for storing model outputs/predictions')

  # pre-init LSTM
  args_parser.add_argument('--prelstm_args', type=str, default='/home/ganesh/objects/neurogp/en_15/nlm/args.json', help='settings file for pre-trained lstm models. leave it empty for no initialization.')

  # elmo
  args_parser.add_argument('--elmo', action='store_false', help='use elmo features? (ensure lstm_dir is present)')

  # lexicon
  args_parser.add_argument('--lexicon', type=str, default='/home/ganesh/data/conll/UDLexicons.0.2/UDLex_English-Apertium.conllul', help='path to the lexicon')
  args_parser.add_argument('--lex_hot', action='store_true', help='use mult-hot vector for representing lexical information')
  args_parser.add_argument('--lex_embed', type=int, default=100, help='embedding dim for representing lexical information. set -1 if you dont want to embed lexical tokens')
  args_parser.add_argument('--lex_attn', type=str, default='Specific', help='what kind of attention to use to combine embeddings? None (Mean) or Specific or Group')
  
  # delex
  args_parser.add_argument('--delex', action='store_true', help='discard lexical information?')

  # aux
  args_parser.add_argument('--lex_trim', action='store_true', help='trim lexical vocab to keep just words in input set? should try to expand lex in test side if required')
  args_parser.add_argument('--vocab_trim', action='store_true', help='trim word vocab to keep just words in input set? should try to expand word in test side if required')

  args =  args_parser.parse_args()
  return args

def save_train_args(path, args, other_args):
  save_args = {}
  save_args['word_dim'] = other_args[0]
  save_args['num_words'] = other_args[1]
  save_args['char_dim'] = args.char_dim
  save_args['num_chars'] = other_args[2]
  save_args['pos_dim'] = args.pos_dim
  save_args['num_pos'] = other_args[3]
  save_args['num_xpos'] = other_args[9]
  save_args['num_filters'] = args.num_filters
  save_args['window'] = other_args[4]
  save_args['hidden_size'] = args.hidden_size
  save_args['num_layers'] = args.num_layers
  save_args['num_types'] = other_args[5]
  save_args['arc_space'] = args.arc_space
  save_args['type_space'] = args.type_space
  save_args['num_epochs'] = args.num_epochs
  save_args['word_path'] = args.word_path
  save_args['train_path'] = args.train_path
  save_args['dev_path'] = args.dev_path
  save_args['test_path'] = args.test_path
  save_args['use_char'] = args.char
  save_args['use_pos'] = args.pos
  save_args['punct_set'] = other_args[6]
  save_args['decode'] = other_args[7]
  save_args['batch_size'] = other_args[8]
  save_args['prelstm_args'] = args.prelstm_args
  save_args['lexicon'] = args.lexicon
  save_args['lex_attn'] = args.lex_attn
  save_args['lex_embed'] = args.lex_embed
  save_args['lex_hot'] = args.lex_hot
  save_args['elmo'] = args.elmo
  save_args['delex'] = args.delex
  save_args['lex_trim'] = args.lex_trim
  save_args['vocab_trim'] = args.vocab_trim
  file_path = os.path.join(path, 'args.json')
  json.dump(save_args, open(file_path, 'w'), indent=4)
  print('saved settings as a json in: '+str(file_path))

def load_train_args(path):
  return json.load(open(os.path.join(path, 'args.json')))





