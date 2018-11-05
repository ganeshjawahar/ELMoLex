import argparse
import json
import os

def parse_train_args():
  args_parser = argparse.ArgumentParser(description="Neural Language Model")

  args_parser.add_argument('--pos', action='store_false', help='use part-of-speech embedding.')
  args_parser.add_argument('--char', action='store_false', help='use character embedding and CNN.')
  args_parser.add_argument('--dry_run', action='store_true', help='run in small scale.')

  args_parser.add_argument('--bptt', type=int, default=10, help='sequence length')
  args_parser.add_argument('--seed', type=int, default=123, help='random seed')

  args_parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units in RNN')
  args_parser.add_argument('--num_layers', type=int, default=3, help='Number of layers of RNN')
  args_parser.add_argument('--num_filters', type=int, default=100, help='Number of filters in CNN')
  args_parser.add_argument('--pos_dim', type=int, default=100, help='Dimension of POS embeddings')
  args_parser.add_argument('--char_dim', type=int, default=100, help='Dimension of Character embeddings')

  args_parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs')
  args_parser.add_argument('--batch_size', type=int, default=32, help='Number of sentences in each batch')
  args_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
  args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
  args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
  args_parser.add_argument('--p_rnn', nargs=2, type=float, help='dropout rate for RNN')
  args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
  args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
  args_parser.add_argument('--schedule', type=int, default=10, help='schedule for learning rate decay')

  args_parser.add_argument('--word_path', type=str, default='/home/ganesh/data/conll/fair_vectors_raw/cc.en.300.vec', help='path for word embedding dict')

  args_parser.add_argument('--train_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-train.conllu', help='train conllu file')
  args_parser.add_argument('--dev_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-dev.conllu', help='dev conllu file')
  args_parser.add_argument('--test_path', type=str, default='/home/ganesh/objects/conll18/udpipe-trained/direct/ud-en_lines/en_lines-ud-model-test.conllu', help='test conllu file')

  args_parser.add_argument('--dest_path', type=str, default='/home/ganesh/objects/neurogp/en_lines/nlm/run2', help='path for storing model outputs/predictions')
  
  # delex
  args_parser.add_argument('--delex', action='store_true', help='discard lexical information?')

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
  save_args['num_filters'] = args.num_filters
  save_args['window'] = other_args[4]
  save_args['hidden_size'] = args.hidden_size
  save_args['num_layers'] = args.num_layers
  save_args['num_epochs'] = args.num_epochs
  save_args['word_path'] = args.word_path
  save_args['train_path'] = args.train_path
  save_args['dev_path'] = args.dev_path
  save_args['test_path'] = args.test_path
  save_args['use_char'] = args.char
  save_args['use_pos'] = args.pos
  save_args['batch_size'] = other_args[5]
  save_args['bptt'] = args.bptt
  save_args['num_xpos'] = other_args[6]
  save_args['delex'] = args.delex
  save_args['n_trainable_params'] = other_args[7]
  file_path = os.path.join(path, 'args.json')
  json.dump(save_args, open(file_path, 'w'), indent=4)
  print('saved settings as a json in: '+str(file_path))


