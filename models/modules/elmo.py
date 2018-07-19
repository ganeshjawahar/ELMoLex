import json
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .variational_rnn import VarMaskedFastLSTM

class Elmo(nn.Module):
  def __init__(self, prelstm_args, p_rnn, kernel_size, p_in):
    super(Elmo, self).__init__()

    # read json file
    args = json.load(open(os.path.join(prelstm_args)))
    self.word_embed = nn.Embedding(args['num_words'], args['word_dim'])
    self.pos_embed = nn.Embedding(args['num_pos'], args['pos_dim']) if args['use_pos'] else None 
    self.char_embed = nn.Embedding(args['num_words'], args['char_dim']) if args['use_char'] else None
    self.conv1d = nn.Conv1d(args['char_dim'], args['num_filters'], kernel_size, padding=kernel_size - 1) if args['use_char'] else None
    self.dropout_in = nn.Dropout2d(p=p_in)
    dim_enc = args['word_dim']
    self.char = args['use_char']
    self.pos = args['use_pos']
    if self.pos:
      dim_enc += args['pos_dim']
    if self.char:
      dim_enc += args['num_filters']
    self.rnn = VarMaskedFastLSTM(dim_enc, args['hidden_size'], num_layers=args['num_layers'], batch_first=True, bidirectional=True, dropout=p_rnn)

    # load pretrained lstm model
    model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(prelstm_args[0:prelstm_args.rfind('/')], 'model_*.pt'))])
    model_id = str(model_id)
    print('elmo: loading model_epoch_%s.pt'%(model_id))
    model_path = os.path.join(prelstm_args[0:prelstm_args.rfind('/')], 'model_epoch_'+str(model_id)+'.pt')
    elmo_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # initialize the rnn parts
    state_dict = self.state_dict()
    new_dict = {}
    for k, v in elmo_dict.items():
      if k in state_dict:
        new_dict[k] = v
    state_dict.update(new_dict)
    self.load_state_dict(new_dict)

  def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
    # [batch, length, word_dim]
    word = self.word_embed(input_word)
    # apply dropout on input
    word = self.dropout_in(word)
    input = word

    if self.char:
      # [batch, length, char_length, char_dim]
      char = self.char_embed(input_char)
      char_size = char.size()
      # first transform to [batch *length, char_length, char_dim]
      # then transpose to [batch * lengtscreen -S your_session_nameh, char_dim, char_length]
      char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
      # put into cnn [batch*length, char_filters, char_length]
      # then put into maxpooling [batch * length, char_filters]
      char, _ = self.conv1d(char).max(dim=2)
      # reshape to [batch, length, char_filters]
      char = torch.tanh(char).view(char_size[0], char_size[1], -1)
      # apply dropout on input
      char = self.dropout_in(char)
      # concatenate word and char [batch, length, word_dim+char_filter]
      input = torch.cat([input, char], dim=2)

    if self.pos:
      # [batch, length, pos_dim]
      pos = self.pos_embed(input_pos)
      # apply dropout on input
      pos = self.dropout_in(pos)
      input = torch.cat([input, pos], dim=2)

    # output from rnn [batch, length, hidden_size]
    output, hn, master_output = self.rnn(input, mask, hx=hx)
    return master_output, word





