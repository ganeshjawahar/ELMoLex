import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .variational_rnn import VarMaskedFastLSTM

class NeuralLangModel(nn.Module):
  def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_xpos, num_filters, kernel_size, hidden_size, num_layers, embed_word=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True, init_emb=False, delex=False):
    super(NeuralLangModel, self).__init__()

    self.word_embed = nn.Embedding(num_words, word_dim) if not delex else False
    self.pos_embed = nn.Embedding(num_pos, pos_dim) if pos else None 
    #self.xpos_embed = nn.Embedding(num_xpos, pos_dim) if pos else None
    self.char_embed = nn.Embedding(num_words, char_dim) if char else None

    self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if delex==False and char else None
    self.dropout_in = nn.Dropout2d(p=p_in)
    self.dropout_out = nn.Dropout2d(p=p_out)
    self.pos = pos
    self.char = char
    self.delex = delex

    dim_enc = 0
    if delex==False:
      dim_enc = word_dim
    if pos:
      dim_enc += pos_dim
    if delex==False and self.char:
      dim_enc += num_filters

    self.rnn = VarMaskedFastLSTM(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

    out_dim = hidden_size * 2
    self.decoder = nn.Linear(out_dim, num_words)

    if init_emb:
      self.initialize_embed(embed_word, pos_dim, char_dim)

  def initialize_embed(self, embed_word, pos_dim, char_dim):
    if self.delex==False:
      self.word_embed.weight.data = embed_word
    if self.pos:
      self.pos_embed.weight.data.uniform_(-(3.0/pos_dim), (3.0/pos_dim))
      #self.xpos_embed.weight.data.uniform_(-(3.0/pos_dim), (3.0/pos_dim))
    if self.delex==False and self.char:
      self.char_embed.weight.data.uniform_(-(3.0/char_dim), (3.0/char_dim))
    self.decoder.bias.data.zero_()

  def forward(self, input_word, input_char, input_pos, input_xpos):
    features = []
    if self.delex==False:
      # [batch, length, word_dim]
      word = self.word_embed(input_word)
      # apply dropout on input
      word = self.dropout_in(word)
      features.append(word)

    if self.delex==False and self.char:
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
      features.append(char)

    if self.pos:
      # [batch, length, pos_dim]
      pos = self.pos_embed(input_pos)
      #xpos = self.xpos_embed(input_xpos)
      # apply dropout on input
      pos = self.dropout_in(pos)
      #xpos = self.dropout_in(xpos)
      #pos = pos+xpos
      features.append(pos)

    input = features[0]
    for i in range(len(features)-1):
      input = torch.cat([input, features[i+1]], dim=2) 

    # output from rnn [batch, length, hidden_size]
    output, hn, _ = self.rnn(input, None, hx=None)

    # apply dropout for output
    # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
    output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

    # predict the target word
    output = self.decoder(output)

    return output




