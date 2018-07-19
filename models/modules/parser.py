import numpy as np
import re
import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .variational_rnn import VarMaskedFastLSTM
from .linear import BiLinear
from .attention import BiAttention

class BiRecurrentConvBiAffine(nn.Module):
  def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size, hidden_size, num_layers, num_labels, arc_space, type_space, embed_word=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True, init_emb=False):
    super(BiRecurrentConvBiAffine, self).__init__()

    self.word_embed = nn.Embedding(num_words, word_dim)
    self.pos_embed = nn.Embedding(num_pos, pos_dim) if pos else None 
    self.char_embed = nn.Embedding(num_words, char_dim) if char else None

    self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
    self.dropout_in = nn.Dropout2d(p=p_in)
    self.dropout_out = nn.Dropout2d(p=p_out)
    self.num_labels = num_labels
    self.pos = pos
    self.char = char

    dim_enc = word_dim
    if pos:
      dim_enc += pos_dim
    if char:
      dim_enc += num_filters

    self.rnn = VarMaskedFastLSTM(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

    out_dim = hidden_size * 2
    self.arc_h = nn.Linear(out_dim, arc_space)
    self.arc_c = nn.Linear(out_dim, arc_space)
    self.attention = BiAttention(arc_space, arc_space, 1, biaffine=biaffine)

    self.type_h = nn.Linear(out_dim, type_space)
    self.type_c = nn.Linear(out_dim, type_space)
    self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    if init_emb:
      self.initialize_embed(embed_word, pos_dim, char_dim)

  def initialize_embed(self, embed_word, pos_dim, char_dim):
    self.word_embed.weight.data = embed_word
    if self.pos:
      self.pos_embed.weight.data.uniform_(-(3.0/pos_dim), (3.0/pos_dim))
    if self.char:
      self.char_embed.weight.data.uniform_(-(3.0/char_dim), (3.0/char_dim))
  
  def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
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
    output, hn, _ = self.rnn(input, mask, hx=hx)

    # apply dropout for output
    # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
    output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

    # output size [batch, length, arc_space]
    arc_h = F.elu(self.arc_h(output))
    arc_c = F.elu(self.arc_c(output))

    # output size [batch, length, type_space]
    type_h = F.elu(self.type_h(output))
    type_c = F.elu(self.type_c(output))

    # apply dropout
    # [batch, length, dim] --> [batch, 2 * length, dim]
    arc = torch.cat([arc_h, arc_c], dim=1)
    type = torch.cat([type_h, type_c], dim=1)

    arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
    arc_h, arc_c = arc.chunk(2, 1)

    type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
    type_h, type_c = type.chunk(2, 1)
    type_h = type_h.contiguous()
    type_c = type_c.contiguous()

    return (arc_h, arc_c), (type_h, type_c), hn, mask, length

  def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
    # output from rnn [batch, length, tag_space]
    arc, type, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
    # [batch, length, length]
    out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
    return out_arc, type, mask, length

  def loss(self, input_word, input_char, input_pos, heads, types, mask=None, length=None, hx=None):
    # out_arc shape [batch, length, length]
    out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
    batch, max_len, _ = out_arc.size()

    if length is not None and heads.size(1) != mask.size(1):
      heads = heads[:, :max_len]
      types = types[:, :max_len]

    # out_type shape [batch, length, type_space]
    type_h, type_c = out_type

    # create batch index [batch]
    batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
    # get vector for heads [batch, length, type_space],
    type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
    # compute output for type [batch, length, num_labels]
    out_type = self.bilinear(type_h, type_c)

    # mask invalid position to -inf for log_softmax
    if mask is not None:
      minus_inf = -1e8
      minus_mask = (1 - mask) * minus_inf
      out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

    # loss_arc shape [batch, length, length]
    loss_arc = F.log_softmax(out_arc, dim=1)
    # loss_type shape [batch, length, num_labels]
    loss_type = F.log_softmax(out_type, dim=2)

    # mask invalid position to 0 for sum loss
    if mask is not None:
      loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
      loss_type = loss_type * mask.unsqueeze(2)
      # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
      num = mask.sum() - batch
    else:
      # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
      num = float(max_len - 1) * batch

    # first create index matrix [length, batch]
    child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
    child_index = child_index.type_as(out_arc.data).long()
    # [length-1, batch]
    loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
    loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

    return -loss_arc.sum() / num, -loss_type.sum() / num

  def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0, decode='mst'):
    if decode == 'mst':
      return self.decode_mst(input_word, input_char, input_pos, mask, length, hx, leading_symbolic)
    return self.decode_greedy(input_word, input_char, input_pos, mask, length, hx, leading_symbolic)
  
  def _decode_types(self, out_type, heads, leading_symbolic):
    # out_type shape [batch, length, type_space]
    type_h, type_c = out_type
    batch, max_len, _ = type_h.size()
    # create batch index [batch]
    batch_index = torch.arange(0, batch).type_as(type_h.data).long()
    # get vector for heads [batch, length, type_space],
    type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
    # compute output for type [batch, length, num_labels]
    out_type = self.bilinear(type_h, type_c)
    # remove the first #leading_symbolic types.
    out_type = out_type[:, :, leading_symbolic:]
    # compute the prediction of types [batch, length]
    _, types = out_type.max(dim=2)
    return types + leading_symbolic

  def decode_greedy(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
    # out_arc shape [batch, length, length]
    out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
    out_arc = out_arc.data
    batch, max_len, _ = out_arc.size()
    # set diagonal elements to -inf
    out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
    # set invalid positions to -inf
    if mask is not None:
      # minus_mask = (1 - mask.data).byte().view(batch, max_len, 1)
      minus_mask = (1 - mask.data).byte().unsqueeze(2)
      out_arc.masked_fill_(minus_mask, -np.inf)

    # compute naive predictions.
    # predition shape = [batch, length]
    _, heads = out_arc.max(dim=1)

    types = self._decode_types(out_type, heads, leading_symbolic)

    return heads.cpu().numpy(), types.data.cpu().numpy()

  def decode_mst(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
    '''
    Args:
        input_word: Tensor
            the word input tensor with shape = [batch, length]
        input_char: Tensor
            the character input tensor with shape = [batch, length, char_length]
        input_pos: Tensor
            the pos input tensor with shape = [batch, length]
        mask: Tensor or None
            the mask tensor with shape = [batch, length]
        length: Tensor or None
            the length tensor with shape = [batch]
        hx: Tensor or None
            the initial states of RNN
        leading_symbolic: int
            number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

    Returns: (Tensor, Tensor)
            predicted heads and types.

    '''
    # out_arc shape [batch, length, length]
    out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)

    # out_type shape [batch, length, type_space]
    type_h, type_c = out_type
    batch, max_len, type_space = type_h.size()

    # compute lengths
    if length is None:
      if mask is None:
        length = [max_len for _ in range(batch)]
      else:
        length = mask.data.sum(dim=1).long().cpu().numpy()

    type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
    type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
    # compute output for type [batch, length, length, num_labels]
    out_type = self.bilinear(type_h, type_c)

    # mask invalid position to -inf for log_softmax
    if mask is not None:
      minus_inf = -1e8
      minus_mask = (1 - mask) * minus_inf
      out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

    # loss_arc shape [batch, length, length]
    loss_arc = F.log_softmax(out_arc, dim=1)
    # loss_type shape [batch, length, length, num_labels]
    loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
    # [batch, num_labels, length, length]
    energy = torch.exp(loss_arc.unsqueeze(1) + loss_type)

    return self._decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)

  def _decode_MST(self, energies, lengths, leading_symbolic=0, labeled = True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 4D tensor
        energies of each edge. the shape is [batch_size, num_labels, n_steps, n_steps],
        where the summy root is at index 0.
    :param masks: numpy 2D tensor
        masks in the shape [batch_size, n_steps].
    :param leading_symbolic: int
        number of symbolic dependency types leading in type alphabets)
    :return:
    """

    def find_cycle(par):
      added = np.zeros([length], np.bool)
      added[0] = True
      cycle = set()
      findcycle = False
      for i in range(1, length):
        if findcycle:
          break

        if added[i] or not curr_nodes[i]:
          continue

        # init cycle
        tmp_cycle = set()
        tmp_cycle.add(i)
        added[i] = True
        findcycle = True
        l = i

        while par[l] not in tmp_cycle:
          l = par[l]
          if added[l]:
            findcycle = False
            break
          added[l] = True
          tmp_cycle.add(l)

        if findcycle:
          lorg = l
          cycle.add(lorg)
          l = par[lorg]
          while l != lorg:
            cycle.add(l)
            l = par[l]
          break

      return findcycle, cycle

    def chuLiuEdmonds():
      par = np.zeros([length], dtype=np.int32)
      # create best graph
      par[0] = -1
      for i in range(1, length):
        # only interested at current nodes
        if curr_nodes[i]:
          max_score = score_matrix[0, i]
          par[i] = 0
          for j in range(1, length):
            if j == i or not curr_nodes[j]:
              continue
            new_score = score_matrix[j, i]
            if new_score > max_score:
              max_score = new_score
              par[i] = j

      # find a cycle
      findcycle, cycle = find_cycle(par)
      # no cycles, get all edges and return them.
      if not findcycle:
        final_edges[0] = -1
        for i in range(1, length):
          if not curr_nodes[i]:
            continue
          pr = oldI[par[i], i]
          ch = oldO[par[i], i]
          final_edges[ch] = pr
        return

      cyc_len = len(cycle)
      cyc_weight = 0.0
      cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
      id = 0
      for cyc_node in cycle:
        cyc_nodes[id] = cyc_node
        id += 1
        cyc_weight += score_matrix[par[cyc_node], cyc_node]

      rep = cyc_nodes[0]
      for i in range(length):
        if not curr_nodes[i] or i in cycle:
          continue

        max1 = float("-inf")
        wh1 = -1
        max2 = float("-inf")
        wh2 = -1

        for j in range(cyc_len):
          j1 = cyc_nodes[j]
          if score_matrix[j1, i] > max1:
            max1 = score_matrix[j1, i]
            wh1 = j1
          scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]
          if scr > max2:
            max2 = scr
            wh2 = j1

        score_matrix[rep, i] = max1
        oldI[rep, i] = oldI[wh1, i]
        oldO[rep, i] = oldO[wh1, i]
        score_matrix[i, rep] = max2
        oldO[i, rep] = oldO[i, wh2]
        oldI[i, rep] = oldI[i, wh2]

      rep_cons = []
      for i in range(cyc_len):
        rep_cons.append(set())
        cyc_node = cyc_nodes[i]
        for cc in reps[cyc_node]:
          rep_cons[i].add(cc)

      for i in range(1, cyc_len):
        cyc_node = cyc_nodes[i]
        curr_nodes[cyc_node] = False
        for cc in reps[cyc_node]:
          reps[rep].add(cc)

      chuLiuEdmonds()

      # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
      found = False
      wh = -1
      for i in range(cyc_len):
        for repc in rep_cons[i]:
          if repc in final_edges:
            wh = cyc_nodes[i]
            found = True
            break
        if found:
          break

      l = par[wh]
      while l != wh:
        ch = oldO[par[l], l]
        pr = oldI[par[l], l]
        final_edges[ch] = pr
        l = par[l]

    if labeled:
      assert energies.ndim == 4, 'dimension of energies is not equal to 4'
    else:
      assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    for i in range(batch_size):
      energy = energies[i]

      # calc the realy length of this instance
      length = lengths[i]

      # calc real energy matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic types).
      if labeled:
        energy = energy[leading_symbolic:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0) + leading_symbolic
        energy = energy.max(axis=0)
      else:
        energy = energy[:length, :length]
        label_id_matrix = None
      # get original score matrix
      orig_score_matrix = energy
      # initialize score matrix to original score matrix
      score_matrix = np.array(orig_score_matrix, copy=True)

      oldI = np.zeros([length, length], dtype=np.int32)
      oldO = np.zeros([length, length], dtype=np.int32)
      curr_nodes = np.zeros([length], dtype=np.bool)
      reps = []

      for s in range(length):
        orig_score_matrix[s, s] = 0.0
        score_matrix[s, s] = 0.0
        curr_nodes[s] = True
        reps.append(set())
        reps[s].add(s)
        for t in range(s + 1, length):
          oldI[s, t] = s
          oldO[s, t] = t

          oldI[t, s] = t
          oldO[t, s] = s

      final_edges = dict()
      chuLiuEdmonds()
      par = np.zeros([max_length], np.int32)
      if labeled:
        type = np.ones([max_length], np.int32)
        type[0] = 0
      else:
        type = None

      for ch, pr in final_edges.items():
        par[ch] = pr
        if labeled and ch != 0:
          type[ch] = label_id_matrix[pr, ch]

      par[0] = 0
      pars[i] = par
      if labeled:
       types[i] = type
    return pars, types
  
  def compute_las(self, words, postags, heads_pred, types_pred, heads, types, word_dictionary, pos_dictionary, lengths, punct_set=None, symbolic_root=False, symbolic_end=False):
    def is_uni_punctuation(word):
      match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
      return match is not None
    
    def is_punctuation(word, pos, punct_set=None):
      if punct_set is None:
        return is_uni_punctuation(word)
      else:
        return pos in punct_set

    batch_size, _ = words.shape
    ucorr = 0.
    lcorr = 0.
    total = 0.
    ucomplete_match = 0.
    lcomplete_match = 0.

    ucorr_nopunc = 0.
    lcorr_nopunc = 0.
    total_nopunc = 0.
    ucomplete_match_nopunc = 0.
    lcomplete_match_nopunc = 0.

    corr_root = 0.
    total_root = 0.
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0

    for i in range(batch_size):
      ucm = 1.
      lcm = 1.
      ucm_nopunc = 1.
      lcm_nopunc = 1.
      for j in range(start, lengths[i] - end):
        word = word_dictionary.get_instance(words[i, j])
        pos = pos_dictionary.get_instance(postags[i, j])

        total += 1
        if heads[i, j] == heads_pred[i, j]:
          ucorr += 1
          if types[i, j] == types_pred[i, j]:
            lcorr += 1
          else:
            lcm = 0
        else:
          ucm = 0
          lcm = 0

        if not is_punctuation(word, pos, punct_set):
          total_nopunc += 1
          if heads[i, j] == heads_pred[i, j]:
            ucorr_nopunc += 1
            if types[i, j] == types_pred[i, j]:
              lcorr_nopunc += 1
            else:
              lcm_nopunc = 0
          else:
            ucm_nopunc = 0
            lcm_nopunc = 0

        if heads[i, j] == 0:
          total_root += 1
          corr_root += 1 if heads_pred[i, j] == 0 else 0

      ucomplete_match += ucm
      lcomplete_match += lcm
      ucomplete_match_nopunc += ucm_nopunc
      lcomplete_match_nopunc += lcm_nopunc

    return (ucorr, lcorr, total, ucomplete_match, lcomplete_match), batch_size






