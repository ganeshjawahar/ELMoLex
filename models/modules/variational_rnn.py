import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from ..functions import variational_rnn as rnn_F

def default_initializer(hidden_size):
  stdv = 1.0 / math.sqrt(hidden_size)
  def forward(tensor):
    nn.init.uniform_(tensor, -stdv, stdv)
  return forward

class VarMaskedRNNBase(nn.Module):
  def __init__(self, Cell, input_size, hidden_size,
               num_layers=1, bias=True, batch_first=False,
               dropout=(0, 0), bidirectional=False, initializer=None, **kwargs):
    super(VarMaskedRNNBase, self).__init__()
    self.Cell = Cell
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.bidirectional = bidirectional
    self.lstm = False
    num_directions = 2 if bidirectional else 1

    self.all_cells = []
    for layer in range(num_layers):
      for direction in range(num_directions):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions

        cell = self.Cell(layer_input_size, hidden_size, self.bias, p=dropout, initializer=initializer, **kwargs)
        self.all_cells.append(cell)
        self.add_module('cell%d' % (layer * num_directions + direction), cell)

  def reset_parameters(self):
    for cell in self.all_cells:
      cell.reset_parameters()

  def reset_noise(self, batch_size):
    for cell in self.all_cells:
      cell.reset_noise(batch_size)

  def forward(self, input, mask=None, hx=None):
    batch_size = input.size(0) if self.batch_first else input.size(1)
    if hx is None:
      num_directions = 2 if self.bidirectional else 1
      hx = torch.autograd.Variable(input.data.new(self.num_layers * num_directions, batch_size, self.hidden_size).zero_())
      if self.lstm:
        hx = (hx, hx)

    func = rnn_F.AutogradVarMaskedRNN(num_layers=self.num_layers,
                                      batch_first=self.batch_first,
                                      bidirectional=self.bidirectional,
                                      lstm=self.lstm)

    self.reset_noise(batch_size)

    output, hidden, master_output = func(input, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1,)))
    return output, hidden, master_output

  def step(self, input, hx=None, mask=None):
    '''
    execute one step forward (only for one-directional RNN).
    Args:
        input (batch, input_size): input tensor of this step.
        hx (num_layers, batch, hidden_size): the hidden state of last step.
        mask (batch): the mask tensor of this step.

    Returns:
        output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
        hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
    '''
    assert not self.bidirectional, "step only cannot be applied to bidirectional RNN."
    batch_size = input.size(0)
    if hx is None:
      hx = torch.autograd.Variable(input.data.new(self.num_layers, batch_size, self.hidden_size).zero_())
      if self.lstm:
        hx = (hx, hx)

    func = rnn_F.AutogradVarMaskedStep(num_layers=self.num_layers, lstm=self.lstm)

    output, hidden = func(input, self.all_cells, hx, mask)
    return output, hidden

class VarRNNCellBase(nn.Module):
  def __repr__(self):
    s = '{name}({input_size}, {hidden_size}'
    if 'bias' in self.__dict__ and self.bias is not True:
      s += ', bias={bias}'
    if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
      s += ', nonlinearity={nonlinearity}'
    s += ')'
    return s.format(name=self.__class__.__name__, **self.__dict__)

  def reset_noise(self, batch_size):
    """
    Should be overriden by all subclasses.
    Args:
        batch_size: (int) batch size of input.
    """
    raise NotImplementedError

class VarFastLSTMCell(VarRNNCellBase):
  """
  A long short-term memory (LSTM) cell with variational dropout.

  .. math::

      \begin{array}{ll}
      i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
      f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
      o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  Args:
      input_size: The number of expected features in the input x
      hidden_size: The number of features in the hidden state h
      bias: If `False`, then the layer does not use bias weights `b_ih` and
          `b_hh`. Default: True
      p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

  Inputs: input, (h_0, c_0)
      - **input** (batch, input_size): tensor containing input features
      - **h_0** (batch, hidden_size): tensor containing the initial hidden
        state for each element in the batch.
      - **c_0** (batch. hidden_size): tensor containing the initial cell state
        for each element in the batch.

  Outputs: h_1, c_1
      - **h_1** (batch, hidden_size): tensor containing the next hidden state
        for each element in the batch
      - **c_1** (batch, hidden_size): tensor containing the next cell state
        for each element in the batch

  Attributes:
      weight_ih: the learnable input-hidden weights, of shape
          `(4*hidden_size x input_size)`
      weight_hh: the learnable hidden-hidden weights, of shape
          `(4*hidden_size x hidden_size)`
      bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
      bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
  """

  def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None):
    super(VarFastLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
    self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
    if bias:
      self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
      self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
    else:
      self.register_parameter('bias_ih', None)
      self.register_parameter('bias_hh', None)

    self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
    self.reset_parameters()
    p_in, p_hidden = p
    if p_in < 0 or p_in > 1:
      raise ValueError("input dropout probability has to be between 0 and 1, "
                         "but got {}".format(p_in))
    if p_hidden < 0 or p_hidden > 1:
      raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                         "but got {}".format(p_hidden))
    self.p_in = p_in
    self.p_hidden = p_hidden
    self.noise_in = None
    self.noise_hidden = None

  def reset_parameters(self):
    for weight in self.parameters():
      if weight.dim() == 1:
        weight.data.zero_()
      else:
        self.initializer(weight.data)

  def reset_noise(self, batch_size):
    if self.training:
      if self.p_in:
        noise = self.weight_ih.data.new(batch_size, self.input_size)
        self.noise_in = Variable(noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in))
      else:
        self.noise_in = None

      if self.p_hidden:
        noise = self.weight_hh.data.new(batch_size, self.hidden_size)
        self.noise_hidden = Variable(noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden))
      else:
        self.noise_hidden = None
    else:
      self.noise_in = None
      self.noise_hidden = None

  def forward(self, input, hx):
    return rnn_F.VarFastLSTMCell(
      input, hx,
      self.weight_ih, self.weight_hh,
      self.bias_ih, self.bias_hh,
      self.noise_in, self.noise_hidden,
    )

class VarMaskedFastLSTM(VarMaskedRNNBase):
  r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
  sequence.


  For each element in the input sequence, each layer computes the following
  function:

  .. math::

          \begin{array}{ll}
          i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
          f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
          g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
          o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
          c_t = f_t * c_{(t-1)} + i_t * g_t \\
          h_t = o_t * \tanh(c_t)
          \end{array}

  where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
  state at time `t`, :math:`x_t` is the hidden state of the previous layer at
  time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
  :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
  and out gates, respectively.

  Args:
      input_size: The number of expected features in the input x
      hidden_size: The number of features in the hidden state h
      num_layers: Number of recurrent layers.
      bias: If False, then the layer does not use bias weights b_ih and b_hh.
          Default: True
      batch_first: If True, then the input and output tensors are provided
          as (batch, seq, feature)
      dropout: (dropout_in, dropout_hidden) tuple.
          If non-zero, introduces a dropout layer on the input and hidden of the each
          RNN layer with dropout rate dropout_in and dropout_hidden, resp.
      bidirectional: If True, becomes a bidirectional RNN. Default: False

  Inputs: input, mask, (h_0, c_0)
      - **input** (seq_len, batch, input_size): tensor containing the features
        of the input sequence.
        **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
      - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
        containing the initial hidden state for each element in the batch.
      - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
        containing the initial cell state for each element in the batch.


  Outputs: output, (h_n, c_n)
      - **output** (seq_len, batch, hidden_size * num_directions): tensor
        containing the output features `(h_t)` from the last layer of the RNN,
        for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
        given as the input, the output will also be a packed sequence.
      - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
        containing the hidden state for t=seq_len
      - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
        containing the cell state for t=seq_len
  """

  def __init__(self, *args, **kwargs):
    super(VarMaskedFastLSTM, self).__init__(VarFastLSTMCell, *args, **kwargs)
    self.lstm = True