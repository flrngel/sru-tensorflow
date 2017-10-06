from linear import _Linear

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

class SRUCell(RNNCell):
  """Simple recurrent unit cell.
  The implementation is based on: https://arxiv.org/abs/1709.02755.
  """

  def __init__(self, num_units,
               state_is_tuple=True, activation=None, reuse=None):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(SRUCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    self._num_units = num_units
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh
    self._linear = None

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Single recurrent unit cell (SRU).
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size x 2 * self.state_size]`.
    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    if self._linear is None:
      self._linear = _Linear([inputs], 3 * self._num_units, True)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    x, f, r = array_ops.split(
        value=self._linear([inputs]), num_or_size_splits=3, axis=1)
    f = sigmoid(f)
    r = sigmoid(r)

    new_c = (
        (c * f) + ((1 - f) * x))
    new_h = (
        (r * self._activation(new_c)) + (1 - r) * inputs)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state
