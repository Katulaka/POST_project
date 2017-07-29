# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

from six.moves import xrange
import tensorflow as tf
import numpy as np


class Hypothesis(object):
  """Defines a hypothesis during beam search."""

  def __init__(self, tokens, prob, state, score):
      """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
      """
      self.tokens = tokens
      self.prob = prob
      self.state = state
      self.score = score

  def _Extend(self, token, prob, new_state):
      """Extend the hypothesis with result from latest step.

      Args:
        token: latest token from decoding.
        log_prob: log prob of the latest decoded tokens.
        new_state: decoder output state. Fed to the decoder for next step.
      Returns:
        New Hypothesis with the results from latest step.
      """
      return Hypothesis(self.tokens + [token], self.prob + [prob],
                        new_state, self.score * prob)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def __str__(self):
    return ('Hypothesis(prob = %.4f, tokens = %s)' % (self.prob,
                                                          self.tokens))


class BeamSearch(object):
  """Beam search."""

  def __init__(self, model, beam_size, start_token, end_token, max_steps):
    """Creates BeamSearch object.

    Args:
      model: Seq2SeqAttentionModel.
      beam_size: int.
      start_token: int, id of the token to start decoding with
      end_token: int, id of the token that completes an hypothesis
      max_steps: int, upper limit on the size of the hypothesis
    """
    self._model = model
    self._beam_size = beam_size
    self._start_token = start_token
    self._end_token = end_token
    self._max_steps = max_steps

# def Extend(self, token, prob, new_state):
#   """Extend the hypothesis with result from latest step.
#
#     Args:
#       token: latest token from decoding.
#       log_prob: log prob of the latest decoded tokens.
#       new_state: decoder output state. Fed to the decoder for next step.
#     Returns:
#       New Hypothesis with the results from latest step.
#   """
#     self.tokens += [token]
#     self.prob += [prob]
#     self.state = new_state
  # def GreedyBeamSearch(self, sess, enc_inputs, enc_seqlen):
  #   """Performs beam search for decoding.
  #
  #   Args:
  #     sess: tf.Session, session
  #     enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode
  #     enc_seqlen: ndarray of shape (1), the length of the sequnce
  #
  #   Returns:
  #     hyps: list of Hypothesis, the best hypotheses found by beam search,
  #         ordered by score
  #   """
  #   # Run the encoder and extract the outputs and final state.
  #   dec_in_state = self._model.encode_top_state( sess, enc_inputs, enc_seqlen)
  #   # Replicate the initial states K times for the first step.
  #   results = []
  #   for i,dec_in in enumerate(dec_in_state):
  #       hyps = [Hypothesis([self._start_token], [1.0], dec_in)]
  #       for steps in xrange(self._max_steps):
  #           latest_tokens = [[hyp.latest_token for hyp in hyps]]
  #           states = [hyp.state for hyp in hyps]
  #
  #           topk_ids, topk_probs, new_states = self._model.decode_topk(
  #                     sess, latest_tokens, states)
  #           top_id = np.argsort(np.squeeze(topk_ids))[-1]
  #           top_prob = np.squeeze(topk_probs)[top_id]
  #           hyps[0].Extend(top_id, top_prob, new_states)
  #       results.extend(hyps)
  #       print ("Finished deocding %d" %(i))
  #   ind = [sum(enc_seqlen[:i]) for i in xrange(len(enc_seqlen)+1)]
  #   res = [results[ind[i]:ind[i+1]] for i in xrange(len(enc_seqlen))]
  #   return res

  def BeamSearch(self, sess, enc_inputs, enc_seqlen):
    """Performs beam search for decoding.

     Args:
        sess: tf.Session, session
        enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode
        enc_seqlen: ndarray of shape (1), the length of the sequnce

     Returns:
        hyps: list of Hypothesis, the best hypotheses found by beam search,
            ordered by score
     """
    # Run the encoder and extract the outputs and final state.
    dec_in_state = self._model.encode_top_state(sess, enc_inputs, enc_seqlen)
    # Replicate the initial states K times for the first step.
    dec_res = []
    tot_i = len(dec_in_state)
    for i, dec_in in enumerate(dec_in_state):
      results = []
      hyps = [Hypothesis([self._start_token], [1.0], dec_in, 1.0)]
      for steps in xrange(self._max_steps):
          # Extend each hypothesis.
          # The first step takes the best K results from first hyps. Following
          # steps take the best K results from K*K hyps.
          all_hyps = []
          for hyp in hyps:
              latest_tokens = [[hyp.latest_token]]
              states = [hyp.state]
              topk_ids, topk_probs, new_states = self._model.decode_topk(
                    sess, latest_tokens, states, self._beam_size)
              for bid in xrange(self._beam_size):
                  all_hyps.append(hyp._Extend(topk_ids[bid], topk_probs[bid],
                            new_states))
          # Filter and collect any hypotheses that have the end token.
          hyps = []
          for h in self._BestHyps(all_hyps):
              if h.latest_token == self._end_token:
                  # Pull the hypothesis off the beam if the end token is reached.
                  results.append(h)
              elif len(results) >= self._beam_size and \
                    h.score < min(map(lambda h: h.score, results)):
                    pass
              else:
                  # Otherwise continue to the extend the hypothesis.
                  hyps.append(h)

      print ("Finished deocding %d / %d" %(i+1, tot_i))
      #TODO verify with John results exceed beam size
      dec_res.append(self._BestHyps(results))
      beam_res = map(lambda res: map(lambda h: (h.tokens[1:-1], h.score),
                                                                res), dec_res)
    return beam_res


  def _BestHyps(self, hyps):
    """Sort the hyps based on log probs and length.

    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse prod prob order.
    """
    #This length normalization is only effective for the final results.
    hyp_sort = sorted(hyps, key=lambda h: h.score, reverse=True)
    return hyp_sort[:self._beam_size]
