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

# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_bool('normalize_by_length', True, 'Whether to normalize')


class Hypothesis(object):
  """Defines a hypothesis during beam search."""

  def __init__(self, tokens, prob, state):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
    """
    self.tokens = tokens
    self.prob = prob
    self.state = state

  def Extend(self, token, prob, new_state):
    """Extend the hypothesis with result from latest step.

    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    self.tokens += [token]
    self.prob += [prob]
    self.state = new_state
    # return Hypothesis(self.tokens + [token], self.prob + [prob],
    #                   new_state)

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
    # enc_top_states, dec_in_state = self._model.encode_top_state(
    dec_in_state = self._model.encode_top_state( sess, enc_inputs, enc_seqlen)
    # Replicate the initial states K times for the first step.
    results = []
    for i,dec_in in enumerate(dec_in_state):
        hyps = [Hypothesis([self._start_token], [1.0], dec_in)]
        # steps = 0
        # while steps < self._max_steps and len(results) < self._beam_size:
        for steps in xrange(self._max_steps):
            latest_tokens = [[hyp.latest_token for hyp in hyps]]
            states = [hyp.state for hyp in hyps]

            topk_ids, topk_probs, new_states = self._model.decode_topk(
                      sess, latest_tokens, states)
            # np.argsort(np.squeeze(topk_ids))[-self._beam_size:]
            top_id = np.argsort(np.squeeze(topk_ids))[-1]
            top_prob = np.squeeze(topk_probs)[top_id]
            hyps[0].Extend(top_id, top_prob, new_states)
            # Extend each hypothesis.
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            # all_hyps = []
            # num_beam_source = 1 if steps == 0 else len(hyps)
            # for i in xrange(num_beam_source):
            #     h, ns, tlp = hyps[i], new_states[i], topk_log_probs[i]
            #     for j in xrange(self._beam_size*2):
            #         all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))
            #     # for topk_id in topk_ids:
            #     #     all_hyps.append(h.Extend(topk_id, tlp, ns))
            #
            # # Filter and collect any hypotheses that have the end token.
            # hyps = []
            # for h in self._BestHyps(all_hyps):
            #     if h.latest_token == self._end_token:
            #         # Pull the hypothesis off the beam if the end token is reached.
            #         results.append(h)
            #     else:
            #         # Otherwise continue to the extend the hypothesis.
            #         hyps.append(h)
            #     if len(hyps) == self._beam_size or len(results) == self._beam_size:
            #         break
            # steps += 1

            # if steps == self._max_steps:
            #     results.extend(hyps)
            # return self._BestHyps(results)

        results.extend(hyps)
        print ("Finished deocding %d" %(i))
    ind = [sum(enc_seqlen[:i]) for i in xrange(len(enc_seqlen)+1)]
    res = [results[ind[i]:ind[i+1]] for i in xrange(len(enc_seqlen))]
    import pdb; pdb.set_trace()
    return res


  def _BestHyps(self, hyps, normalize_by_length=False):
    """Sort the hyps based on log probs and length.

    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    # This length normalization is only effective for the final results.
    if normalize_by_length:
        return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
    else:
        return sorted(hyps, key=lambda h: h.log_prob, reverse=True)













# def beam_search(prev, i, beam_size):
#
#     if output_projection is not None:
#         prev = tf.nn.xw_plus_b(
#             prev, output_projection[0], output_projection[1])
#     probs = tf.log(tf.nn.softmax(prev))
#
#     if i > 1:
#         probs = tf.reshape(probs + log_beam_probs[-1],
#         [-1, beam_size * num_symbols])
#     best_probs, indices = tf.nn.top_k(probs, beam_size)
#     indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
#     best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))
#
#     symbols = indices % num_symbols # Which word in vocabulary.
#     beam_parent = indices // num_symbols # Which hypothesis it came from.
#
#     beam_symbols.append(symbols)
#     beam_path.append(beam_parent)
#     log_beam_probs.append(best_probs)
#     return tf.nn.embedding_lookup(embedding, symbols)
#
#     # Compute
#     #  log P(next_word, hypothesis) =
#     #  log P(next_word | hypothesis)*P(hypothesis) =
#     #  log P(next_word | hypothesis) + log P(hypothesis)
#     # for each hypothesis separately, then join them together
#     # on the same tensor dimension to form the example's
#     # beam probability distribution:
#     # [P(word1, hypothesis1), P(word2, hypothesis1), ...,
#     #  P(word1, hypothesis2), P(word2, hypothesis2), ...]
#
#     # If TF had a log_sum_exp operator, then it would be
#     # more numerically stable to use:
#     #   probs = prev - tf.log_sum_exp(prev, reduction_dims=[1])
#     # i == 1 corresponds to the input being "<GO>", with
#     # uniform prior probability and only the empty hypothesis
#     # (each row is a separate example).
#
#     # Get the top `beam_size` candidates and reshape them such
#     # that the number of rows = batch_size * beam_size, which
#     # allows us to process each hypothesis independently.
#
#
#
#
#
# # log_beam_probs: list of [beam_size, 1] Tensors
# #  Ordered log probabilities of the `beam_size` best hypotheses
# #  found in each beam step (highest probability first).
# # beam_symbols: list of [beam_size] Tensors
# #  The ordered `beam_size` words / symbols extracted by the beam
# #  step, which will be appended to their corresponding hypotheses
# #  (corresponding hypotheses found in `beam_path`).
# # beam_path: list of [beam_size] Tensor
# #  The ordered `beam_size` parent indices. Their values range
# #  from [0, `beam_size`), and they denote which previous
# #  hypothesis each word should be appended to.
# log_beam_probs, beam_symbols, beam_path  = [], [], []
#
# # Setting up graph.
# inputs = [tf.placeholder(tf.float32, shape=[None, num_symbols])
#           for i in range(num_steps)]
# for i in range(num_steps):
#     beam_search(inputs[i], i + 1)
#
# # Running the graph.
# input_vals = [0, 0, 0]
# l = np.log
# eps = -10 # exp(-10) ~= 0
#
# # These values mimic the distribution of vocabulary words
# # from each hypothesis independently (in log scale since
# # they will be put through exp() in softmax).
# input_vals[0] = np.array([[0, eps, l(2), eps, l(3)]])
# # Step 1 beam hypotheses =
# # (1) Path: [4], prob = log(1 / 2)
# # (2) Path: [2], prob = log(1 / 3)
# # (3) Path: [0], prob = log(1 / 6)
#
# input_vals[1] = np.array([[l(1.2), 0, 0, l(1.1), 0], # Path [4]
#                           [0,   eps, eps, eps, eps], # Path [2]
#                           [0,  0,   0,   0,   0]])   # Path [0]
# # Step 2 beam hypotheses =
# # (1) Path: [2, 0], prob = log(1 / 3) + log(1)
# # (2) Path: [4, 0], prob = log(1 / 2) + log(1.2 / 5.3)
# # (3) Path: [4, 3], prob = log(1 / 2) + log(1.1 / 5.3)
#
# input_vals[2] = np.array([[0,  l(1.1), 0,   0,   0], # Path [2, 0]
#                           [eps, 0,   eps, eps, eps], # Path [4, 0]
#                           [eps, eps, eps, eps, 0]])  # Path [4, 3]
# # Step 3 beam hypotheses =
# # (1) Path: [4, 0, 1], prob = log(1 / 2) + log(1.2 / 5.3) + log(1)
# # (2) Path: [4, 3, 4], prob = log(1 / 2) + log(1.1 / 5.3) + log(1)
# # (3) Path: [2, 0, 1], prob = log(1 / 3) + log(1) + log(1.1 / 5.1)
#
# input_feed = {inputs[i]: input_vals[i][:beam_size, :]
#               for i in xrange(num_steps)}
# output_feed = beam_symbols + beam_path + log_beam_probs
# session = tf.InteractiveSession()
# outputs = session.run(output_feed, feed_dict=input_feed)
#
# expected_beam_symbols = [[4, 2, 0],
#                          [0, 0, 3],
#                          [1, 4, 1]]
# expected_beam_path = [[0, 0, 0],
#                       [1, 0, 0],
#                       [1, 2, 0]]
#
# print("predicted beam_symbols vs. expected beam_symbols")
# for ind, predicted in enumerate(outputs[:num_steps]):
#     print(list(predicted), expected_beam_symbols[ind])
# print("\npredicted beam_path vs. expected beam_path")
# for ind, predicted in enumerate(outputs[num_steps:num_steps * 2]):
#     print(list(predicted), expected_beam_path[ind])
# print("\nlog beam probs")
# for log_probs in outputs[2 * num_steps:]:
#     print(log_probs)
