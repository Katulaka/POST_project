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
          prob: prob of the start tokens, usually 1.
          state: decoder initial states.
        """
        self.tokens = tokens
        self.prob = prob
        self.state = state
        self.score = score

    def extend_(self, token, prob, new_state):
        """Extend the hypothesis with result from latest step.

        Args:
          token: latest token from decoding.
          prob: prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens+[token], self.prob+[prob],
                            new_state, self.score + np.log(prob))

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(prob = %.4f, tokens = %s)' % (self.prob,
                                                              self.tokens))


class BeamSearch(object):
    """Beam search."""

    def __init__(self, beam_size, start_token, end_token, max_steps):
        """Creates BeamSearch object.

        Args:
          model:
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def beam_search(self, encode_top_state, decode_topk, enc_bv):
        """Performs beam search for decoding.

         Args:
            sess: tf.Session, session
            enc_win: ndarray of shape (enc_length, 1),
                        the document ids to encode
            enc_wlen: ndarray of shape (1), the length of the sequnce

         Returns:
            hyps: list of Hypothesis, the best hypotheses found by beam search,
                    ordered by score
         """
        # Run the encoder and extract the outputs and final state.
        enc_state_batch = encode_top_state(enc_bv)
        decs = []
        decs_out_beam = []
        dec_len = len(enc_state_batch)
        for j, enc_state in enumerate(enc_state_batch):
            print ("Starting batch %d / %d" % (j+1, dec_len))
            enc_w_len = enc_bv['word']['len'][j] - 1
            for i, dec_in in enumerate(enc_state[1:enc_w_len]):
                dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                                    np.expand_dims(dec_in, axis=0),
                                    np.expand_dims(np.zeros_like(dec_in),
                                                    axis=0))
                res = []
                res_out_beam = []
                hyps = [Hypothesis([self._start_token], [1.0], dec_in_state, 0.0)]
                for steps in xrange(self._max_steps):
                    # Extend each hypothesis.
                    # The first step takes the best K results from first hyps.
                    # Following steps take the best K results from K*K hyps.
                    all_hyps = []
                    for hyp in hyps:
                        latest_token = [[hyp.latest_token]]
                        states = hyp.state
                        ids, probs, new_state = decode_topk(latest_token,
                                                            states,
                                                            [enc_state],
                                                            self._beam_size)
                        for j in xrange(self._beam_size):
                            all_hyps.append(hyp.extend_(ids[j],
                                            probs[j],
                                            new_state))

                    hyps = []
                    all_hyps_sorted = self.sort_hyps(all_hyps)
                    #collect completed hyps that are outside the beam
                    for h in self.out_beam_hyps(all_hyps_sorted):
                        if h.latest_token == self._end_token:
                            res_out_beam.append(h)

                    for h in self.best_hyps(all_hyps_sorted):
                        # Filter and collect any hypotheses that have the end token.
                        if h.latest_token == self._end_token:
                            # Pull the hypothesis off the beam
                            #if the end token is reached.
                            res.append(h)
                        elif len(res) >= self._beam_size \
                            and h.score < min(res, key=lambda h: h.score).score:
                            pass
                        else:
                            # Otherwise continue to the extend the hypothesis.
                            hyps.append(h)
                print ("Finished beam search for %d / %d" % (i+1, enc_w_len - 1))
                decs.append(self.best_hyps(self.sort_hyps(res)))
                decs_out_beam.append(res_out_beam)

        beams = dict()
        beams['tokens'] = [[h.tokens[1:-1] for h in r if len(h.tokens)>2]
                            for r in decs]
        beams['scores'] = [[h.score for h in r if len(h.tokens)>2]
                             for r in decs]
        return beams, decs_out_beam

    def sort_hyps(self, hyps):
        """Sort the hyps based on probs.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis from highest prob to lowest
        """
        return sorted(hyps, key=lambda h: h.score, reverse=True)

    def best_hyps(self, hyp_sort):
        """return top <beam_size> hyps.

        Args:
          hyp_sort: A list of sorted hypothesis.
        Returns:
          hyps: A sub list of top <beam_size> hyps.
        """
        return hyp_sort[:self._beam_size]

    def out_beam_hyps(self, hyp_sort):
        """return hyps outside the beam.

        Args:
          hyp_sort: A list of sorted hypothesis.
        Returns:
          hyps: A sub list of all hyps outside of the beam.
        """
        return hyp_sort[self._beam_size:]

    def greedy_beam_search(self, encode_top_state, decode_topk, enc_bv):
        """Performs beam search for decoding.

        Args:
          sess: tf.Session, session
          enc_win: ndarray of shape (enc_length, 1), the document ids to encode
          enc_wlen: ndarray of shape (1), the length of the sequnce

        Returns:
          hyps: list of Hypothesis, the best hypotheses found by beam search,
              ordered by score
        """
    # Run the encoder and extract the outputs and final state.
        enc_state_batch = encode_top_state(enc_bv)
        # Replicate the initial states K times for the first step.
        decs = []
        dec_len = len(enc_state_batch)
        for j, enc_state in enumerate(enc_state_batch):
            print ("Starting batch %d / %d" % (j+1, dec_len))
            enc_w_len = enc_wlen[j] - 1
            res = []
            for i, dec_in in enumerate(enc_state[1:enc_w_len]):
                dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                                    np.expand_dims(dec_in, axis=0),
                                    np.expand_dims(np.zeros_like(dec_in),
                                                    axis=0))
                hyp = Hypothesis([self._start_token], [1.0], dec_in_state, 0.0)
                for steps in xrange(self._max_steps):
                    latest_token = [[hyp.latest_token]]
                    if latest_token[0][0] == self._end_token:
                        break
                    states = hyp.state
                    ids, probs, new_state = decode_topk(sess,
                                                    latest_token,
                                                    states,
                                                    [enc_state],
                                                    1)
                    hyp = hyp.extend_(ids[0], probs[0], new_state)
                res.append(hyp)
            decs.append(res)
        beams = dict()
        beams['tokens'] = map(lambda r: map(lambda h: h.tokens[1:-1], r), decs)
        beams['scores'] = map(lambda r: map(lambda h: h.score, r), decs)
        return beams
