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
          score: decoder intial score.
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
                            new_state, self.score *prob)

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
        # decs_out_beam = []
        dec_len = len(enc_state_batch)
        #iterate over batch
        for j, enc_state in enumerate(enc_state_batch):
            print ("Starting batch %d / %d" % (j+1, dec_len))
            enc_w_len = enc_bv['word']['len'][j] - 1
            #iterate over words in seq
            for i, dec_in in enumerate(enc_state[1:enc_w_len]):
                dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                                    np.expand_dims(dec_in, axis=0),
                                    np.expand_dims(np.zeros_like(dec_in),
                                                    axis=0))
                res = []
                # res_out_beam = []
                hyps = [Hypothesis([self._start_token], [1.0], dec_in_state, 1.0)]
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
                    # all_hyps_sorted = self.sort_hyps(all_hyps)
                    #collect completed hyps that are outside the beam
                    # for h in self.out_beam_hyps(all_hyps_sorted):
                    #     if h.latest_token == self._end_token:
                    #         res_out_beam.append(h.tokens[1:-1])

                    for h in self.best_hyps(self.sort_hyps(all_hyps)):
                        # Filter and collect any hypotheses that have the end token.
                        if h.latest_token == self._end_token and len(h.tokens)>2:
                            # Pull the hypothesis off the beam
                            #if the end token is reached.
                            res.append(h)
                        elif h.latest_token == self._end_token:
                            pass
                        elif len(res) >= self._beam_size \
                            and h.score < min(res, key=lambda h: h.score).score:
                            pass
                        else:
                            # Otherwise continue to the extend the hypothesis.
                            hyps.append(h)
                print ("Finished beam search for %d / %d" % (i+1, enc_w_len - 1))
                decs.append(self.best_hyps(self.sort_hyps(res)))
                # decs_out_beam.append(res_out_beam)

        beams = dict()
        beams['tokens'] = [[h.tokens[1:-1] for h in r ] for r in decs]
        beams['scores'] = [[h.score for h in r ] for r in decs]
        return beams
        # # return beams, decs_out_beam

    # def beam_search_b(self, encode_top_state, decode_topk, enc_bv):
    #     """Performs beam search for decoding.
    #
    #      Args:
    #         sess: tf.Session, session
    #         enc_win: ndarray of shape (enc_length, 1), the document ids to encode
    #         enc_wlen: ndarray of shape (1), the length of the sequnce
    #
    #      Returns:
    #         hyps: list of Hypothesis, the best hypotheses found by beam search,
    #                 ordered by score
    #      """
    #     # Run the encoder and extract the outputs and final state.
    #     decs = []
    #     enc_state_batch = encode_top_state(enc_bv)
    #     batch = np.shape(enc_state_batch)[0]
    #     #iterate over max sentence lenght
    #     for dec_in in np.transpose(enc_state_batch, (1, 0, 2)):
    #         # initialization iterate over batch
    #         hyps = []
    #         for din in dec_in:
    #             hyp_state = tf.contrib.rnn.LSTMStateTuple(din, np.zeros_like(din))
    #             hyps.append([Hypothesis([self._start_token], [1.0], hyp_state, 1.0)])
    #
    #         res = [[]] * batch
    #         for steps in xrange(self._max_steps):
    #
    #             h_cell = []
    #             c_cell = []
    #             latest_tokens = []
    #             enc_state = []
    #             for hb, esb in zip(hyps, enc_state_batch):
    #                 for h in hb:
    #                     latest_tokens.append([h.latest_token])
    #                     h_cell.append(h.state[0])
    #                     c_cell.append(h.state[1])
    #                     enc_state.append(esb)
    #             states = tf.contrib.rnn.LSTMStateTuple(np.array(h_cell), np.array(c_cell))
    #             batch_size = len(latest_tokens)
    #             if batch_size != 0:
    #                 ids, probs, new_state = decode_topk(latest_tokens, states,
    #                                                 enc_state, batch_size,
    #                                                  self._beam_size)
    #             else:
    #                 break
    #
    #             hyps_ = []
    #             strt, end = 0, 0
    #             new_state = np.transpose(new_state, (1,0,2))
    #             for b, hb in enumerate(hyps):
    #                 all_hyps = []
    #                 end += len(hb)
    #                 ids_b = np.array(ids)[strt:end, :]
    #                 probs_b = np.array(probs)[strt:end, :]
    #                 new_state_b = np.array(new_state)[strt:end, :, :]
    #                 for hi, h in enumerate(hb):
    #                     h_cell = new_state_b[hi][0]
    #                     c_cell = new_state_b[hi][1]
    #                     state = tf.contrib.rnn.LSTMStateTuple(h_cell, c_cell)
    #                     all_hyps.extend([h.extend_(ids_b[hi][j],probs_b[hi][j], state)
    #                                     for j in xrange(self._beam_size)])
    #
    #                 hyps_b = []
    #                 for h in self.best_hyps(self.sort_hyps(all_hyps)):
    #                     # Filter and collect any hypotheses that have the end token.
    #                     if h.latest_token == self._end_token and len(h.tokens)>2:
    #                         # Pull the hypothesis off the beam
    #                         #if the end token is reached.
    #                         res[b].append(h)
    #                     elif h.latest_token == self._end_token:
    #                         pass
    #                     elif len(res[b]) >= self._beam_size and \
    #                         h.score < min(res[b], key=lambda h: h.score).score:
    #                         pass
    #                     else:
    #                         # Otherwise continue to the extend the hypothesis.
    #                         hyps_b.append(h)
    #                 hyps_.append(hyps_b)
    #                 strt = end
    #             hyps = hyps_
    #         decs.append([self.best_hyps(self.sort_hyps(rb)) for rb in res])
    #     decs = np.transpose(decs, (1,0,2))
    #     beams = {}
    #     for b, len_b in enumerate(enc_bv['word']['len']):
    #         decs_b = decs[b,1:len_b-1,:]
    #         beams.setdefault('tokens', []).append([[h.tokens[1:-1] for h in r ] for r in decs_b])
    #         beams.setdefault('scores', []).append([[h.score for h in r ] for r in decs_b])
    #     return beams


    def sort_hyps(self, hyps):
        """Sort the hyps based on probs.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis from highest prob to lowest
        """
        return sorted(hyps, key=lambda h: h.score, reverse=True)

    def filter_hyps(self, hyps):
        """Sort the hyps based on probs.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis from highest prob to lowest
        """
        return [h for h in hyps if h.latest_token != self._end_token]

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
