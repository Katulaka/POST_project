from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy
import utils.data_preproc as dp


class NNModel(object):

    def __init__(self, config):
        self.config = config


    def train_step(self, session, word_seq_lens, tag_seq_lens, word_inputs,
                    tag_inputs, targets):
        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs,
                        self.targets: targets}
        output_feed = [self.train_op, self.summaries, self.loss, self.global_step]
        return session.run(output_feed, input_feed)


    def eval_step(self, session, word_seq_lens, tag_seq_lens, word_inputs,
                    tag_inputs, targets):
        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs,
                        self.targets: targets}
        output_feed = [self.summaries, self.loss, self.global_step]
        return session.run(output_feed, input_feed)


    def decode_step(self, session, word_seq_lens, tag_seq_lens, word_inputs,
                    tag_inputs, targets):
        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs,
                        self.targets: targets}
        output_feed = [self.outputs, self.global_step]
        return session.run(output_feed, input_feed))


    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        self.word_seq_lens = tf.placeholder(tf.int32,
                                            shape=[None],
                                            name='word-sequence-length')
        self.tag_seq_lens = tf.placeholder(tf.int32,
                                            shape=[None],
                                            name='tag-sequence-length')
        self.word_inputs = tf.placeholder(tf.int32,
                                            shape=[None, None],
                                            name="word-input")
        self.tag_inputs = tf.placeholder(tf.int32,
                                            shape=[None, None],
                                            name="tag-input")
        self.targets = tf.placeholder(tf.int32,
                                        shape=[None, None, None],
                                        name="targets")


    def _add_emb_rnn_layers(self):

        LSTM = tf.contrib.rnn.BasicLSTMCell
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

        # Look up embeddings for inputs.
        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable(
                        'word-embeddings',
                        initializer = tf.random_uniform(
                        word_vocabulary_size, word_embedding_size], -1.0, 1.0))
            emb_word_input = tf.nn.embedding_lookup(word_embed_matrix,
                            self.word_inputs, name='word-embed')
            tag_embedding = tf.get_variable(
                        'tag-embeddings',
                        initializer = tf.random_uniform(
                        [tag_vocabulary_size, tag_embedding_size], -1.0, 1.0))
            emb_tag_input = tf.nn.embedding_lookup(tag_embed_matrix,
                                    self.tag_inputs, name='tag-embed')

        with tf.variable_scope('encoder'):
            # Bidirectional LSTM
            # Forward and Backward direction cell
            fw_cell = LSTM(n_hidden_fw,
                                forget_bias=1.0, state_is_tuple=True)
            bw_cell = LSTM(n_hidden_bw,
                                forget_bias=1.0, state_is_tuple=True)

            # Get lstm cell output
            encoder_out, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                                        fw_cell,
                                        bw_cell,
                                        emb_word_input,
                                        sequence_length=self.word_seq_lens,
                                        dtype=dtype)

        with tf.variable_scope('projection'):
            w_out = tf.get_variable('W-out',
                        initializer = tf.random_uniform(
                        [n_hidden_lstm, tag_vocabulary_size], -1.0, 1.0))
            b_out = tf.get_variable('b-out',
                        initializer = tf.zeros([tag_vocabulary_size]))

        with tf.variable_scope('decoder'):
            # decoder
            self.decoder_init = tf.concat(encoder_out, 2, name = 'lstm-init')
            self.decoder_init = tf.reshape(self.decoder_init,
                                    [-1, n_hidden_fw + n_hidden_bw])
            # remove padding:
            mask = tf.not_equal(tf.reshape(self.word_inputs, [-1]), 0)
            self.decoder_init = tf.boolean_mask(self.decoder_init, mask)
            self.decoder_init = LSTMStateTuple(self.decoder_init,
                                        tf.zeros_like(self.decoder_init))

            self.dec_in_state = emb_tag_input

            cell = LSTM(n_hidden_lstm, forget_bias=1.0, state_is_tuple=True)
            decoder_out, _ = tf.nn.dynamic_rnn(cell,
                                            self.dec_in_state,
                                            initial_state = self.decoder_init,
                                            equence_length = self.tag_seq_lens,
                                            dtype = dtype)

        with tf.variable_scope('output'):
            decoder_out_sahpe = tf.shape(decoder_out)
            decoder_out = tf.reshape(decoder_out, [-1, n_hidden_lstm])
            self.logits = tf.nn.xw_plus_b(decoder_out, w_out, b_out)
            self.pred = tf.nn.softmax(self.logits, name='pred')

        #TODO
        # if hps.mode == 'decode':
        #     with tf.variable_scope('decode_output'):
        #         best_outputs = [tf.argmax(x, 1) for x in self.logits]
        #         # tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
        #         self.outputs = tf.concat(
        #             axis=1, values=[tf.reshape(x, [batch_size, 1]) for x in best_outputs])
        #         self._topk_log_probs, self._topk_ids = tf.nn.top_k(
        #             tf.log(tf.nn.softmax(self.pred[-1])), hps.batch_size*2)

        with tf.variable_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.logits,
                            labels=self.targets)
            self.loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', tf.minimum(12.0, self.loss))


    def _add_train_op(self):
        config = self.config
        self.learning_rate = tf.get_variable(
                                'learning_rate',
                                initializer = float(learning_rate),
                                trainable = False,
                                dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self.loss, tvars), config.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')


    def build_graph(self):
        self._add_placeholders()
        self._add_emb_rnn_layers()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
          self._add_train_op()
        self.summaries = tf.summary.merge_all()

#TODO
    def encode_top_state(self, sess, enc_inputs, enc_len):
        """Return the top states from encoder for decoder."""
        input_feed = {
            self.word_inputs: enc_inputs,
            self.word_seq_lens: enc_len}
        output_feed = [self.decoder_init, self.dec_in_state]
        # results = sess.run(output_feed, input_feed)
        # return results[0], results[1][0]
        return sess.run(output_feed, input_feed)

  # def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
  #   """Return the topK results and new decoder states."""
  #   input_feed = {
  #       self.decoder_init: enc_top_states,
  #       self.dec_in_state:
  #           np.squeeze(np.array(dec_init_states)),
  #       self.tag_inputs:
  #           np.transpose(np.array([latest_tokens])),
  #       self.tag_seq_lens: np.ones([len(dec_init_states)], np.int32)}
  #
  #   output_feed = [self._topk_ids, self._topk_log_probs, self._dec_out_state]
  #   results = sess.run(output_feed,input_feed)
  #
  #   ids, probs, states = results[0], results[1], results[2]
  #   new_states = [s for s in states]
  #   return ids, probs, new_states
