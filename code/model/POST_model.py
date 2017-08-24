from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class POSTModel(object):

    def __init__(self, batch_size, word_embedding_size, tag_embedding_size,
                n_hidden_fw, n_hidden_bw, n_hidden_lstm, word_vocabulary_size,
                tag_vocabulary_size, learning_rate,learning_rate_decay_factor,
                add_pos_in, mode, dtype=tf.float32, scope_name='nn_model'):

        self.scope_name = scope_name
        self.w_embed_size = word_embedding_size
        self.t_embed_size = tag_embedding_size
        self.pos_embed_size = tag_embedding_size
        self.pos_vocab_size = tag_vocabulary_size
        self.w_vocab_size = word_vocabulary_size
        self.t_vocab_size = tag_vocabulary_size
        self.n_hidden_fw = n_hidden_fw
        self.n_hidden_bw = n_hidden_bw
        self.n_hidden_lstm = n_hidden_lstm
        self.dtype = dtype
        self.lr = learning_rate
        self.lr_decay_factor = learning_rate_decay_factor
        self.dtype = dtype
        self.mode = mode
        self.add_pos_in = add_pos_in

    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        self.w_in = tf.placeholder(tf.int32, [None, None], 'word-input')
        self.pos_in = tf.placeholder(tf.int32, [None, None], 'pos-input')
        self.w_seq_len = tf.placeholder(tf.int32, [None], 'word-sequence-length')
        self.t_in = tf.placeholder(tf.int32, [None, None], 'tag-input')
        self.t_seq_len = tf.placeholder(tf.int32, [None], 'tag-sequence-length')
        # self.targets = tf.placeholder(tf.int32, [None, None, None], 'targets')
        self.targets = tf.placeholder(tf.int32, [None], 'targets')


    def _add_embeddings(self):
        """ Look up embeddings for inputs. """
        with tf.name_scope('embedding'):
            w_embed_mat_init = tf.random_uniform([self.w_vocab_size,
                                                self.w_embed_size],
                                                -1.0, 1.0)
            w_embed_mat = tf.Variable(w_embed_mat_init, name='word-embeddings')
            self.word_embed = tf.nn.embedding_lookup(w_embed_mat,
                                                    self.w_in,
                                                    name='word-embed')

            t_embed_mat_init = tf.random_uniform([self.t_vocab_size,
                                                self.t_embed_size],
                                                -1.0, 1.0)
            t_embed_mat = tf.Variable(t_embed_mat_init, name='tag-embeddings')
            self.tag_embed = tf.nn.embedding_lookup(t_embed_mat,
                                                    self.t_in,
                                                    name='tag-embed')

            p_embed_mat_init = tf.random_uniform([self.pos_vocab_size,
                                                self.pos_embed_size],
                                                -1.0, 1.0)
            pos_embed_mat = tf.Variable(p_embed_mat_init, name='pos-embeddings')
            self.pos_embed = tf.nn.embedding_lookup(pos_embed_mat,
                                                    self.pos_in,
                                                    name='pos-embed')

    def _add_bidi_bridge(self):
        with tf.name_scope('BiDi-Bridge'):
            bidi_w_pos = tf.concat([self.word_embed, self.pos_embed],
                                    2, 'bidi-in')
            self.bidi_in = bidi_w_pos if self.add_pos_in else self.word_embed
            self.bidi_in_seq_len = self.w_seq_len

    def _add_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.name_scope('bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_fw,
                                    forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_bw,
                                    forget_bias=1.0, state_is_tuple=True)
            # Get lstm cell output
            self.bidi_out, self.bidi_states = tf.nn.bidirectional_dynamic_rnn(
                                    lstm_fw_cell, lstm_bw_cell,
                                    self.bidi_in,
                                    sequence_length=self.bidi_in_seq_len,
                                    dtype=self.dtype)

    def _add_bridge(self, special_tokens):
        with tf.name_scope('Bridge'):
            # LSTM
            lstm_init = tf.concat(self.bidi_out, 2, name='lstm-init')
            lstm_init = tf.reshape(lstm_init,
                                [-1, self.n_hidden_fw + self.n_hidden_bw])
            # remove padding:
            mask_pad = tf.not_equal(tf.reshape(self.w_in, [-1]),
                                            special_tokens['PAD'])
            mask_go = tf.not_equal(tf.reshape(self.w_in, [-1]),
                                    special_tokens['GO'])
            mask_eos = tf.not_equal(tf.reshape(self.w_in, [-1]),
                                    special_tokens['EOS'])

            mask = tf.logical_and(tf.logical_and(mask_pad, mask_go), mask_eos)
            self.dec_init_state = tf.boolean_mask(lstm_init, mask)

    def _add_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.name_scope('LSTM-Layer'):
            self.lstm_init = tf.contrib.rnn.LSTMStateTuple(
                                        self.dec_init_state,
                                        tf.zeros_like(self.dec_init_state))

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_lstm,
                                    forget_bias=1.0, state_is_tuple=True)
            self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                    self.tag_embed,
                                    initial_state=self.lstm_init,
                                    sequence_length=self.t_seq_len,
                                    dtype=self.dtype)


    def _add_projection(self):
        # compute softmax
        with tf.name_scope('predictions'):
            w_uniform_dist = tf.random_uniform([self.n_hidden_lstm,
                                                self.t_vocab_size],
                                                -1.0, 1.0)
            self.w_out = w_out = tf.Variable(w_uniform_dist, name='W-out')
            self.b_out = b_out = tf.Variable(tf.zeros([self.t_vocab_size]),
                                            name='b-out')

            outs_reshape = tf.reshape(self.lstm_out, [-1, self.n_hidden_lstm])
            self.proj = tf.matmul(outs_reshape, w_out) + b_out

            lstm_out_sahpe = tf.shape(self.lstm_out)
            self.logits = tf.reshape(self.proj,
                            [lstm_out_sahpe[0], lstm_out_sahpe[1], -1])
            self.pred = tf.nn.softmax(self.logits, name='pred')

    def _add_train_op(self, special_tokens):

        self.learning_rate = tf.Variable(float(self.lr),
                                trainable=False, dtype=self.dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
                        self.learning_rate * self.lr_decay_factor)

        with tf.name_scope("loss"):
            mask_pad = tf.not_equal(tf.reshape(self.t_in, [-1]),
                                    special_tokens['PAD'])
            proj = tf.boolean_mask(self.proj, mask_pad)
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #                             logits=proj, labels=self.targets)
            targets_1hot = tf.one_hot(self.targets, self.t_vocab_size)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                    logits=proj, labels=targets_1hot)
            self.loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_step)


    def build_graph(self, special_tokens):
        """ Function builds the computation graph """
        with tf.variable_scope(self.scope_name):
            self.global_step = tf.Variable(0, trainable=False, name='g_step')
            self._add_placeholders()
            self._add_embeddings()
            self._add_bidi_bridge()
            self._add_bidi_lstm()
            self._add_bridge(special_tokens)
            self._add_lstm_layer()
            self._add_projection()
            if (self.mode == 'train'):
                self._add_train_op(special_tokens)
        all_variables = [k for k in tf.global_variables()
                        if k.name.startswith(self.scope_name)]
        self.saver = tf.train.Saver(all_variables)


    def step(self, session, w_seq_len, t_seq_len, w_in, pos_in, t_in, targets):
        """ Training step, returns the prediction, loss"""
        input_feed = {
            self.w_seq_len: w_seq_len,
            self.t_seq_len: t_seq_len,
            self.w_in: w_in,
            self.t_in: t_in,
            self.targets: targets}
        if self.add_pos_in:
            input_feed[self.pos_in] = pos_in
        output_feed = [self.loss, self.optimizer]
        return session.run(output_feed, input_feed)

    def encode_top_state(self, session, enc_inputs, enc_len, enc_aux_inputs):
        """Return the top states from encoder for decoder."""
        input_feed = {
            self.w_in: enc_inputs,
            self.w_seq_len: enc_len}
        if self.add_pos_in:
            input_feed[self.pos_in] = enc_aux_inputs
        output_feed = self.dec_init_state
        dec_init_states = session.run(output_feed, input_feed)
        return [tf.contrib.rnn.LSTMStateTuple(np.expand_dims(i, axis=0),
                np.expand_dims(np.zeros_like(i), axis=0))
                for i in dec_init_states]

    def decode_topk(self, sess, latest_tokens, dec_init_states, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.lstm_init: dec_init_states,
            self.t_in: latest_tokens,
            self.t_seq_len: np.ones(1, np.int32)}
        output_feed = [self.pred , self.lstm_state]
        results = sess.run(output_feed, input_feed)
        probs, states = results[0], results[1]
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        return topk_ids, topk_probs, states

    # def decode_one_tag(sess, w_seq_len, words):
    #     input_feed = {
    #         self.word_inputs: words,
    #         self.w_seq_lens: w_seq_len
    #     }
    #     output_feed = self.logits
    #     results = sess.run(output_feed, input_feed)
    #     return np.argmax(results, axis=1)

    # def output_tags():
    #     """Generate only the first tag"""
    #     with tf.name_scope('predict-tags'):
    #         w_uniform_dist = tf.random_uniform([n_hidden_fw + n_hidden_bw,
    #                                             t_vocab_size], -1.0, 1.0)
    #         self.w_out = w_out = tf.Variable(w_uniform_dist, name='W-out')
    #         self.b_out = b_out = tf.Variable(
    #                         tf.zeros([t_vocab_size]), name='b-out')
    #         self.logits = tf.matmul(self.dec_init_state, w_out) + b_out
    #         self.pred = tf.nn.softmax(self.logits, name='pred')
    #
    #     with tf.name_scope("loss"):
    #         first_targets_only = tf.slice(self.targets, (0, 1, 0), (-1, 1, -1)) # Keep only the first real tag (not "go")
    #         self.tag_targets = tag_targets = tf.squeeze(first_targets_only, axis=1)
    #         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, targets=self.tag_targets)
    #         self.loss = tf.reduce_mean(cross_entropy)
