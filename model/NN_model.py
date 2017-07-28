from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class NNModel(object):
    #
    # LSTM = tf.contrib.rnn.BasicLSTMCell
    # LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

    def __init__(self, batch_size, word_embedding_size, tag_embedding_size,
                n_hidden_fw, n_hidden_bw, n_hidden_lstm, word_vocabulary_size,
                tag_vocabulary_size,num_steps, learning_rate,
                learning_rate_decay_factor, max_gradient_norm, adam=True,
                dtype=tf.float32, scope_name='nn_model'):

        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):

            self.word_seq_lens = tf.placeholder(tf.int32, shape=[None],
                                                name='word-sequence-length')
            self.tag_seq_lens = tf.placeholder(tf.int32, shape=[None],
                                                name='tag-sequence-length')

            self.learning_rate = tf.Variable(float(learning_rate),
                                    trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                            self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

            with tf.name_scope('input'):
                self.word_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                                            name="word-input")
                self.tag_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                                            name="tag-input")
                self.targets = tf.placeholder(tf.int32,shape=[None, None, None],
                                                                name="targets")

            # Look up embeddings for inputs.
            with tf.name_scope('embedding'):
                word_embed_matrix_init = tf.random_uniform(
                    [word_vocabulary_size, word_embedding_size], -1.0, 1.0)
                word_embed_matrix = tf.Variable(word_embed_matrix_init,
                                                    name='word-embeddings')
                word_embed = tf.nn.embedding_lookup(word_embed_matrix,
                                        self.word_inputs, name='word-embed')

                tag_embed_matrix_init = tf.random_uniform(
                        [tag_vocabulary_size, tag_embedding_size], -1.0, 1.0)
                tag_embed_matrix = tf.Variable(tag_embed_matrix_init,
                                                        name='tag-embeddings')
                tag_embed = tf.nn.embedding_lookup(tag_embed_matrix,
                                            self.tag_inputs, name='tag-embed')

            with tf.name_scope('bidirectional-LSTM-Layer'):
                # Bidirectional LSTM
                # Forward and Backward direction cell
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_fw,
                                        forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_bw,
                                        forget_bias=1.0, state_is_tuple=True)

            # Get lstm cell output
                bidi_out, bidi_states = tf.nn.bidirectional_dynamic_rnn(
                                        lstm_fw_cell, lstm_bw_cell, word_embed,
                                        sequence_length=self.word_seq_lens,
                                        dtype=dtype)

                self.bidi_states = bidi_states
                self.bidi_out = bidi_out

            with tf.name_scope('Bridge'):
                # LSTM
                lstm_init = tf.concat(bidi_out, 2, name='lstm-init')
                lstm_init = tf.reshape(lstm_init, [-1, n_hidden_fw + n_hidden_bw])
                # remove padding:
                mask = tf.not_equal(tf.reshape(self.word_inputs, [-1]), 0)
                self.dec_init_state = tf.boolean_mask(lstm_init, mask)

            def output_tag_sequences():
                """Generate sequences of tags"""
                with tf.name_scope('LSTM-Layer'):
                    self.lstm_init = tf.contrib.rnn.LSTMStateTuple(
                                                self.dec_init_state,
                                                tf.zeros_like(self.dec_init_state))

                    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_lstm,
                                            forget_bias=1.0, state_is_tuple=True)
                    lstm_out, lstm_state = tf.nn.dynamic_rnn(lstm_cell, tag_embed,
                                            initial_state=self.lstm_init,
                                            sequence_length=self.tag_seq_lens,
                                            dtype=dtype)

                self.tag_embed = tag_embed
                self.lstm_out = lstm_out
                self.lstm_state =  lstm_state
                # compute softmax
                with tf.name_scope('predictions'):
                    w_uniform_dist = tf.random_uniform([n_hidden_lstm,
                                                tag_vocabulary_size], -1.0, 1.0)
                    self.w_out = w_out = tf.Variable(w_uniform_dist, name='W-out')
                    self.b_out = b_out = tf.Variable(
                                    tf.zeros([tag_vocabulary_size]), name='b-out')

                    outputs_reshape = tf.reshape(lstm_out, [-1, n_hidden_lstm])
                    self.proj = tf.matmul(outputs_reshape, w_out) + b_out

                    lstm_out_sahpe = tf.shape(lstm_out)
                    self.logits = tf.reshape(self.proj, [lstm_out_sahpe[0],
                                                            lstm_out_sahpe[1], -1])
                    self.pred = tf.nn.softmax(self.logits, name='pred')

                with tf.name_scope("loss"):
                    self.targets_flat = tf.reshape(self.targets, [-1, tag_vocabulary_size])
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.proj, labels=self.targets_flat)
                    self.loss = tf.reduce_mean(cross_entropy)

            def output_tags():
                """Generate only the first tag"""
                with tf.name_scope('predict-tags'):
                    w_uniform_dist = tf.random_uniform([n_hidden_fw + n_hidden_bw,
                                                        tag_vocabulary_size], -1.0, 1.0)
                    self.w_out = w_out = tf.Variable(w_uniform_dist, name='W-out')
                    self.b_out = b_out = tf.Variable(
                                    tf.zeros([tag_vocabulary_size]), name='b-out')
                    self.logits = tf.matmul(self.dec_init_state, w_out) + b_out
                    self.pred = tf.nn.softmax(self.logits, name='pred')

                with tf.name_scope("loss"):
                    first_targets_only = tf.slice(self.targets, (0, 1, 0), (-1, 1, -1)) # Keep only the first real tag (not "go")
                    self.tag_targets = tag_targets = tf.squeeze(first_targets_only, axis=1)
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tag_targets)
                    self.loss = tf.reduce_mean(cross_entropy)

            # output_tags()
            output_tag_sequences()

            if adam:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
            self.saver = tf.train.Saver(all_variables)


    def step(self, session, word_seq_lens, tag_seq_lens, word_inputs, tag_inputs, targets):

        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs,
                        self.targets: targets}

        output_feed = [self.pred, self.loss, self.optimizer]
        outputs = session.run(output_feed, input_feed)
        return outputs


    def encode_top_state(self, session, enc_inputs, enc_len):
        """Return the top states from encoder for decoder."""
        input_feed = {
            self.word_inputs: enc_inputs,
            self.word_seq_lens: enc_len}

        output_feed = self.dec_init_state
        dec_init_states = session.run(output_feed, input_feed)
        return [tf.contrib.rnn.LSTMStateTuple(np.expand_dims(i, axis=0),
                np.expand_dims(np.zeros_like(i), axis=0))
                for i in dec_init_states]

    # def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    def decode_topk(self, sess, latest_tokens, dec_init_states, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.lstm_init: dec_init_states,
            self.tag_inputs: latest_tokens,
            self.tag_seq_lens: np.ones(1, np.int32)}
        output_feed = [self.pred , self.lstm_state]
        results = sess.run(output_feed,input_feed)
        probs, states = results[0], results[1]
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        return topk_ids, topk_probs, states
