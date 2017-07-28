from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy
import utils.data_preproc as dp


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
        self.batch_size = batch_size
        self.w_embed_size = word_embedding_size
        self.t_embed_size = tag_embedding_size
        self.n_hidden_fw = n_hidden_fw
        self.n_hidden_bw = n_hidden_bw
        self.n_hidden_lstm = n_hidden_lstm
        self.w_vocab_size = word_vocabulary_size
        self.t_vocab_size = tag_vocabulary_size
        # self.num_steps = num_steps
        self.lr = learning_rate
        self.lr_decay_factor = learning_rate_decay_factor
        # self.max_grad_norm = max_gradient_norm
        self.adam = adam
        self.dtype = dtype


    def _add_placeholders(self):

        self.word_seq_lens = tf.placeholder(tf.int32, shape=[None],
                                            name='word-sequence-length')
        self.tag_seq_lens = tf.placeholder(tf.int32, shape=[None],
                                            name='tag-sequence-length')

        self.word_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                                    name="word-input")
        self.tag_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                                    name="tag-input")
        self.targets = tf.placeholder(tf.int32,shape=[None, None, None],
                                                        name="targets")


    def _add_embedding(self):

        with tf.name_scope('tag_embedding'):
            tag_embed_matrix_init = tf.random_uniform([self.t_vocab_size,
                                                self.t_embed_size], -1.0, 1.0)
            tag_embed_matrix = tf.Variable(tag_embed_matrix_init,
                                                    name='tag-embeddings')
            self.tag_embed = tf.nn.embedding_lookup(tag_embed_matrix,
                                            self.tag_inputs, name='tag-embed')

        with tf.name_scope('word_embedding'):
            word_embed_matrix_init = tf.random_uniform([self.w_vocab_size,
                                                self.w_embed_size], -1.0, 1.0)
            word_embed_matrix = tf.Variable(word_embed_matrix_init,
                                                name='word-embeddings')
            self.word_embed = tf.nn.embedding_lookup(word_embed_matrix,
                                        self.word_inputs, name='word-embed')


    def _add_bidi_layer(self):

        with tf.name_scope('bidirectional-LSTM-Layer'):
            # Bidirectional LSTM
            # Forward and Backward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_fw,
                                    forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_bw,
                                    forget_bias=1.0, state_is_tuple=True)

        # Get lstm cell output
            self.bidi_out, self.bidi_states = tf.nn.bidirectional_dynamic_rnn(
                                    lstm_fw_cell, lstm_bw_cell, self.word_embed,
                                    sequence_length=self.word_seq_lens,
                                    dtype=self.dtype)


    def _add_lstm_layer(self):

        with tf.name_scope('LSTM-Layer'):
            # LSTM
            lstm_init = tf.concat(self.bidi_out, 2, name='lstm-init')
            lstm_init = tf.reshape(lstm_init, [-1,
                                        self.n_hidden_fw + self.n_hidden_bw])
            # remove padding:
            mask = tf.not_equal(tf.reshape(self.word_inputs, [-1]), 0)
            self.dec_init_state = tf.boolean_mask(lstm_init, mask)
            self.lstm_init = tf.contrib.rnn.LSTMStateTuple(
                    self.dec_init_state, tf.zeros_like(self.dec_init_state))

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_lstm,
                                    forget_bias=1.0, state_is_tuple=True)
            self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                        self.tag_embed,
                                        initial_state=self.lstm_init,
                                        sequence_length=self.tag_seq_lens,
                                        dtype=self.dtype)

    def _add_pred_op(self):

        with tf.name_scope('predictions'):
            w_uniform_dist = tf.random_uniform([self.n_hidden_lstm,
                                            self.t_vocab_size], -1.0, 1.0)
            self.w_out = w_out = tf.Variable(w_uniform_dist, name='W-out')
            self.b_out = b_out = tf.Variable(tf.zeros([self.t_vocab_size]),
                                                                name='b-out')

            outputs_reshape = tf.reshape(self.lstm_out,
                                            [-1, self.n_hidden_lstm])
            self.logits = tf.matmul(outputs_reshape, w_out) + b_out
            lstm_out_sahpe = tf.shape(self.lstm_out)
            self.logits = tf.reshape(self.logits, [lstm_out_sahpe[0],
                                                    lstm_out_sahpe[1], -1])
            self.pred = tf.nn.softmax(self.logits, name='pred')


    def _add_train_op(self):

        self.learning_rate = tf.Variable(float(self.lr),
                                        trainable=False, dtype=self.dtype)

        self.learning_rate_decay_op = self.learning_rate.assign(
                            self.learning_rate * self.lr_decay_factor)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits =
                                            self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(cross_entropy)

        if self.adam:
            self.optimizer = tf.train.AdamOptimizer(learning_rate =
                                        self.learning_rate).minimize(self.loss,
                                                global_step=self.global_step)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(
                                        self.learning_rate).minimize(self.loss,
                                        global_step=self.global_step)


    def build_graph(self):

        with tf.variable_scope(self.scope_name):
            self.global_step = tf.Variable(0, trainable=False)
            self._add_placeholders()
            self._add_embedding()
            self._add_bidi_layer()
            self._add_lstm_layer()
            self._add_pred_op()
            # if self._hps.mode == 'train': #TODO
            self._add_train_op()

        all_variables = [k for k in tf.global_variables() if
                                    k.name.startswith(self.scope_name)]
        self.saver = tf.train.Saver(all_variables)


    def step(self, session, word_seq_lens, tag_seq_lens, word_inputs,
                                                    tag_inputs, targets):

        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs,
                        self.targets: targets}

        output_feed = [self.pred, self.loss, self.optimizer]
        outputs = session.run(output_feed, input_feed)
        return outputs


    def get_batch(self, train_data, tag_vocabulary_size, batch_size=32):

        def arr_dim(a): return 1 + arr_dim(a[0]) if (type(a) == list) else 0

        bv = dp.generate_batch(train_data, batch_size)
        bv_w = copy.copy(bv['word'])
        bv_t = copy.copy(bv['tag'])
        if arr_dim(bv_t.tolist()) == 3:
            bv_t = [x for y in bv_t for x in y]

        seq_len_w = map(lambda x: len(x), bv_w)
        dp.data_padding(bv_w)
        # import pdb; pdb.set_trace()
        bv_w = np.vstack([np.expand_dims(x, 0) for x in bv_w])

        bv_t = dp.add_xos(bv_t)
        seq_len_t = map(lambda x: len(x), bv_t)
        # import pdb; pdb.set_trace()
        bv_t_1hot = map(lambda x: dp._to_onehot(x, max(seq_len_t), tag_vocabulary_size), bv_t)
        bv_t_1hot = np.vstack([np.expand_dims(x, 0) for x in bv_t_1hot])
        dp.data_padding(bv_t)
        bv_t = np.vstack([np.expand_dims(x, 0) for x in bv_t])

        return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_1hot


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

    def decode_topk(self, sess, latest_tokens, dec_init_states, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.lstm_init: dec_init_states,
            self.tag_inputs: latest_tokens,
            self.tag_seq_lens: np.ones(1, np.int32)}
        # output_feed = [self.lstm_out, self.pred , self.lstm_state]
        output_feed = [self.pred , self.lstm_state]
        results = sess.run(output_feed,input_feed)
        # import pdb; pdb.set_trace()
        probs, states = results[0], results[1]
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        #TODO check ids are not shifted
        return topk_ids, topk_probs, states
