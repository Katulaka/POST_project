from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class POSTModel(object):

    def __init__(self, batch_size, word_embedding_size, tag_embedding_size,
                n_hidden_fw, n_hidden_bw, n_hidden_lstm, word_vocabulary_size,
                tag_vocabulary_size, learning_rate, learning_rate_decay_factor,
                w_attention, pos, mode, loss_type,
                dtype=tf.float32, scope_name='nn_model'):

        self.scope_name = scope_name
        self.dtype = dtype

        self.w_embed_size = word_embedding_size
        self.t_embed_size = tag_embedding_size
        self.pos_embed_size = tag_embedding_size
        self.w_vocab_size = word_vocabulary_size
        self.t_vocab_size = tag_vocabulary_size
        self.pos_vocab_size = tag_vocabulary_size

        self.n_hidden_fw = n_hidden_fw
        self.n_hidden_bw = n_hidden_bw
        self.n_hidden_lstm = n_hidden_lstm

        self.lr = learning_rate
        self.lr_decay_factor = learning_rate_decay_factor

        self.mode = mode
        self.pos = pos
        self.w_attn = w_attention
        self.loss_type = loss_type

    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        self.w_in = tf.placeholder(tf.int32, [None, None], 'word-input')
        self.pos_in = tf.placeholder(tf.int32, [None, None], 'pos-input')
        self.w_seq_len = tf.placeholder(tf.int32, [None], 'word-sequence-len')
        self.t_in = tf.placeholder(tf.int32, [None, None], 'tag-input')
        self.t_seq_len = tf.placeholder(tf.int32, [None], 'tag-sequence-len')
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

    def _add_pos_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.name_scope('pos-LSTM-Layer'):
            # Forward and Backward direction cell
            pos_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_fw,
                                    forget_bias=1.0, state_is_tuple=True)
            pos_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_bw,
                                    forget_bias=1.0, state_is_tuple=True)
            # Get lstm cell output
            pos_bidi_out, pos_bidi_states = tf.nn.bidirectional_dynamic_rnn(
                                        pos_lstm_fw_cell,
                                        pos_lstm_fw_cell,
                                        self.word_embed,
                                        sequence_length=self.w_seq_len,
                                        dtype=self.dtype,
                                        scope='pos-bidi')
            self.pos_bidi_out = tf.concat(pos_bidi_out, 2, name='pos_bidi_out')

    def _add_pos_prediction(self):
        with tf.name_scope('POS-prediction'):
            self.pos_logits = tf.layers.dense(self.pos_bidi_out, self.t_vocab_size)
            self.pos_in_pred = tf.argmax(self.pos_logits, 2, name='pos_in_pred')

    def _add_pos_loss(self):
        with tf.name_scope('POS-loss'):
            self.pos_in_1hot = tf.one_hot(self.pos_in, self.t_vocab_size)
            self.pos_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.pos_logits,
                                                    labels=self.pos_in_1hot)
            self.pos_loss = tf.reduce_mean(self.pos_cross_entropy)

    def _add_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.name_scope('bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_fw,
                                    forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_bw,
                                    forget_bias=1.0, state_is_tuple=True)
            # Get lstm cell output
            self.bidi_in = tf.concat([self.word_embed, self.pos_embed], 2, 'bidi-in')
            self.bidi_out, self.bidi_states = tf.nn.bidirectional_dynamic_rnn(
                                        lstm_fw_cell,
                                        lstm_bw_cell,
                                        self.bidi_in,
                                        sequence_length=self.w_seq_len,
                                        dtype=self.dtype)
            self._bidi_out = tf.concat(self.bidi_out, 2, name='lstm-init')

    def _add_lstm_bridge(self):
        with tf.name_scope('Bridge'):
            # LSTM
            self.attn_state = tf.concat([self.bidi_in, self._bidi_out], 2)
            self.bo_shape = tf.shape(self.attn_state)
            self.lstm_shape = self.w_embed_size + self.pos_embed_size + \
                                self.n_hidden_fw + self.n_hidden_bw
            self.dec_init_state = tf.reshape(self.attn_state,
                                                [-1, self.lstm_shape])

    def _add_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.name_scope('LSTM-Layer'):
            self.lstm_init = tf.contrib.rnn.LSTMStateTuple(
                                        self.dec_init_state,
                                        tf.zeros_like(self.dec_init_state))

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_shape,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True)

            self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                self.tag_embed,
                                                initial_state=self.lstm_init,
                                                sequence_length=self.t_seq_len,
                                                dtype=self.dtype)

    def _add_attention(self):
        with tf.name_scope('Attention'):
            lo_shape = tf.shape(self.lstm_out)

            self.atten_key = tf.reshape(self.lstm_out,
                                [self.bo_shape[0], -1, lo_shape[-1]])

            # TODO: add feed forward layer for atten_key

            alpha = tf.nn.softmax(tf.einsum('aij,akj->aik',
                                            self.atten_key,
                                            self.attn_state))
            score = tf.einsum('aij,ajk->aik', alpha, self.attn_state)

            score_tag = tf.reshape(score, [-1, lo_shape[-2], lo_shape[-1]])

            con_lstm_score = tf.concat([self.lstm_out, score_tag], 2)

            w_att_uniform_dist = tf.random_uniform([self.lstm_shape * 2,
                                                    self.n_hidden_lstm],
                                                    -1.0, 1.0)
            w_att = tf.Variable(w_att_uniform_dist, name='W-att')
            # b_att = tf.Variable(tf.zeros([self.n_hidden_lstm]), name='b-att')

            lstm_att_pad = tf.einsum('aij,jk->aik', con_lstm_score, w_att)

            lstm_att_pad = tf.tanh(tf.einsum('aij,jk->aik', con_lstm_score,
                                                w_att))

            mask_t = tf.sequence_mask(self.t_seq_len)
            self.proj_in = tf.boolean_mask(lstm_att_pad, mask_t)

    def _add_project_bridge(self):
        w_dist = tf.random_uniform([self.lstm_shape, self.n_hidden_lstm],
                                    -1.0, 1.0)
        w_proj = tf.Variable(w_dist, name='W-dist')
        proj_in_pad = tf.tanh(tf.einsum('aij,jk->aik', self.lstm_out, w_proj))
        mask_t = tf.sequence_mask(self.t_seq_len)
        self.proj_in = tf.boolean_mask(proj_in_pad, mask_t)

    def _add_projection(self):
        # compute softmax
        with tf.name_scope('predictions'):

            v = self.proj_in

            #E from notes
            w_uniform_dist = tf.random_uniform([self.n_hidden_lstm,
                                                self.t_vocab_size],
                                                -1.0, 1.0)

            E_out = tf.Variable(w_uniform_dist, name='E-out')
            E_out_t = tf.transpose(E_out, name='E-out-t')
            b_out = tf.Variable(tf.zeros([self.t_vocab_size]), name='b-out')
            E_t_E = tf.matmul(E_out_t, E_out)
            E_v = tf.matmul(v, E_out)
            E_v_E_t_E = tf.matmul(E_v, E_t_E)
            self.logits = E_v + b_out
            self.mod_logits = E_v_E_t_E + b_out
            self.pred = tf.nn.softmax(self.logits, name='pred')


    def _add_loss(self):

        with tf.name_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.t_vocab_size)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.logits,
                                                    labels=targets_1hot)
            mod_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.mod_logits,
                                                    labels=targets_1hot)
            self.reg_loss = tf.reduce_mean(cross_entropy)
            # self.mod_loss = tf.reduce_mean(mod_cross_entropy)
            # self.comb_loss = tf.reduce_mean([loss, self.mod_loss]) #TODO
            # self.loss = loss if self.reg_loss else self.comb_loss


    def _add_train_op(self):

        self.loss = self.pos_loss if self.pos else self.reg_loss
        self.learning_rate = self.pos_lr if self.pos else self.reg_lr
        self.learning_rate_decay_op = self.learning_rate.assign(
                        self.learning_rate * self.lr_decay_factor)
        self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(
                        self.loss, global_step=self.global_step)

    def build_graph(self, graph):
        with graph.as_default():
            with tf.variable_scope(self.scope_name):
                self.global_step = tf.Variable(0, trainable=False, name='g_step')
                self._add_placeholders()
                self._add_embeddings()
                with tf.name_scope("POS"):
                    self.build_pos_graph()
                with tf.name_scope("SUPERTAGS"):
                    self.build_suptag_graph()
                if (self.mode == 'train'):
                    self._add_train_op()

            all_variables = [k for k in tf.global_variables()
                            if k.name.startswith(self.scope_name)]
            self.saver = tf.train.Saver(all_variables, max_to_keep = 4)

    def build_suptag_graph(self):
        """ Function builds the computation graph """
        self.suptag_step = tf.Variable(0, trainable=False, name='sup_step')
        self._add_bidi_lstm()
        self._add_lstm_bridge()
        self._add_lstm_layer()
        if self.w_attn:
            self._add_attention()
        else:
            self._add_project_bridge()
        self._add_projection()
        if (self.mode == 'train'):
            self._add_loss()
            self.reg_lr = tf.Variable(float(self.lr), trainable=False,
                                        dtype=self.dtype)

    def build_pos_graph(self):
        self.pos_step = tf.Variable(0, trainable=False, name='pos_step')
        self._add_pos_bidi_lstm()
        self._add_pos_prediction()
        if (self.mode == 'train'):
            self._add_pos_loss()
            self.pos_lr = tf.Variable(float(self.lr), trainable=False,
                                        dtype=self.dtype)


    def step(self, session, w_seq_len, t_seq_len, w_in, pos_in, t_in, targets):
        """ Training step, returns the prediction, loss"""

        input_feed = {
            self.w_seq_len: w_seq_len,
            self.t_seq_len: t_seq_len,
            self.w_in: w_in,
            self.t_in: t_in,
            self.pos_in : pos_in,
            self.targets: targets}
        if self.pos:
            self.increment_step_op = tf.assign(self.pos_step, self.pos_step+1)
        else:
            self.increment_step_op = tf.assign(self.suptag_step, self.suptag_step+1)
        output_feed = [self.loss, self.optimizer, self.increment_step_op]
        return session.run(output_feed, input_feed)

    def eval_step(self, session, w_seq_len, t_seq_len, w_in, pos_in, t_in, targets):
        """ Training step, returns the prediction, loss"""

        input_feed = {
            self.w_seq_len: w_seq_len,
            self.t_seq_len: t_seq_len,
            self.w_in: w_in,
            self.t_in: t_in,
            self.targets: targets,
            self.pos_in: pos_in}
        self.loss = self.reg_loss
        output_feed = self.loss
        return session.run(output_feed, input_feed)

    def pos_decode(self, session, w_in, w_seq_len):
        input_feed = {
            self.w_in: w_in,
            self.w_seq_len: w_seq_len}
        output_feed = self.pos_in_pred
        return session.run(output_feed, input_feed)

    def encode_top_state(self, session, enc_inputs, enc_len, enc_aux_inputs):
        """Return the top states from encoder for decoder."""
        input_feed = {
            self.w_in: enc_inputs,
            self.w_seq_len: enc_len,
            self.pos_in: enc_aux_inputs
            }
        output_feed = self.attn_state
        return session.run(output_feed, input_feed)

    def decode_topk(self, session, latest_tokens, dec_init_states, atten_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.lstm_init : dec_init_states,
            self.t_in: np.array(latest_tokens),
            self.attn_state : atten_state,
            self.t_seq_len: np.ones(1, np.int32)}
        output_feed = [self.lstm_state, self.pred]
        states, probs = session.run(output_feed, input_feed)
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        return topk_ids, topk_probs, states
