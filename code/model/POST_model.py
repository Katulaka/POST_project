from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class POSTModel(object):

    def __init__ (self, model_parms, mode, dtype=tf.float32, scope_name='nn_model'):

        self.scope_name = scope_name
        self.dtype = dtype
        self.mode = mode

        self.dim_char = model_parms.dim_char
        self.dim_word = model_parms.dim_word
        self.dim_tag = model_parms.dim_tag
        self.dim_pos = model_parms.dim_pos

        self.nchars = model_parms.nchars
        self.nwords = model_parms.nwords
        self.ntags = model_parms.ntags
        self.npos = model_parms.ntags

        self.hidden_char = model_parms.hidden_char
        self.hidden_pos = model_parms.hidden_pos
        self.hidden_word = model_parms.hidden_word
        self.hidden_tag = model_parms.hidden_tag

        self.lr = model_parms.lr
        self.lr_decay_factor = model_parms.lr_decay_factor

        self.pos = model_parms.pos
        self.use_pos = model_parms.use_pos
        self.attn = model_parms.attn
        self.use_c_embed = model_parms.use_c_embed
        self.comb_loss = model_parms.comb_loss

    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        #shape = (batch_size, max length of sentence, max lenght of word)
        self.char_in = tf.placeholder(tf.int32, [None, None], 'char-in')
        self.char_len = tf.placeholder(tf.int32, [None], 'char-sequence-len')
        #shape = (batch_size, max length of sentence)
        self.w_in = tf.placeholder(tf.int32, [None, None], 'word-input')
        #shape = (batch_size, max length of sentence)
        self.pos_in = tf.placeholder(tf.int32, [None, None], 'pos-input')
        #shape = (batch_size)
        self.word_len = tf.placeholder(tf.int32, [None], 'word-sequence-len')
        #shape = (batch_size * max length of sentence, max length of tag)
        self.t_in = tf.placeholder(tf.int32, [None, None], 'tag-input')
        #shape = (batch_size * max length of sentence)
        self.tag_len = tf.placeholder(tf.int32, [None], 'tag-sequence-len')
        #shape = (batch_size * length of sentences)
        self.targets = tf.placeholder(tf.int32, [None], 'targets')

    def _add_embeddings(self):
        """ Look up embeddings for inputs. """
        with tf.name_scope('embedding'):

            ch_embed_mat = tf.get_variable('char-embedding', dtype=self.dtype,
                            shape=[self.nchars,self.dim_char],
                            initializer=tf.contrib.layers.xavier_initializer())
            self.char_embed = tf.nn.embedding_lookup(ch_embed_mat,
                                                    self.char_in,
                                                    name='char-embed')

            self.w_embed_mat = tf.get_variable('word-embeddings',
                            shape=[self.nwords,self.dim_word], dtype=self.dtype,
                            initializer=tf.contrib.layers.xavier_initializer())
            self.word_embed = tf.nn.embedding_lookup(self.w_embed_mat,
                                                    self.w_in,
                                                    name='word-embed')

            t_embed_mat = tf.get_variable('tag-embeddings',
                            shape=[self.ntags,self.dim_tag],dtype=self.dtype,
                            initializer=tf.contrib.layers.xavier_initializer())
            self.tag_embed = tf.nn.embedding_lookup(t_embed_mat,
                                                    self.t_in,
                                                    name='tag-embed')

            pos_embed_mat = tf.get_variable('pos-embeddings',
                            shape=[self.npos,self.dim_pos], dtype=self.dtype,
                            initializer=tf.contrib.layers.xavier_initializer())
            # p_embed_mat_init = tf.random_uniform([self.npos, self.dim_pos],
            #                                         -1.0, 1.0)
            # pos_embed_mat = tf.Variable(p_embed_mat_init, name='pos-embedding',
            #                             dtype = self.dtype)
            self.pos_embed = tf.nn.embedding_lookup(pos_embed_mat,
                                                    self.pos_in,
                                                    name='pos-embed')

    def _add_char_lstm(self):
        with tf.name_scope('char-LSTM-Layer'):
            char_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_char)

            _, ch_state = tf.nn.dynamic_rnn(char_cell,
                                            self.char_embed,
                                            sequence_length=self.char_len,
                                            dtype=self.dtype,
                                            scope='char-lstm')

            W_char = tf.get_variable('W_char', dtype=self.dtype,
                            shape=[self.hidden_char, self.dim_word],
                            initializer=tf.contrib.layers.xavier_initializer())

            char_out = tf.einsum('aj,jk->ak', ch_state[1], W_char)
            char_out_reshape =  tf.reshape(char_out, tf.shape(self.word_embed))
            self.word_embed_f = tf.concat([self.word_embed, char_out_reshape],
                                        -1, 'mod_word_embed')
            self.dim_word_f = self.dim_word * 2

    def _add_char_bridge(self):
        with tf.name_scope('char-Bridge'):
            self.word_embed_f = self.word_embed
            self.dim_word_f = self.dim_word

    def _add_pos_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.name_scope('pos-LSTM-Layer'):
            # Forward and Backward direction cell
            pos_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_pos)
            pos_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_pos)
            # Get lstm cell output
            self._pos_out, self.pos_s = tf.nn.bidirectional_dynamic_rnn(
                                                pos_cell_fw,
                                                pos_cell_bw,
                                                self.word_embed_f,
                                                sequence_length=self.word_len,
                                                dtype=self.dtype,
                                                scope='pos-bidi')

            self.pos_out = tf.concat(self._pos_out, -1, name='pos_out')

    def _add_pos_prediction(self):
        with tf.name_scope('POS-prediction'):
            self.pos_logits = tf.layers.dense(self.pos_out, self.ntags)
            self.pos_pred = tf.argmax(self.pos_logits, 2, name='pos_pred')

    def _add_pos_loss(self):
        with tf.name_scope('POS-loss'):
            pos_in_1hot = tf.one_hot(self.pos_in, self.ntags)
            pos_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.pos_logits,
                                                    labels=pos_in_1hot)
            self.pos_loss = tf.reduce_mean(pos_cross_entropy)

    def _add_word_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.name_scope('word-bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            word_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_word)
            word_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_word)
            # Get lstm cell output
            self.w_bidi_in = tf.concat([self.word_embed_f, self.pos_embed],
                                        -1, 'word-bidi-in')
            w_bidi_out, _ = tf.nn.bidirectional_dynamic_rnn(
                                                word_cell_fw,
                                                word_cell_bw,
                                                self.w_bidi_in,
                                                sequence_length=self.word_len,
                                                dtype=self.dtype)
            self.w_bidi_out = tf.concat(w_bidi_out, -1, name='word-bidi-out')

    def _add_tag_lstm_bridge(self):
        with tf.name_scope('tag-LSTM-Bridge'):
            # LSTM
            self.attn_state = tf.concat([self.w_bidi_in, self.w_bidi_out], -1)
            self.attn_shape = tf.shape(self.attn_state)
            self.lstm_shape = self.dim_word_f + self.dim_pos + self.hidden_word * 2
            self.dec_init_state = tf.reshape(self.attn_state, [-1, self.lstm_shape])

    def _add_tag_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.name_scope('tag-LSTM-Layer'):
            self.tag_init = tf.contrib.rnn.LSTMStateTuple(self.dec_init_state,
                                        tf.zeros_like(self.dec_init_state))

            tag_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_shape)

            self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(tag_cell,
                                                self.tag_embed,
                                                initial_state=self.tag_init,
                                                sequence_length=self.tag_len,
                                                dtype=self.dtype)

    def _add_attention(self):
        with tf.name_scope('Attention'):
            lo_shape = tf.shape(self.lstm_out)

            self.atten_key = tf.reshape(self.lstm_out,
                                [self.attn_shape[0], -1, lo_shape[-1]])

            # TODO: add feed forward layer for atten_key

            alpha = tf.nn.softmax(tf.einsum('aij,akj->aik',
                                            self.atten_key,
                                            self.attn_state))
            score = tf.einsum('aij,ajk->aik', alpha, self.attn_state)

            score_tag = tf.reshape(score, [-1, lo_shape[-2], lo_shape[-1]])

            con_lstm_score = tf.concat([self.lstm_out, score_tag], -1)

            w_att = tf.get_variable('W-att', dtype = self.dtype,
                            shape=[self.lstm_shape * 2, self.hidden_tag],
                            initializer=tf.contrib.layers.xavier_initializer())

            # b_att = tf.Variable(tf.zeros([self.hidden_tag]), name='b-att')

            # lstm_att_pad = tf.einsum('aij,jk->aik', con_lstm_score, w_att)

            lstm_att_pad = tf.tanh(tf.einsum('aij,jk->aik', con_lstm_score,
                                                w_att))

            mask_t = tf.sequence_mask(self.tag_len)
            self.proj_in = tf.boolean_mask(lstm_att_pad, mask_t)

    def _add_project_bridge(self):
        w_proj = tf.get_variable('W-proj', dtype = self.dtype,
                            shape = [self.lstm_shape, self.hidden_tag],
                            initializer=tf.contrib.layers.xavier_initializer())
        proj_in_pad = tf.tanh(tf.einsum('aij,jk->aik', self.lstm_out, w_proj))
        mask_t = tf.sequence_mask(self.tag_len)
        self.proj_in = tf.boolean_mask(proj_in_pad, mask_t)

    def _add_projection(self):
        # compute softmax
        with tf.name_scope('predictions'):

            v = self.proj_in
            #E from notes
            E_out = tf.get_variable('E-out', dtype=self.dtype,
                            shape=[self.hidden_tag, self.ntags],
                            initializer=tf.contrib.layers.xavier_initializer())
            E_out_t = tf.transpose(E_out, name='E-out-t')
            b_out = tf.get_variable('b-out', shape=[self.ntags], dtype=self.dtype,
                            initializer=tf.contrib.layers.xavier_initializer())
            E_t_E = tf.matmul(E_out_t, E_out)
            E_v = tf.matmul(v, E_out)
            E_v_E_t_E = tf.matmul(E_v, E_t_E)
            self.logits = E_v + b_out
            self.mod_logits = E_v_E_t_E + b_out
            self.pred = tf.nn.softmax(self.logits, name='pred')

    def _add_loss(self):

        with tf.name_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.ntags)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.logits,
                                                    labels=targets_1hot)
            mod_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.mod_logits,
                                                    labels=targets_1hot)
            self.reg_loss = tf.reduce_mean(cross_entropy)

            self.mod_loss = tf.reduce_mean(mod_cross_entropy)
            self.mod_loss = (self.reg_loss + self.mod_loss)/2

            self.reg_loss = self.mod_loss if self.comb_loss else self.reg_loss


    def _add_train_op(self):

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
                if self.use_c_embed:
                    self._add_char_lstm()
                else:
                    self._add_char_bridge()
                with tf.name_scope("POS"):
                    self.build_pos_graph()
                with tf.name_scope("SUPERTAGS"):
                    self.build_suptag_graph()
                self.loss = self.pos_loss if self.pos else self.reg_loss
                if (self.mode == 'train'):
                    self._add_train_op()
                    self._step = self.pos_step if self.pos else self.suptag_step
                    self.increment_step_op = tf.assign(self._step, self._step+1)
            all_variables = [k for k in tf.global_variables()
                            if k.name.startswith(self.scope_name)]
            self.saver = tf.train.Saver(all_variables, max_to_keep = 4)

    def build_suptag_graph(self):
        """ Function builds the computation graph """
        self.suptag_step = tf.Variable(0, trainable=False, name='sup_step')
        self._add_word_bidi_lstm()
        self._add_tag_lstm_bridge()
        self._add_tag_lstm_layer()
        if self.attn:
            self._add_attention()
        else:
            self._add_project_bridge()
        self._add_projection()
        self._add_loss()
        if (self.mode == 'train'):
            self.reg_lr = tf.Variable(float(self.lr), trainable=False,
                                    dtype=self.dtype)

    def build_pos_graph(self):
        self.pos_step = tf.Variable(0, trainable=False, name='pos_step')
        self._add_pos_bidi_lstm()
        self._add_pos_prediction()
        self._add_pos_loss()
        if (self.mode == 'train'):
            self.pos_lr = tf.Variable(float(self.lr), trainable=False,
                                        dtype=self.dtype)

    def step(self, session, w_in, w_len, c_in, c_len, p_in, t_in, t_len, trgts):
        """ Training step, returns the prediction, loss"""
        if self.use_pos:
            p_in = self.pos_decode(session, w_in, w_len, c_in, c_len)
        input_feed = {
            self.w_in: w_in,
            self.word_len: w_len,
            self.char_in : c_in,
            self.char_len : c_len,
            self.pos_in : p_in,
            self.t_in: t_in,
            self.tag_len: t_len,
            self.targets: trgts}
        output_feed = [self.loss, self.optimizer, self.increment_step_op]
        return session.run(output_feed, input_feed)

    def dev_step(self, sess, w_in, w_len, c_in, c_len, p_in, t_in, t_len, trgt):
        """ Training step, returns the prediction, loss"""
        if self.use_pos:
            p_in = self.pos_decode(sess, w_in, w_len, c_in, c_len)
        input_feed = {self.w_in : w_in,
                        self.word_len : w_len,
                        self.char_in : c_in,
                        self.char_len : c_len,
                        self.pos_in : p_in,
                        self.t_in : t_in,
                        self.tag_len : t_len,
                        self.targets : trgt}
        output_feed = self.loss
        return sess.run(output_feed, input_feed)

    def pos_decode(self, session, w_in, w_len, c_in, c_len):
        input_feed = {self.w_in: w_in,
                    self.word_len: w_len,
                    self.char_in : c_in,
                    self.char_len : c_len}
        output_feed = self.pos_pred
        return session.run(output_feed, input_feed)

    def encode_top_state(self, sess, enc_win, enc_wlen, enc_cin, enc_clen, enc_pin):
        """Return the top states from encoder for decoder."""
        if self.use_pos:
            enc_pin = self.pos_decode(sess, enc_win, enc_wlen, enc_cin, enc_clen)
        input_feed = {self.w_in: enc_win,
                    self.word_len: enc_wlen,
                    self.char_in : enc_cin,
                    self.char_len : enc_clen,
                    self.pos_in: enc_pin}
        output_feed = self.attn_state
        return sess.run(output_feed, input_feed)

    def decode_topk(self, session, latest_tokens, dec_init_states, atten_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.tag_init : dec_init_states,
            self.t_in: np.array(latest_tokens),
            self.attn_state : atten_state,
            self.tag_len: np.ones(1, np.int32)}
        output_feed = [self.lstm_state, self.pred]
        states, probs = session.run(output_feed, input_feed)
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        return topk_ids, topk_probs, states
