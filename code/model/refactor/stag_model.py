from __future__ import division
from __future__ import print_function

import copy
import math
import sys
import time

import numpy as np
import tensorflow as tf

from .basic_model import BasicModel
from beam.refactor.search import BeamSearch
from astar.search import solve_tree_search


class STAGModel(BasicModel):

    def __init__ (self, config):
        BasicModel.__init__(self, config)
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.optimizer_fn = tf.train.GradientDescentOptimizer(self.lr)
        self.init_op = tf.global_variables_initializer()
        if self.config['use_pretrained_pos']:
            self.pos_g = self.load_graph(self.config['frozen_graph_fname'])
            self.pos_sess = tf.Session(config = self.sess_config, graph = self.pos_g)

    def _add_placeholders(self):
        with tf.variable_scope('placeholders'):
            """Inputs to be fed to the graph."""
            #shape = (batch_size, max length of sentence, max lenght of word)
            self.char_in = tf.placeholder(tf.int32, [None, None], 'char-in')
            self.char_len = tf.placeholder(tf.int32, [None], 'char-seq-len')
            #shape = (batch_size, max length of sentence)
            self.w_in = tf.placeholder(tf.int32, [None, None], 'word-input')
            #shape = (batch_size, max length of sentence)
            self.pos_in = tf.placeholder(tf.int32, [None, None], 'pos-input')
            #shape = (batch_size)
            self.word_len = tf.placeholder(tf.int32, [None], 'word-seq-len')
            #shape = (batch_size * max length of sentence, max length of tag)
            self.t_in = tf.placeholder(tf.int32, [None, None], 'tag-input')
            #shape = (batch_size * max length of sentence)
            self.tag_len = tf.placeholder(tf.int32, [None], 'tag-seq-len')
            #shape = (batch_size * length of sentences)
            self.targets = tf.placeholder(tf.int32, [None], 'targets')

    def _add_embeddings(self):
        """ Look up embeddings for inputs. """
        with tf.variable_scope('embedding', initializer=self.initializer, dtype=self.dtype):

            ch_mat_shape = [self.config['nchars'], self.config['dim_char']]
            ch_embed_mat = tf.get_variable('ch-embed-mat', shape=ch_mat_shape)
            self.char_embed = tf.nn.embedding_lookup(ch_embed_mat, self.char_in,
                                                    name='char-embed')

            w_mat_shape = [self.config['nwords'], self.config['dim_word']]
            w_embed_mat = tf.get_variable('word-embed-mat', shape=w_mat_shape)
            self.word_embed = tf.nn.embedding_lookup(w_embed_mat, self.w_in,
                                                    name='word-embed')

            t_mat_shape = [self.config['ntags'], self.config['dim_tag']]
            t_embed_mat = tf.get_variable('tag-embed-mat', shape=t_mat_shape)
            self.tag_embed = tf.nn.embedding_lookup(t_embed_mat, self.t_in,
                                                    name='tag-embed')

            pos_mat_shape = [self.config['npos'], self.config['dim_pos']]
            pos_embed_mat = tf.get_variable('pos-embed-mat', shape=pos_mat_shape)
            self.pos_embed = tf.nn.embedding_lookup(pos_embed_mat, self.pos_in,
                                                    name='pos-embed')

    def _add_char_lstm(self):
        with tf.variable_scope('char-LSTM-Layer', initializer=self.initializer):
            char_cell = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_char'])

            _, ch_state = tf.nn.dynamic_rnn(char_cell,
                                            self.char_embed,
                                            sequence_length=self.char_len,
                                            dtype=self.dtype,
                                            scope='char-lstm')

            char_out = tf.layers.dense(ch_state[1], self.config['dim_word'],
                                        use_bias=False)
                                        # kernel_initializer= self.init)
            char_out_reshape =  tf.reshape(char_out, tf.shape(self.word_embed))
            self.word_embed_f = tf.concat([self.word_embed, char_out_reshape],
                                        -1, 'mod_word_embed')
            self.dim_word_f = self.config['dim_word'] * 2

    def _add_char_bridge(self):
        with tf.variable_scope('char-Bridge'):
            self.word_embed_f = self.word_embed
            self.dim_word_f = self.config['dim_word']

    def _add_word_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.variable_scope('word-bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            word_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_word'])
            word_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_word'])
            # Get lstm cell output
            self.w_bidi_in = tf.concat([self.word_embed_f, self.pos_embed], -1,
                                        name='word-bidi-in')
            w_bidi_out, _ = tf.nn.bidirectional_dynamic_rnn(word_cell_fw,
                                                            word_cell_bw,
                                                            self.w_bidi_in,
                                                            sequence_length=self.word_len,
                                                            dtype=self.dtype)
            self.w_bidi_out = tf.concat(w_bidi_out, -1, name='word-bidi-out')

            self.encode_state = tf.concat([self.w_bidi_in, self.w_bidi_out], -1)

    def _add_tag_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.variable_scope('tag-LSTM-Layer'):
            # lstm_shape = self.dim_word_f + self.config['dim_pos'] + self.config['hidden_word'] * 2
            # dec_init_state = tf.reshape(self.encode_state, [-1, lstm_shape])
            self.dec_in_dim = self.config['hidden_word'] * 2
            encode_state = tf.layers.dense(self.encode_state, self.dec_in_dim, use_bias=False)
            dec_init_state = tf.reshape(encode_state, [-1, self.dec_in_dim])

            self.tag_init = tf.contrib.rnn.LSTMStateTuple(dec_init_state,
                                        tf.zeros_like(dec_init_state))

            tag_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_in_dim)

            self.decode_out, self.decode_state = tf.nn.dynamic_rnn(tag_cell,
                                                self.tag_embed,
                                                initial_state=self.tag_init,
                                                sequence_length=self.tag_len,
                                                dtype=self.dtype)

    def _add_attention(self):
        with tf.variable_scope('Attention', initializer=self.initializer):
            do_shape = tf.shape(self.decode_out)
            es_shape = tf.shape(self.encode_state)
            ak_shape = [es_shape[0], -1, do_shape[-1]]
            atten_k = tf.reshape(self.decode_out, ak_shape)
            atten_q = tf.layers.dense(self.encode_state, self.dec_in_dim,
                                        activation=tf.nn.relu, use_bias=False)

            alpha = tf.nn.softmax(tf.einsum('aij,akj->aik', atten_k, atten_q))
                                            # self.encode_state))

            _context = tf.einsum('aij,ajk->aik', alpha, self.encode_state)
            c_dim = self.dim_word_f + self.config['dim_pos'] + self.dec_in_dim
            context = tf.reshape(_context, [do_shape[0],do_shape[1],c_dim])

            dec_out_w_attn = tf.concat([self.decode_out, context], -1)

            self.proj_in = tf.layers.dense(dec_out_w_attn,
                                            self.config['hidden_tag'],
                                            use_bias=False,
                                            activation=tf.tanh)

    def _add_projection(self):

        with tf.variable_scope('predictions', initializer=self.initializer, dtype=self.dtype):

            proj_in = tf.layers.dense(self.proj_in,
                                        self.config['hidden_tag'],
                                        use_bias=False,
                                        activation=tf.tanh)
            mask_t = tf.sequence_mask(self.tag_len)
            v = tf.boolean_mask(proj_in, mask_t)

            #E from notes
            E_out_shape = [self.config['hidden_tag'], self.config['ntags']]
            E_out = tf.get_variable('E-out', shape=E_out_shape)
            E_out_t = tf.transpose(E_out, name='E-out-t')
            b_out = tf.get_variable('b-out', shape=[self.config['ntags']])
            E_t_E = tf.matmul(E_out_t, E_out)
            E_v = tf.matmul(v, E_out)
            E_v_E_t_E = tf.matmul(E_v, E_t_E)
            self.logits = E_v + b_out
            self.mod_logits = E_v_E_t_E + b_out
            # compute softmax
            self.pred = tf.nn.softmax(self.logits, name='pred')

    def _add_loss(self):

        with tf.variable_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.config['ntags'])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.logits,
                                                    labels=targets_1hot)
            mod_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.mod_logits,
                                                    labels=targets_1hot)
            self.reg_loss = tf.reduce_mean(cross_entropy)

            mod_loss = tf.reduce_mean(mod_cross_entropy)
            self.mod_loss = (self.reg_loss + mod_loss)/2

            self.loss = self.mod_loss if self.config['comb_loss'] else self.reg_loss

    def _add_train_op(self):
        self.optimizer = self.optimizer_fn(self.lr).minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        with tf.Graph().as_default() as g:
            with tf.variable_scope(self.config['scope_name']):
                self.lr = tf.Variable(float(self.config['lr']), trainable=False,
                                        dtype=self.dtype, name='learning_rate')
                self.global_step =  tf.Variable(0, trainable=False,
                                                dtype=tf.int32, name='g_step')
                self._add_placeholders()
                self._add_embeddings()
                if self.config['use_c_embed']:
                    self._add_char_lstm()
                else:
                    self._add_char_bridge()
                self._add_word_bidi_lstm()
                self._add_tag_lstm_layer()
                if self.config['attn']:
                    self._add_attention()
                else:
                    self.proj_in = self.decode_out
                self._add_projection()
                self._add_loss()
                if (self.config['mode'] == 'train'):
                    self._add_train_op()
        return g

        """"TRAIN Part """
    def pos_step(self, bv):
        input_feed = {}
        for op in self.pos_g.get_operations():
            #get only name wihtout the scope
            op_key = op.name.split('/')[-1]
            #if operation is placeholder
            #get the tensor from graph and assign the input batch vector
            if 'placeholders' in op.name:
                tf_key = self.pos_g.get_tensor_by_name(op.name+':0')
                bv_key = op_key.split('-')
                input_feed[tf_key] = bv[bv_key[0]][bv_key[-1]]
            #if operation is the prediction get the tensor from grapg
            if op_key == 'pos_pred':
                out = self.pos_g.get_tensor_by_name(op.name+':0')
        return self.pos_sess.run(out, input_feed)

    def step(self, bv, dev=False):
        """ Training step, returns the loss"""
        input_feed = {
            self.w_in: bv['word']['in'],
            self.word_len: bv['word']['len'],
            self.char_in : bv['char']['in'],
            self.char_len : bv['char']['len'],
            self.pos_in : bv['pos']['in'],
            self.t_in: bv['tag']['in'],
            self.tag_len: bv['tag']['len'],
            self.targets: bv['tag']['out']}

        if self.config['use_pretrained_pos']:
            input_feed[self.pos_in] = self.pos_step(bv)
        output_feed = [self.loss, self.optimizer] if not dev else [self.loss]
        return self.sess.run(output_feed, input_feed)[0]

    def train_epoch(self, batcher, dev=False):
        step_time, loss = 0.0, 0.0
        current_step = self.sess.run(self.global_step) if not dev else 0
        steps_per_ckpt = self.config['steps_per_ckpt'] if not dev else 1
        for bv in batcher.get_permute_batch():
            start_time = time.time()
            step_loss = self.step(batcher.process(bv), dev)
            current_step += 1
            step_time += (time.time() - start_time) / steps_per_ckpt
            loss += step_loss / steps_per_ckpt
            ret_loss = loss
            if  current_step % steps_per_ckpt == 0:
                if not dev:
                    self.save()
                perplex = math.exp(loss) if loss < 300 else float('inf')
                print ("[[train_epoch]] step %d learning rate %f step-time %.3f"
                           " perplexity %.6f (loss %.6f)" %
                           (current_step, self.sess.run(self.lr),
                           step_time, perplex, loss))
                ret_loss = loss
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()
        return ret_loss

    def train(self, batcher, dataset):
        dev_loss = np.inf
        while dev_loss > self.config['th_loss']:
            batcher.use_data(dataset['train'])
            for epoch_id in range(0, self.num_epochs):
                self.train_epoch(batcher)
            batcher.use_data(dataset['dev'])
            for epoch_id in range(0, self.num_epochs):
                dev_loss = self.train_epoch(batcher, True)

        """"Decode Part """

    def encode_top_state(self, enc_bv):
        """Return the top states from encoder for decoder."""
        input_feed = {self.w_in: enc_bv['word']['in'],
                        self.word_len: enc_bv['word']['len'],
                        self.char_in : enc_bv['char']['in'],
                        self.char_len : enc_bv['char']['len'],
                        self.pos_in: enc_bv['pos']['in']}
        if self.config['use_pretrained_pos']:
            input_feed[self.pos_in] = self.pos_step(enc_bv)
        output_feed = self.encode_state
        return self.sess.run(output_feed, input_feed)

    def decode_topk(self, latest_tokens, dec_init_states, enc_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.tag_init : dec_init_states,
            self.t_in: np.array(latest_tokens),
            # self.encode_state : enc_state,
            self.tag_len: np.ones(1, np.int32)}
        output_feed = [self.decode_state, self.pred]
        states, probs = self.sess.run(output_feed, input_feed)
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        import pdb; pdb.set_trace()
        return topk_ids, topk_probs, states

    def decode_bs(self, vocab, bv, t_op):

        bs = BeamSearch(self.config['beam_size'],
                        vocab['tags'].token_to_id('GO'),
                        vocab['tags'].token_to_id('EOS'),
                        self.config['dec_timesteps'])

        bv_cp = copy.copy(bv)

        beams = bs.beam_search(self.encode_top_state, self.decode_topk, bv_cp)
        tags = t_op.combine_fn(vocab['tags'].to_tokens(beams['tokens']))
        tag_score_pairs = map(lambda x, y: zip(x, y), tags, beams['scores'])
        return tag_score_pairs

    def decode_batch(self, beam_pair, word_tokens):

        ngoals = self.config['num_goals']
        t_out = self.config['time_out']
        no_val = self.config['no_val_gap']
        decode_trees = []
        nsentences = len(word_tokens)
        for i, (beam_tag, sent) in enumerate(zip(beam_pair, word_tokens)):
            print ("[[decode_batch]] Staring astar search for sentence %d/%d [tag length %d]"
                        %(i+1, nsentences, len(beam_tag)))

            if all(beam_tag):
                trees = solve_tree_search(beam_tag, sent, no_val, ngoals, t_out)
            else:
                trees = []
            decode_trees.append(trees)
        return decode_trees

    def decode(self, vocab, batcher, t_op):

        decoded_trees = []
        for bv in batcher.get_batch():
            bv = batcher.process(bv)
            words_id = batcher.remove_delim_len(bv['word'])
            words_token = vocab['words'].to_tokens(words_id)
            tag_score_pairs = batcher.restore(self.decode_bs(vocab, bv, t_op))

            decoded_trees.extend(self.decode_batch(tag_score_pairs,words_token))

        return decoded_trees
