from __future__ import division
from __future__ import print_function

import math
import sys
import time
import numpy as np
import tensorflow as tf
from .basic_model import BasicModel
from beam.search import BeamSearch

from astar.search import solve_tree_search


class STAGModel(BasicModel):

    def __init__ (self, config):
        BasicModel.__init__(self, config)
        self.initializer = tf.contrib.layers.xavier_initializer()
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
            #dropout rate 0-char 1-word 2-tag
            self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep-prob")


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

            # char_out = tf.layers.dense(ch_state[1], self.config['dim_word'],
            #                             use_bias=False)
            # char_out_reshape =  tf.reshape(char_out, tf.shape(self.word_embed))

            ch_state_drop = tf.nn.dropout(ch_state[1], self.keep_prob,
                                    name='char-lstm-dropout')

            we_shape = tf.shape(self.word_embed)
            co_shape = [we_shape[0], we_shape[1], self.config['hidden_char']]
            # char_out_reshape = tf.reshape(ch_state[1], co_shape)
            char_out_reshape = tf.reshape(ch_state_drop, co_shape)

            self.word_embed_f = tf.concat([self.word_embed, char_out_reshape],
                                        -1, 'mod_word_embed')
            self.c_dim = self.config['hidden_char'] + self.config['dim_word']

    def _add_char_bridge(self):
        with tf.variable_scope('char-Bridge'):
            self.word_embed_f = self.word_embed
            self.c_dim = self.config['dim_word']

    def _add_word_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.variable_scope('word-bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            word_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_word'])
            word_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_word'])
            # Get lstm cell output
            w_bidi_in = tf.concat([self.word_embed_f, self.pos_embed], -1,
                                        name='word-bidi-in')
            w_bidi_out , _ = tf.nn.bidirectional_dynamic_rnn(word_cell_fw,
                                                word_cell_bw,
                                                w_bidi_in,
                                                sequence_length=self.word_len,
                                                dtype=self.dtype)

            w_bidi_out_c = tf.concat(w_bidi_out , -1, name='word-bidi-out')
            w_bidi_out_drop = tf.nn.dropout(w_bidi_out_c,
                                            self.keep_prob,
                                            name='word-lstm-dropout')

            self.w_bidi_in_out = tf.concat([w_bidi_in, w_bidi_out_drop], -1)

    def _no_affine_trans(self):
        with tf.variable_scope('no-affine'):
            self.encode_state = self.w_bidi_in_out
            self.c_dim += self.config['dim_pos'] + 2*self.config['hidden_word']

    def _affine_trans(self):
        with tf.variable_scope('affine'):
            self.encode_state = tf.layers.dense(self.w_bidi_in_out,
                                                self.config['hidden_tag'],
                                                use_bias=True)
            self.c_dim = self.config['hidden_tag']

    def _add_tag_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.variable_scope('tag-LSTM-Layer'):
            dec_init_state = tf.reshape(self.encode_state, [-1, self.c_dim])

            self.tag_init = tf.contrib.rnn.LSTMStateTuple(dec_init_state,
                                        tf.zeros_like(dec_init_state))

            tag_cell = tf.contrib.rnn.BasicLSTMCell(self.c_dim)

            decode_out, self.decode_state = tf.nn.dynamic_rnn(tag_cell,
                                                self.tag_embed,
                                                initial_state=self.tag_init,
                                                sequence_length=self.tag_len,
                                                dtype=self.dtype)

            self.decode_out = tf.nn.dropout(decode_out, self.keep_prob,
                                            name='tag-lstm-dropout')

    def _add_attention(self):
        with tf.variable_scope('Attention', initializer=self.initializer):
            self.dec_in_dim = self.config['hidden_word'] * 2
            do_shape = tf.shape(self.decode_out)
            es_shape = tf.shape(self.encode_state)
            ak_shape = [es_shape[0], -1, self.dec_in_dim]

            self.k = atten_k = tf.reshape(tf.layers.dense(self.decode_out,
                                                self.dec_in_dim,
                                                use_bias=False), ak_shape)

            self.q = atten_q = tf.layers.dense(self.encode_state,
                                                self.dec_in_dim,
                                                activation=self.activation_fn,
                                                use_bias=False)

            self.a = alpha = tf.nn.softmax(tf.einsum('aij,akj->aik',
                                                    atten_k, atten_q))
            self.c = context = tf.reshape(tf.einsum('aij,ajk->aik', alpha,
                                                self.encode_state), do_shape)
            self.proj_in = tf.concat([self.decode_out, context], -1)


    def _add_projection(self):

        with tf.variable_scope('predictions', initializer=self.initializer, dtype=self.dtype):

            proj_in = tf.layers.dense(self.proj_in, self.config['hidden_tag'],
                                        use_bias=False,
                                        activation=self.activation_fn)

            mask_t = tf.sequence_mask(self.tag_len, dtype=tf.int32)
            v = tf.dynamic_partition(proj_in, mask_t, 2)

            self.logits = tf.layers.dense(v[1], self.config['ntags'], use_bias=True)
            # compute softmax
            self.pred = tf.nn.softmax(self.logits, name='pred')

    def _add_loss(self):

        with tf.variable_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.config['ntags'])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.logits,
                                                    labels=targets_1hot)
            self.loss = tf.reduce_mean(cross_entropy)


    def _add_train_op(self):
        self.lr = tf.Variable(float(self.config['lr']), trainable=False,
                                dtype=self.dtype, name='learning_rate')
        self.global_step =  tf.Variable(0, trainable=False,
                                        dtype=tf.int32, name='g_step')
        self.epoch =  tf.Variable(0, trainable=False,
                                        dtype=tf.int32, name='epoch')

        if self.config['grad_clip']:
            gradients, variables = zip(*self.optimizer_fn(self.lr).compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config['grad_norm'])
            self.optimizer = self.optimizer_fn(self.lr).apply_gradients(zip(gradients, variables),
                global_step=self.global_step)
        else:
            self.optimizer = self.optimizer_fn(learning_rate=self.lr).minimize(
                                        self.loss, global_step=self.global_step)

    def build_graph(self):
        with tf.device('/gpu:0'):
            with tf.Graph().as_default() as g:
                with tf.variable_scope(self.config['scope_name']):
                    self._add_placeholders()
                    self._add_embeddings()
                    if self.config['no_c_embed']:
                        self._add_char_bridge()
                    else:
                        self._add_char_lstm()
                    self._add_word_bidi_lstm()
                    if self.config['affine']:
                        self._affine_trans()
                    else:
                        self._no_affine_trans()
                    self._add_tag_lstm_layer()
                    if self.config['no_attn']:
                        self.proj_in = self.decode_out
                    else:
                        self._add_attention()
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

    def step(self, bv, output_feed, dev=False):
        """ Training step, returns the loss"""
        input_feed = {
            self.w_in: bv['word']['in'],
            self.word_len: bv['word']['len'],
            self.char_in : bv['char']['in'],
            self.char_len : bv['char']['len'],
            self.pos_in : bv['pos']['in'],
            self.t_in: bv['tag']['in'],
            self.tag_len: bv['tag']['len'],
            self.targets: bv['tag']['out'],
            self.keep_prob: self.config['keep_prob']}

        if self.config['use_pretrained_pos']:
            input_feed[self.pos_in] = self.pos_step(bv)
        if dev:
            input_feed[self.keep_prob] = 1.0
        return self.sess.run(output_feed, input_feed)

    def train(self, batcher):

        if self.config['use_subset']:
            subset_idx = batcher.get_subset_idx(self.config['subset_file'], 0.1, 'train')
            subset_idx_dev = batcher.get_subset_idx(self.config['subset_file_dev'], 0.1, 'dev')
        else:
            subset_idx = None
            subset_idx_dev = None
        summary_writer = tf.summary.FileWriter(self.result_dir+'/graphs', self.graph)
        summary_writer_dev = tf.summary.FileWriter(self.result_dir+'/graphs/dev', self.graph)
        # Create a summary to monitor loss tensor
        t_loss = tf.summary.scalar("loss", self.loss)

        current_epoch = self.sess.run(self.epoch)
        for epoch_id in range(self.num_epochs):
            current_step = self.sess.run(self.global_step)
            loss = []
            for bv in batcher.get_batch(mode='train', permute=True, subset_idx=subset_idx):
                input_feed = batcher.process(bv)
                output_feed = [self.loss, t_loss, self.optimizer]
                step_loss, summary_loss, _ = self.step(input_feed, output_feed)
                loss.append(step_loss)
                summary_writer.add_summary(summary_loss, current_step)
                current_step += 1
                # if current_step % self.config['steps_per_ckpt'] == 0:
                #     self.save()
                #     sys.stdout.flush()

            summary = tf.Summary()
            summary.value.add(tag="loss_epoch", simple_value=np.mean(loss))
            summary_writer.add_summary(summary, current_epoch)

            mean_loss = np.mean([self.step(batcher.process(bv), self.loss, dev=True)
            for bv in batcher.get_batch(mode='dev', subset_idx=subset_idx_dev)])
            summary = tf.Summary()
            summary.value.add(tag="loss_epoch", simple_value=mean_loss)
            summary_writer_dev.add_summary(summary, current_epoch)

            epoch_inc = tf.assign_add(self.epoch, 1)
            current_epoch = self.sess.run(epoch_inc)
            self.save()
            sys.stdout.flush()

        summary_writer.close()
        summary_writer_dev.close()


        """"Decode Part """

    def encode_top_state(self, enc_bv):
        """Return the top states from encoder for decoder."""
        input_feed = {self.w_in: enc_bv['word']['in'],
                        self.word_len: enc_bv['word']['len'],
                        self.char_in : enc_bv['char']['in'],
                        self.char_len : enc_bv['char']['len'],
                        self.pos_in: enc_bv['pos']['in'],
                        self.keep_prob: 1.0}
        if self.config['use_pretrained_pos']:
            input_feed[self.pos_in] = self.pos_step(enc_bv)
        output_feed = self.encode_state
        return self.sess.run(output_feed, input_feed)

    def decode_topk(self, latest_tokens, dec_init_states, enc_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.tag_init : dec_init_states,
            self.t_in: np.array(latest_tokens),
            self.encode_state : enc_state,
            self.tag_len: np.ones(1, np.int32),
            self.keep_prob: 1.0}
        output_feed = [self.decode_state, self.pred]
        states, probs = self.sess.run(output_feed, input_feed)
        topk_ids = np.argsort(np.squeeze(probs))[-k:]
        topk_probs = np.squeeze(probs)[topk_ids]
        return topk_ids, topk_probs, states

    # def decode_topk_b(self, latest_tokens, dec_init_states, enc_state, batch_size, k):
    #     """Return the topK results and new decoder states."""
    #     input_feed = {
    #         self.tag_init : dec_init_states,
    #         self.t_in: latest_tokens,
    #         self.encode_state : enc_state,
    #         self.tag_len: np.ones(batch_size, np.int32)}
    #     output_feed = [self.decode_state, self.pred]
    #     states, probs = self.sess.run(output_feed, input_feed)
    #     topk_ids = []
    #     topk_probs = []
    #     for b in range(batch_size):
    #         topk_ids.append(np.argsort(probs[b])[-k:])
    #         topk_probs.append(probs[b][topk_ids[b]])
    #     return topk_ids, topk_probs, states

    def decode(self, batcher):

        decode_trees = []
        bs = BeamSearch(self.config['beam_size'],
                        batcher._vocab['tags'].token_to_id('GO'),
                        batcher._vocab['tags'].token_to_id('EOS'),
                        self.config['beam_timesteps'])

        if self.config['use_subset']:
            subset_idx = batcher.get_subset_idx(self.config['subset_file'], 0.1)
        else:
            subset_idx = None
        for bv in batcher.get_batch(subset_idx=subset_idx):

            words_token = batcher._vocab['words'].to_tokens(bv['words'][0])

            beams, _ = bs.beam_search(self.encode_top_state,
                                        self.decode_topk,
                                        batcher.process(bv))

            tags = batcher._vocab['tags'].to_tokens(beams['tokens'])
            tags = batcher._t_op.combine_fn(batcher._t_op.modify_fn(tags))
            tag_score_mat = map(lambda x, y: zip(x, y), tags, beams['scores'])
            tag_score_mat = batcher.restore(tag_score_mat)
            for ts_entry, w_entry in zip(tag_score_mat, words_token):
                if all(tag_score_mat):
                    trees, _ = solve_tree_search(ts_entry, w_entry,
                                            batcher._t_op.no_val_gap,
                                            self.config['num_goals'],
                                            self.config['time_out'])
                else:
                    trees = []

                decode_trees.append(trees)
        return decode_trees

    def _decode(self, batcher):
        import pickle as dill
        import os

        decode_trees = []
        astar_ranks = []
        s_idx = 0
        if os.path.exists('decode_trees.p'):
            with open('decode_trees.p', 'rb') as f:
                while True:
                    try:
                        decode_trees.append(dill.load(f))
                    except EOFError:
                        break
            s_idx = len(decode_trees)

        if os.path.exists('astar_ranks.p'):
            with open('astar_ranks.p', 'rb') as f:
                while True:
                    try:
                        astar_ranks.append(dill.load(f))
                    except EOFError:
                        break

        with open('beams.p', 'rb') as fout:
            all_beams = dill.load(fout)

        for bv_w, beams in zip(batcher._ds['test']['words'][s_idx:], all_beams[s_idx:]):
            words_token = batcher._vocab['words'].to_tokens(bv_w)

            tags = batcher._vocab['tags'].to_tokens(beams['tokens'])
            tags = batcher._t_op.combine_fn(batcher._t_op.modify_fn(tags))
            tag_score_mat = map(lambda x, y: zip(x, y), tags, beams['scores'])
            batcher._seq_len = [len(words_token)]
            tag_score_mat = batcher.restore(tag_score_mat)
            # import pdb; pdb.set_trace()
            if all(tag_score_mat):
                trees, astar_rank = solve_tree_search(tag_score_mat[0],
                                        words_token,
                                        batcher._t_op.no_val_gap,
                                        self.config['num_goals'],
                                        self.config['time_out'])
            else:
                trees = []
                astar_rank = []
            decode_trees.append(trees)
            astar_ranks.append(astar_rank)
            with open('decode_trees.p', 'ab') as fin:
                dill.dump(trees, fin)
            with open('astar_ranks.p', 'ab') as fin:
                dill.dump(astar_rank, fin)

        return decode_trees, astar_ranks

    def stats(self, mode, batcher):


        beams_rank = []
        beams = []
        tags = []
        bs = BeamSearch(self.config['beam_size'],
                        batcher._vocab['tags'].token_to_id('GO'),
                        batcher._vocab['tags'].token_to_id('EOS'),
                        self.config['beam_timesteps'])

        if self.config['use_subset']:
            subset_idx = batcher.get_subset_idx(self.config['subset_file'], 0.1, mode)
        else:
            subset_idx = None
        for bv in batcher.get_batch(mode=mode, subset_idx=subset_idx):

            beams.append(bs.beam_search(self.encode_top_state,
                                        self.decode_topk,
                                        batcher.process(bv)))
            tags.append(bv['tags'][-1])

            beam = beams[-1]['tokens']
            tag = bv['tags'][-1]

            beams_rank.append([b.index(t)+1 if t in b else -1 for b,t in zip(beam,tag)])
            # import pdb; pdb.set_trace()
            if -1 in beams_rank[-1]:
                bs = BeamSearch(79,
                                batcher._vocab['tags'].token_to_id('GO'),
                                batcher._vocab['tags'].token_to_id('EOS'),
                                self.config['beam_timesteps'])
                beams[-1] =  bs.beam_search(self.encode_top_state,
                                            self.decode_topk,
                                            batcher.process(bv))
                beam = beams[-1]['tokens']
                beams_rank[-1] = [b.index(t)+1 if t in b else -1 for b,t in zip(beam,tag)]
                bs = BeamSearch(self.config['beam_size'],
                                batcher._vocab['tags'].token_to_id('GO'),
                                batcher._vocab['tags'].token_to_id('EOS'),
                                self.config['beam_timesteps'])

        return beams, tags, beams_rank
