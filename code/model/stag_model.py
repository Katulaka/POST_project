from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .basic_model import BasicModel
from beam.search import BeamSearch
from astar.search import solve_tree_search


class STAGModel(BasicModel):

    def __init__ (self, config):
        BasicModel.__init__(self, config)


    def _add_placeholders(self):
        with tf.variable_scope('placeholders'):
            """Inputs to be fed to the graph."""
            #shape = (batch_size, max length of sentence, max lenght of word)
            self.char_in = tf.placeholder(tf.int32, [None, None], 'char-in')
            self.char_len = tf.placeholder(tf.int32, [None], 'char-seq-len')
            #shape = (batch_size, max length of sentence)
            self.w_in = tf.placeholder(tf.int32, [None, None], 'word-input')
            self.w_t_in = tf.placeholder(tf.string, [None, None], 'word-token-input')
            self.word_t_len = tf.placeholder(tf.int32, [None], 'word-tok-seq-len')
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
            #dropout rate
            self.is_train = tf.placeholder(tf.bool, shape=(), name='is-train')


    def _add_elmo(self):
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        word_elmo = elmo(inputs={"tokens": self.w_t_in,
                                "sequence_len": self.word_t_len
                                },
                                signature="tokens",
                                as_dict=True)["elmo"]
        word_elmo_pad = tf.pad(word_elmo, [[0,0],[1,1],[0,0]], "CONSTANT")
        word_elmo_t = tf.layers.dense(word_elmo_pad, self.elmo_dim)
        self.word_embed_elmo = tf.contrib.layers.layer_norm(word_elmo_t)

    def _add_embeddings(self):
        """ Look up embeddings for inputs. """
        with tf.variable_scope('embedding', initializer=self.initializer, dtype=self.dtype):

            ch_mat_shape = [self.nchars, self.dim_char]
            ch_embed_mat = tf.get_variable('ch-embed-mat', shape=ch_mat_shape)
            char_embed = tf.nn.embedding_lookup(ch_embed_mat, self.char_in,
                                                    name='char-embed')
            self.char_embed = tf.layers.dropout(char_embed, self.drop_rate,
                                                    training = self.is_train,
                                                    name='char-embed-dropout')

            w_mat_shape = [self.nwords, self.dim_word]
            w_embed_mat = tf.get_variable('word-embed-mat', shape=w_mat_shape)
            word_embed = tf.nn.embedding_lookup(w_embed_mat, self.w_in,
                                                    name='word-embed')
            self.word_embed = tf.layers.dropout(word_embed, self.drop_rate,
                                                    training = self.is_train,
                                                    name='word-embed-dropout')

            t_mat_shape = [self.nlabels, self.dim_label]
            t_embed_mat = tf.get_variable('tag-embed-mat', shape=t_mat_shape)
            tag_embed = tf.nn.embedding_lookup(t_embed_mat, self.t_in,
                                                    name='tag-embed')
            self.tag_embed = tf.layers.dropout(tag_embed, self.drop_rate,
                                                    training = self.is_train,
                                                    name='tag-embed-dropout')

            pos_mat_shape = [self.ntags, self.dim_tag]
            pos_embed_mat = tf.get_variable('pos-embed-mat', shape=pos_mat_shape)
            pos_embed = tf.nn.embedding_lookup(pos_embed_mat, self.pos_in,
                                                    name='pos-embed')
            self.pos_embed = tf.layers.dropout(pos_embed, self.drop_rate,
                                            training = self.is_train,
                                            name='pos-embed-dropout')

    def _add_char_lstm(self):
        with tf.variable_scope('char-LSTM-Layer', initializer=self.initializer):
            char_cell = self._single_cell(self.h_char,
                                            1. - self.drop_rate,
                                            self.is_train)

            _, ch_state = tf.nn.dynamic_rnn(char_cell,
                                            self.char_embed,
                                            sequence_length=self.char_len,
                                            dtype=self.dtype,
                                            scope='char-lstm')

            # char_out_shape = [self.batch_size, -1, self.h_char]
            char_out_shape = [tf.shape(self.word_embed)[0], -1, self.h_char]
            char_out_reshape = tf.reshape(ch_state[1], char_out_shape)

            self.word_embed_ch_lstm = tf.concat([self.word_embed, char_out_reshape],
                                        -1, 'mod_word_embed')

    def _add_word_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.variable_scope('word-bidirectional-LSTM-Layer'):
            # Forward and Backward direction cell
            word_cell_fw = self._multi_cell(self.h_word,
                                            tf.constant(self.kp_bidi),
                                            self.is_train,
                                            self.n_layers,
                                            self.is_stack)

            word_cell_bw = self._multi_cell(self.h_word,
                                            tf.constant(self.kp_bidi),
                                            self.is_train,
                                            self.n_layers,
                                            self.is_stack)

            w_bidi_in = tf.concat([self.word_embed_f, self.pos_embed], -1,
                                        name='word-bidi-in')

            # Get lstm cell output
            if self.is_stack:
                w_bidi_out_c, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                word_cell_fw,
                                                word_cell_bw,
                                                w_bidi_in,
                                                dtype=self.dtype)
            else:
                w_bidi_out , _ = tf.nn.bidirectional_dynamic_rnn(
                                    tf.contrib.rnn.MultiRNNCell(word_cell_fw),
                                    tf.contrib.rnn.MultiRNNCell(word_cell_bw),
                                    w_bidi_in,
                                    sequence_length=self.word_len,
                                    dtype=self.dtype)
                w_bidi_out_c = tf.concat(w_bidi_out , -1, name='word-bidi-out')

            self.encode_state = tf.concat([w_bidi_in, w_bidi_out_c], -1)
            hw_p = self.n_layers if self.is_stack else 1
            self.c_dim += self.dim_tag + 2**hw_p*self.h_word

    def _add_tag_lstm_layer(self):
        """Generate sequences of tags"""
        with tf.variable_scope('tag-LSTM-Layer'):
            dec_init_state = tf.reshape(self.encode_state, [-1, self.c_dim])

            self.tag_init = tf.contrib.rnn.LSTMStateTuple(dec_init_state,
                                        tf.zeros_like(dec_init_state))

            tag_cell = self._single_cell(self.c_dim,
                                         1. - self.drop_rate,
                                          self.is_train)

            self.decode_out, self.decode_state = tf.nn.dynamic_rnn(tag_cell,
                                                self.tag_embed,
                                                initial_state=self.tag_init,
                                                sequence_length=self.tag_len,
                                                dtype=self.dtype)

    def _add_attention(self):
        with tf.variable_scope('Attention', initializer=self.initializer):
            self.dec_in_dim = self.h_word * 2
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

            proj_in = tf.layers.dense(self.proj_in, self.h_label,
                                        use_bias=False,
                                        activation=self.activation_fn)

            mask_t = tf.sequence_mask(self.tag_len, dtype=tf.int32)
            v = tf.dynamic_partition(proj_in, mask_t, 2)

            self.logits = tf.layers.dense(v[1], self.nlabels, use_bias=True)
            # compute softmax
            self.pred = tf.nn.softmax(self.logits, name='pred')

    def _add_loss(self):

        with tf.variable_scope("loss"):
            targets_1hot = tf.one_hot(self.targets, self.nlabels)

            self.loss = tf.losses.softmax_cross_entropy(
                                logits=self.logits,
                                onehot_labels=targets_1hot,
                                reduction=tf.losses.Reduction.MEAN)

    def _add_train_op(self):
        self.lr = tf.Variable(float(self.learn_rate), trainable=False,
                                dtype=self.dtype, name='learning_rate')
        self.global_step =  tf.Variable(0, trainable=False,
                                        dtype=tf.int32, name='g_step')
        self.epoch =  tf.Variable(0, trainable=False,
                                        dtype=tf.int32, name='epoch')

        if self.grad_clip:
            gradients, variables = zip(*self.optimizer_fn(self.lr).compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_norm)
            self.optimizer = self.optimizer_fn(self.lr).apply_gradients(zip(gradients, variables),
                global_step=self.global_step)
        else:
            self.optimizer = self.optimizer_fn(learning_rate=self.lr).minimize(
                                        self.loss, global_step=self.global_step)

    def build_graph(self):
        with tf.Graph().as_default() as g:
            with tf.device('/gpu:%d' %self.gpu_n):
                with tf.variable_scope(self.scope_name):
                    self._add_placeholders()
                    self._add_embeddings()
                    if self.is_add_elmo:
                        self._add_elmo()
                        self.word_embed_f = self.word_embed_elmo
                        self.c_dim = self.elmo_dim
                    elif not self.no_c_embed:
                        self._add_char_lstm()
                        self.word_embed_f = self.word_embed_ch_lstm
                        self.c_dim = self.h_char + self.dim_word
                    else:
                        self.word_embed_f = self.word_embed
                        self.c_dim = self.dim_word
                    self._add_word_bidi_lstm()
                    self._add_tag_lstm_layer()
                    if self.no_attn:
                        self.proj_in = self.decode_out
                    else:
                        self._add_attention()
                    self._add_projection()
                    if (self.mode != 'test'):
                        self._add_loss()
                    if (self.mode == 'train'):
                        self._add_train_op()
            return g

        """"TRAIN Part """
    def step(self, bv, output_feed, is_train=False):
        """ Training step, returns the loss"""
        input_feed = {
            self.w_in: bv['word']['in'],
            self.word_len: bv['word']['len'],
            self.w_t_in: bv['word_t']['in'],
            self.word_t_len: bv['word_t']['len'],
            self.char_in : bv['char']['in'],
            self.char_len : bv['char']['len'],
            self.pos_in : bv['pos']['in'],
            self.t_in: bv['tag']['in'],
            self.tag_len: bv['tag']['len'],
            self.targets: bv['tag']['out'],
            self.is_train : is_train}
        return self.sess.run(output_feed, input_feed)

    def train(self, batcher):

        # Create a summary to monitor loss tensor
        t_loss = tf.summary.scalar("loss", self.loss)
        current_epoch = self.sess.run(self.epoch)
        for epoch_id in range(self.num_epochs):
            current_step = self.sess.run(self.global_step)
            loss = []
            for bv in batcher.get_batch(mode='train', shuffle=True):
                input_feed = batcher.process(bv)
                output_feed = [t_loss, self.optimizer]
                print('running train')
                summary_loss, _ = self.step(input_feed, output_feed, True)
                self.sw.add_summary(summary_loss, current_step)
                current_step += 1
                if self.steps_per_ckpt!=0:
                    if current_step % self.steps_per_ckpt == 0:
                        self.save()
                        sys.stdout.flush()

            t_dev_loss = tf.Summary()
            dev_losses = []
            for bv in batcher.get_batch(mode='dev'):
                dev_loss = [self.step(batcher.process(bv), self.loss, False)]
                dev_losses.append(dev_loss)
            mean_dev_loss = np.mean(dev_losses)
            t_dev_loss.value.add(tag="loss_epoch", simple_value=mean_dev_loss)
            self.sw.add_summary(t_dev_loss, current_epoch)
            epoch_inc = tf.assign_add(self.epoch, 1)
            current_epoch = self.sess.run(epoch_inc)

            if self.steps_per_ckpt==0:
                self.save()
                sys.stdout.flush()
        self.sw.close()

        """"Decode Part """

    def encode_top_state(self, enc_bv):
        """Return the top states from encoder for decoder."""
        input_feed = {self.w_in: enc_bv['word']['in'],
                        self.word_len: enc_bv['word']['len'],
                        self.w_t_in: enc_bv['word_t']['in'],
                        self.word_t_len: enc_bv['word_t']['len'],
                        self.char_in : enc_bv['char']['in'],
                        self.char_len : enc_bv['char']['len'],
                        self.pos_in: enc_bv['pos']['in'],
                        self.is_train : False}
        output_feed = self.encode_state
        return self.sess.run(output_feed, input_feed)

    def decode_topk(self, latest_tokens, dec_init_states, enc_state, k):
        """Return the topK results and new decoder states."""
        input_feed = {
            self.tag_init : dec_init_states,
            self.t_in: np.array(latest_tokens),
            self.encode_state : enc_state,
            self.tag_len: np.ones(len(latest_tokens), np.int32),
            self.is_train : False}
        output_feed = [self.decode_state, self.pred]
        states, probs = self.sess.run(output_feed, input_feed)
        topk_ids = np.array([np.argsort(np.squeeze(p))[-k:] for p in probs])
        topk_probs = np.array([p[k_id] for p,k_id in zip(probs,topk_ids)])
        return topk_ids, topk_probs, states

    def decode(self, mode, batcher):
        import pickle as dill
        import os

        if not os.path.exists(self.decode_dir):
            try:
                os.makedirs(self.decode_dir)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        decode_trees = []

        if os.path.exists(self.decode_trees_file):
            with open(self.decode_trees_file, 'rb') as f:
                while True:
                    try:
                        decode_trees.append(dill.load(f))
                    except EOFError:
                        break
        s_idx = len(decode_trees)

        # bs = BeamSearch(self.nlabels,
        bs = BeamSearch(self.beam_size,
                        batcher._vocab['tags'].token_to_id('GO'),
                        batcher._vocab['tags'].token_to_id('EOS'),
                        self.beam_timesteps)

        for bv in batcher.get_batch(mode=mode):

            words_token = batcher._vocab['words'].to_tokens(bv['words'])

            beams = bs.beam_search(self.encode_top_state,
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
                                            self.num_goals,
                                            self.time_out,
                                            self.time_th,
                                            self.cost_coeff_rate)
                else:
                    trees = []

                decode_trees.append(trees)

            with open(self.decode_trees_file, 'ab') as f:
                dill.dump(trees, f)

        return decode_trees

    # def _decode(self, batcher):
    #     import pickle as dill
    #     import os
    #
    #     decode_trees = []
    #     astar_ranks = []
    #     s_idx = 0
    #     if os.path.exists(self.config['decode_trees_file']):
    #         with open(self.config['decode_trees_file'], 'rb') as f:
    #             while True:
    #                 try:
    #                     decode_trees.append(dill.load(f))
    #                 except EOFError:
    #                     break
    #         s_idx = len(decode_trees)
    #
    #     if os.path.exists(self.config['astar_ranks_file']):
    #         with open(self.config['astar_ranks_file'], 'rb') as f:
    #             while True:
    #                 try:
    #                     astar_ranks.append(dill.load(f))
    #                 except EOFError:
    #                     break
    #
    #     with open(self.config['beams_file'], 'rb') as fout:
    #         all_beams = dill.load(fout)
    #
    #     len_beam = len(all_beams)
    #
    #     cnt_idx = s_idx
    #     words = batcher._ds['test']['words']
    #     for bv_w, beams in zip(words[s_idx:], all_beams[s_idx:]):
    #         print('decode %d/%d' %(cnt_idx,len_beam))
    #         cnt_idx += 1
    #         words_token = batcher._vocab['words'].to_tokens(bv_w)
    #         tags = batcher._vocab['tags'].to_tokens(beams['tokens'])
    #         tags = batcher._t_op.combine_fn(batcher._t_op.modify_fn(tags))
    #         tag_score_mat = map(lambda x, y: zip(x, y), tags, beams['scores'])
    #         batcher._seq_len = [len(words_token)]
    #         tag_score_mat = batcher.restore(tag_score_mat)
    #         if all(tag_score_mat):
    #             trees, astar_rank = solve_tree_search(tag_score_mat[0],
    #                                     words_token,
    #                                     batcher._t_op.no_val_gap,
    #                                     self.config['num_goals'],
    #                                     self.config['time_out'])
    #         else:
    #             trees = []
    #             astar_rank = []
    #         decode_trees.append(trees)
    #         astar_ranks.append(astar_rank)
    #         with open(self.config['decode_trees_file'], 'ab') as fin:
    #             dill.dump(trees, fin)
    #         with open(self.config['astar_ranks_file'], 'ab') as fin:
    #             dill.dump(astar_rank, fin)
    #
    #     return decode_trees, astar_ranks
    #
    # def stats(self, mode, batcher):
    #     import pickle as dill
    #     import os
    #
    #     beams_rank = []
    #     beams = []
    #
    #     bs = BeamSearch(self.config['ntags'],
    #                     batcher._vocab['tags'].token_to_id('GO'),
    #                     batcher._vocab['tags'].token_to_id('EOS'),
    #                     self.config['beam_timesteps'])
    #
    #     if self.config['use_subset']:
    #         subset_idx = batcher.get_subset_idx(self.config['subset_file'], 0.1, mode)
    #     else:
    #         subset_idx = None
    #
    #     if os.path.exists(self.config['beams_file']):
    #         with open(self.config['beams_file'], 'rb') as f:
    #             while True:
    #                 try:
    #                     beams.append(dill.load(f))
    #                 except EOFError:
    #                     break
    #
    #     if os.path.exists(self.config['beams_rank_file']):
    #         with open(self.config['beams_rank_file'], 'rb') as f:
    #             while True:
    #                 try:
    #                     beams_rank.append(dill.load(f))
    #                 except EOFError:
    #                     break
    #
    #     s_idx = len(beams_rank)
    #
    #     for bv in batcher.get_batch(mode=mode, subset_idx=subset_idx)[s_idx:]:
    #
    #         beams.append(bs.beam_search(self.encode_top_state,
    #                                     self.decode_topk,
    #                                     batcher.process(bv))
    #
    #         beams_rank.append([b.index(t) if t in b else -1 for b,t in zip(beams[-1]['tokens'],bv['tags'][-1])])
    #
    #         with open(self.config['beams_file'], 'ab') as f:
    #             dill.dump(beams[-1], f)
    #
    #         with open(self.config['beams_rank_file'], 'ab') as f:
    #             dill.dump(beams_rank[-1], f)
    #
    #     return beams, tags, beams_rank
