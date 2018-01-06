from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from basic_model import BasicModel
import time
import math
import sys

class POSModel(BasicModel):
    def __init__(self, config):
        BasicModel.__init__(self, config)

    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        with tf.variable_scope('placeholders'):
            #shape = (batch_size, max length of sentence, max lenght of word)
            self.char_in = tf.placeholder(tf.int32, [None, None], 'char-in')
            self.char_len = tf.placeholder(tf.int32, [None], 'char-seq-len')
            #shape = (batch_size, max length of sentence)
            self.w_in = tf.placeholder(tf.int32, [None, None], 'word-in')
            #shape = (batch_size)
            self.word_len = tf.placeholder(tf.int32, [None], 'word-seq-len')
            #shape = (batch_size, max length of sentence)
            self.pos_in = tf.placeholder(tf.int32, [None, None], 'pos-in')

    def _add_embeddings(self):
        """ Look up embeddings for inputs. """
        with tf.variable_scope('embedding'):

            ch_mat_shape = [self.config['nchars'], self.config['dim_char']]
            ch_embed_mat = tf.get_variable('char-embedding',
                                            shape=ch_mat_shape,
                                            dtype=self.dtype,
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.char_embed = tf.nn.embedding_lookup(ch_embed_mat,
                                                    self.char_in,
                                                    name='char-embed')

            w_mat_shape = [self.config['nwords'], self.config['dim_word']]
            w_embed_mat = tf.get_variable('word-embeddings',
                                            dtype=self.dtype,
                                            shape=w_mat_shape,
                                            initializer=self.initializer)
            self.word_embed = tf.nn.embedding_lookup(w_embed_mat,
                                                    self.w_in,
                                                    name='word-embed')

    def _add_char_lstm(self):
        with tf.variable_scope('LSTM-layer-char'):
            char_cell = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_char'])

            _, ch_state = tf.nn.dynamic_rnn(char_cell,
                                            self.char_embed,
                                            sequence_length=self.char_len,
                                            dtype=self.dtype,
                                            scope='char-lstm')

            W_char_shape = [self.config['hidden_char'], self.config['dim_word']]
            W_char = tf.get_variable('W_char',
                                        dtype=self.dtype,
                                        shape=W_char_shape,
                                        initializer=self.initializer)

            char_out = tf.einsum('aj,jk->ak', ch_state[1], W_char)
            char_out_reshape =  tf.reshape(char_out, tf.shape(self.word_embed))
            self.word_embed_f = tf.concat([self.word_embed, char_out_reshape],
                                        -1, name='mod_word_embed')
            self.dim_word_f = self.config['dim_word'] * 2

    def _add_char_bridge(self):
        with tf.variable_scope('bridge-lc-lpos'):
            self.word_embed_f = self.word_embed
            self.dim_word_f = self.config['dim_word']

    def _add_pos_bidi_lstm(self):
        """ Bidirectional LSTM """
        with tf.variable_scope('LSTM-layer-pos'):
            # Forward and Backward direction cell
            pos_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_pos'])
            pos_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_pos'])
            # Get lstm cell output
            pos_out, _ = tf.nn.bidirectional_dynamic_rnn(
                                            pos_cell_fw,
                                            pos_cell_bw,
                                            self.word_embed_f,
                                            sequence_length=self.word_len,
                                            dtype=self.dtype,
                                            scope='pos-bidi')

            self.pos_out = tf.concat(pos_out, -1, name='pos_out')

    def _add_pos_prediction(self):
        with tf.variable_scope('prediction'):
            self.pos_logits = tf.layers.dense(self.pos_out, self.config['ntags'])
            self.pos_pred = tf.argmax(self.pos_logits, 2, name='pos_pred')

    def _add_pos_loss(self):
        with tf.variable_scope('loss'):
            pos_in_1hot = tf.one_hot(self.pos_in, self.config['ntags'])
            pos_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                                    logits=self.pos_logits,
                                                    labels=pos_in_1hot)
            self.loss = tf.reduce_mean(pos_cross_entropy)

    def _add_train_op(self):
        self.optimizer = self.optimizer_fn(self.lr).minimize(self.loss, global_step=self.global_step)

    def _add_to_collection(self):
        tf.add_to_collection('c_in', self.char_in)
        tf.add_to_collection('c_len', self.char_len)
        tf.add_to_collection('w_in', self.w_in)
        tf.add_to_collection('w_len', self.word_len)
        tf.add_to_collection('pos_pred', self.pos_pred)

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
                self._add_pos_bidi_lstm()
                self._add_pos_prediction()
                self._add_pos_loss()
                if (self.config['mode'] == 'train'):
                    self._add_train_op()
            self._add_to_collection()
        return g

    def step(self, bv, dev):
        """ Training step, returns the prediction, loss"""
        input_feed = {
        self.w_in: bv['word']['in'],
        self.word_len: bv['word']['len'],
        self.char_in : bv['char']['in'],
        self.char_len : bv['char']['len'],
        self.pos_in : bv['pos']['out']}
        if not dev:
            output_feed = [self.loss, self.optimizer]
        else:
            output_feed = self.loss
        return self.sess.run(output_feed, input_feed)

    def train_epoch(self, batcher, dev=False):
        step_time, loss = 0.0, 0.0
        current_step = self.sess.run(self.global_step) if not dev else 0
        steps_per_ckpt = self.config['steps_per_ckpt'] if not dev else 1
        for bv in batcher.get_permute_batch():
            start_time = time.time()
            step_loss, _ = self.step(batcher.process(bv), dev)
            current_step += 1
            step_time += (time.time() - start_time) / steps_per_ckpt
            loss += step_loss / steps_per_ckpt
            ret_loss = loss
            if  not dev and current_step % (steps_per_ckpt*10) == 0:
                _ = self.freeze_graph(self.pos_pred.name.split(':')[0])
            if  current_step % steps_per_ckpt == 0:
                if not dev:
                    self.save()
                perplex = math.exp(loss) if loss < 300 else float('inf')
                print ("[[train_epoch:]] step %d learning rate %f step-time %.3f"
                           " perplexity %.6f (loss %.6f)" %
                           (current_step, self.sess.run(self.lr),
                           step_time, perplex, loss))
                ret_loss = loss
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()
        return ret_loss

    def train(self, batcher_train, batcher_dev):
        dev_loss = np.inf
        while dev_loss > self.config['th_loss']:
            for epoch_id in range(0, self.num_epochs):
                self.train_epoch(batcher_train)
            for epoch_id in range(0, self.num_epochs):
                dev_loss = self.train_epoch(batcher_dev, True)

    def decode(self, w_in, w_len, c_in, c_len):
        input_feed = {self.w_in: bv['word']['in'],
                    self.word_len: w_len,
                    self.char_in : c_in,
                    self.char_len : c_len}
        output_feed = self.pos_pred
        return self.sess.run(output_feed, input_feed)
