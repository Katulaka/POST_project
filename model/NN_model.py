from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy
import utils.data_preproc as dp


class NNModel(object):

    def __init__(self, batch_size, word_embedding_size, tag_embedding_size, n_hidden_fw, n_hidden_bw,
                 n_hidden_lstm, word_vocabulary_size, tag_vocabulary_size,num_steps, learning_rate,
                 learning_rate_decay_factor, max_gradient_norm, stags=True, dtype=tf.float32, scope_name='nn_model'):

        # try:
        #     LSTM = tf.nn.rnn_cell.BasicLSTMCell
        #     LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        # except:
        LSTM = tf.contrib.rnn.BasicLSTMCell
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):

            self.word_seq_lens = tf.placeholder(tf.int32, shape=[None], name='word-sequence-length')
            self.tag_seq_lens = tf.placeholder(tf.int32, shape=[None], name='tag-sequence-length')

            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)

            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

            with tf.name_scope('input'):
                self.word_inputs = tf.placeholder(tf.int32, shape=[None, None], name="word-input")
                if stags:
                    self.tag_inputs = tf.placeholder(tf.int32, shape=[None, None, None], name="tag-input") #TODO
                    self.y = tf.placeholder(tf.int32, shape=[None, None, None, None], name="y-input")
                else:
                    self.tag_inputs = tf.placeholder(tf.int32, shape=[None, None], name="tag-input")
                    self.y = tf.placeholder(tf.int32, shape=[None, None, None], name="y-input")

            # Look up embeddings for inputs.
            with tf.name_scope('embedding'):
                word_embed_matrix_init = tf.random_uniform([word_vocabulary_size, word_embedding_size], -1.0, 1.0)
                word_embed_matrix = tf.Variable(word_embed_matrix_init, name='word-embeddings')
                word_embed = tf.nn.embedding_lookup(word_embed_matrix, self.word_inputs, name='word-embed')

                tag_embed_matrix_init = tf.random_uniform([tag_vocabulary_size, tag_embedding_size], -1.0, 1.0)
                tag_embed_matrix = tf.Variable(tag_embed_matrix_init, name='tag-embeddings')
                tag_embed = tf.nn.embedding_lookup(tag_embed_matrix, self.tag_inputs, name='tag-embed')

                self.tag_embed = tag_embed

            with tf.name_scope('bidirectional-LSTM-Layer'):
                # Bidirectional LSTM
                # Forward and Backward direction cell
                lstm_fw_cell = LSTM(n_hidden_fw, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = LSTM(n_hidden_bw, forget_bias=1.0, state_is_tuple=True)

            # Get lstm cell output
                # try:
                bidi_out, bidi_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                            word_embed,
                                                                            sequence_length=self.word_seq_lens,
                                                                            dtype=dtype)

                # except Exception:  # Old TensorFlow version only returns outputs not states
                #     bidi_out = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, word_embed,
                                                            #    sequence_length=self.word_seq_lens, dtype=dtype)

            self.bidi_out = bidi_out
            self.bidi_states = bidi_states

            with tf.name_scope('LSTM-Layer'):
                # LSTM
                lstm_init = tf.concat(bidi_out, 2, name='lstm-init')
                lstm_init_reshape = tf.reshape(lstm_init, [-1, n_hidden_fw + n_hidden_bw]) #TODO

                lstm_init_reshape = LSTMStateTuple(lstm_init_reshape, tf.zeros_like(lstm_init_reshape))

                self.lstm_init_reshape = lstm_init_reshape
                self.input_shape = input_shape = tf.shape(tag_embed)
                self.tag_embedding_size = tag_embedding_size
                if stags:
                    lstm_input = tf.reshape(tag_embed, [-1, input_shape[2], tag_embedding_size])
                else:
                    lstm_input = tf.expand_dims(tf.reshape(tag_embed, [-1,  input_shape[2]]), 1)
                lstm_input.set_shape([None,None,tag_embedding_size])

                lstm_cell = LSTM(n_hidden_lstm, forget_bias=1.0, state_is_tuple=True)  # TODO

            #    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell, tag_embed,sequence_length=self.tag_seq_lens, dtype=dtype)
                # try:
                lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=lstm_init_reshape,
                                                    sequence_length=self.tag_seq_lens, dtype=dtype)

                # except Exception:  # Old TensorFlow version only returns outputs not states
                #     lstm_out = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=lstm_init_reshape,
                #                                  sequence_length=self.tag_seq_lens, dtype=dtype)

            self.lstm_out = lstm_out

            # compute softmax
            with tf.name_scope('predictions'):
                w_uniform_dist = tf.random_uniform([n_hidden_lstm, tag_vocabulary_size], -1.0, 1.0)
                w_out = tf.Variable(w_uniform_dist, name='W-out')
                b_out = tf.Variable(tf.zeros([tag_vocabulary_size]), name='b-out')

                outputs_reshape = tf.reshape(lstm_out, [-1, n_hidden_lstm])
                self.pred = tf.tanh(tf.matmul(outputs_reshape, w_out) + b_out, name='pred')
                self.logits = tf.matmul(outputs_reshape, w_out) + b_out

            if stags:
                self.logits = tf.reshape(self.logits, [input_shape[0],input_shape[1],input_shape[2], tag_vocabulary_size])
            else:
                self.logits = tf.reshape(self.logits, [input_shape[0],input_shape[1],tag_vocabulary_size])
         #   import pdb; pdb.set_trace()
            #targets = [self.tag_inputs[i + 1] for i in xrange(len(self.tag_inputs) - 1)] #TODO
            targets = self.tag_inputs

            with tf.name_scope("loss"):           #TODO check this
                #for logit, target, weight in zip(logits, targets, weights):
            #    for logit, target in zip(self.logits, targets):
             #       target = tf.reshape(target, [-1])
            #        cross_entropy = nn_ops.tf.nn.softmax_cross_entropy_with_logits(logit, target)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y) #TODO
                self.loss = tf.reduce_mean(cross_entropy)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            #TODO check if this makes sense in this model
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            #self.global_step = tf.Variable(0, trainable=False)
            #self.tvars = tf.trainable_variables()
            #gradients = tf.gradients(self.loss, self.tvars)
            #clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            #self.update = opt.apply_gradients( zip(clipped_gradients, self.tvars), global_step = self.global_step))

            all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
            self.saver = tf.train.Saver(all_variables)


    def _step(self, session, word_seq_lens, tag_seq_lens, word_inputs, tag_inputs, y):

        word_inputs = np.vstack([np.expand_dims(x, 0) for x in word_inputs])
        tag_inputs =  np.vstack([np.expand_dims(x, 0) for x in tag_inputs])

        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs, self.y: y}

        output_feed = [self.pred, self.loss, self.optimizer]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def step(self, session, word_seq_lens, tag_seq_lens, word_inputs, tag_inputs, y):

        word_inputs = np.vstack([np.expand_dims(x, 0) for x in word_inputs])
        import pdb; pdb.set_trace()
        tag_inputs =  np.vstack([np.expand_dims(x, 0) for x in tag_inputs])

        input_feed = {self.word_seq_lens: word_seq_lens,
                        self.tag_seq_lens: tag_seq_lens,
                        self.word_inputs: word_inputs,
                        self.tag_inputs: tag_inputs, self.y: y}

        output_feed = [self.pred, self.loss, self.optimizer]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def get_batch(self, train_data, tag_vocabulary_size, batch_size=32):

        bv = dp.generate_batch(train_data, batch_size)
        bv_w = copy.copy(bv['word'])
        bv_t = copy.copy(bv['tag'])

        seq_len_w = map(lambda x: len(x), bv_w)
        dp.data_padding(bv_w)

        seq_len_t = map(lambda y: map(lambda x: len(x), y), bv_t)
        max_len = max(map(lambda x: max(x), seq_len_t))
        bv_t_1hot = map(lambda y: map(lambda x: dp._to_onehot(x, max_len, tag_vocabulary_size), y), bv_t)
        for i, bv_i in enumerate(bv_t):
            bv_t[i] = np.vstack(dp.data_padding(bv_i, mlen = max_len))

        return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_1hot

    def _get_batch(self, train_data, tag_vocabulary_size, batch_size=32):  # TODO fix this for real general data

        bv = dp.generate_batch(train_data, batch_size)
        bv_w = copy.copy(bv['word'])
        bv_t = copy.copy(bv['tag'])

        seq_len_w = map(lambda x: len(x), bv_w)
        dp.data_padding(bv_w)

        seq_len_t = map(lambda x: len(x), bv_t)
        max_len = len(max(bv_t, key=len))
        bv_t_1hot = map(lambda x: dp._to_onehot(x, max_len, tag_vocabulary_size), bv_t)
        dp.data_padding(bv_t)

        return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_1hot
