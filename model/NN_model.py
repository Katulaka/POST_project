from __future__ import print_function

import tensorflow as tf

import numpy as np
import random

class nnModel(object):

    def __init__(self, batch_size, word_embedding_size, tag_embedding_size, n_hidden_fw, n_hidden_bw,
                n_hidden_lstm, word_vocabulary_size, tag_vocabulary_size, num_steps, learning_rate, 
                learning_rate_decay_factor, scope_name = 'nn_model'):

        try:
            LSTM                = tf.nn.rnn_cell.BasicLSTMCell
            LSTMStateTuple      = tf.nn.rnn_cell.LSTMStateTuple
        except:
            LSTM                = tf.contrib.rnn.BasicLSTMCell
            LSTMStateTuple      = tf.contrib.rnn.LSTMStateTuple
        
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):  
      
            self.word_seq_lens           = tf.placeholder(tf.int32, shape = [batch_size], name='word-sequence-length')
            self.tag_seq_lens            = tf.placeholder(tf.int32, shape = [None,], name='word-sequence-length')

            self.learning_rate          = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)

            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
            #self.learning_rate_decay_op_1 = self.learning_rate.assign(0.05)


            #self.global_step            = tf.Variable(0, trainable=False)
   
            with tf.name_scope('input'):
                self.word_inputs                 = tf.placeholder(tf.int32, shape=[None, None], name = "word-input")
                self.tag_inputs                  = tf.placeholder(tf.int32, shape=[None, None, None], name= "tag-input")
                self.y                           = tf.placeholder(tf.int32, shape=[None, None], name = "y-input")

            # Look up embeddings for inputs.
            with tf.name_scope('embedding'):
                word_embed_matrix_init      = tf.random_uniform([word_vocabulary_size, word_embedding_size], -1.0, 1.0)
                word_embed_matrix           = tf.Variable(word_embed_matrix_init, name='word-embeddings')
                word_embed                  = tf.nn.embedding_lookup(word_embed_matrix, word_inputs, name ='word-embed')

                tag_embed_matrix_init       = tf.random_uniform([tag_vocabulary_size, tag_embedding_size], -1.0, 1.0)
                tag_embed_matrix            = tf.Variable(tag_embed_matrix_init, name='tag-embeddings')
                tag_embed                   = tf.nn.embedding_lookup(tag_embed_matrix, tag_inputs, name='tag-embed')


            with tf.name_scope('bidirectional-LSTM-Layer'):
            #Bidirectional LSTM
            # Forward and Backward direction cell
                lstm_fw_cell            = LSTM(n_hidden_fw, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell            = LSTM(n_hidden_bw, forget_bias=1.0, state_is_tuple=True)
     
            # Get lstm cell output
                try:
                    bidi_out, bidi_states   = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                  lstm_bw_cell, 
                                                                  word_embed, 
                                                                  sequence_length = word_seq_lens,
                                                                  dtype = tf.float32)
                                                
                except Exception: # Old TensorFlow version only returns outputs not states
                    bidi_out                = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                  lstm_bw_cell, 
                                                                  word_embed, 
                                                                  sequence_length = word_seq_lens,
                                                                  dtype = tf.float32)

            with tf.name_scope('LSTM-Layer'): 
            #LSTM    
                lstm_init               = tf.concat(bidi_out, 2, name='lstm-init') 
                lstm_init_reshape       = tf.reshape(lstm_init, [-1,n_hidden_fw + n_hidden_bw])
   
                lstm_init_reshape       = LSTMStateTuple(lstm_init_reshape, tf.zeros_like(lstm_init_reshape))

                input_shape             = tf.shape(tag_embed)
                lstm_input              = tf.reshape(tag_embed, [input_shape[0]*input_shape[1], input_shape[2],tag_embedding_size])
 
                lstm_cell               = LSTM(n_hidden_lstm, forget_bias = 1.0, state_is_tuple = True) #TODO

                try:
                    lstm_out, _   = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=lstm_init_reshape, sequence_length=tag_seq_lens,  dtype = tf.float32)

                except Exception: # Old TensorFlow version only returns outputs not states
                    lstm_out                = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=lstm_init_reshape, sequence_length=tag_seq_lens, dtype = tf.float32)
        
  
            #compute softmax   
            with tf.name_scope('predictions'):
        
                W_uniform_dist  = tf.random_uniform([n_hidden_lstm, tag_vocabulary_size], -1.0, 1.0)
                W_out           = tf.Variable(W_uniform_dist, name='W-out')    
                b_out           = tf.Variable(tf.zeros([tag_vocabulary_size]), name='b-out')

                outputs_reshape = tf.reshape(lstm_out, [-1,n_hidden_lstm])
                self.pred            = tf.tanh(tf.matmul(outputs_reshape,W_out) + b_out, name='pred')
   

            with tf.name_scope("train") as scope:
                with tf.name_scope("loss") as scope:
                    cross_entropy               = tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)
                    self.loss                        = tf.reduce_mean(cross_entropy)

                self.optimizer                   = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    
    def step(self, session, word_seq_lens, tag_seq_lens, word_inputs, tag_inputs, y):

        input_feed = {  self.word_seq_lens  : word_seq_lens,
                        self.tag_seq_lens   : tag_seq_lens,
                        self.word_inputs    : word_inputs,
                        self.tag_inputs     : tag_inputs,
                        self.y              : y }

        output_feed = {self.pred, self.loss, self.optimizer}

        outputs = session.run(output_feed, input_feed)

        return outputs

    def get_batch(self, train_data):

        #TODO
