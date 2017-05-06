from __future__ import print_function

import tensorflow as tf

#import collections
#from os import listdir
#from os.path import isfile, join

import numpy as np
import random

#class NN_model(object):

#    def __init__():

def gen_toy_data(batch_size,tag_vocabulary_size,word_vocabulary_size, max_len = 3):
    ex = np.random.randint(tag_vocabulary_size, size=(max_len*max_len*batch_size))
    labels = np.zeros((max_len*max_len*batch_size,tag_vocabulary_size))
    labels[np.arange(max_len*max_len*batch_size),ex] = 1
    w_in = np.random.randint(word_vocabulary_size, size = (batch_size, max_len))
    t_in = np.random.randint(tag_vocabulary_size, size = (batch_size, max_len, max_len))
    w_s_len = [max_len]* batch_size
    t_s_len = [max_len]* (max_len*batch_size)

    return t_s_len, w_s_len, t_in, w_in, labels



def build_model(batch_size             = 100, 
                word_embedding_size    = 128, 
                tag_embedding_size     = 128, 
                n_hidden_fw            = 128,
                n_hidden_bw            = 128,
                n_hidden_lstm          = 256,
                word_vocabulary_size   = 100000,
                tag_vocabulary_size    = 100000,
                num_steps              = 100001,
                learning_rate          = 0.1
                ):

    try:
        LSTM                = tf.nn.rnn_cell.BasicLSTMCell
        LSTMStateTuple      = tf.nn.rnn_cell.LSTMStateTuple
    except:
        LSTM                = tf.contrib.rnn.BasicLSTMCell
        LSTMStateTuple      = tf.contrib.rnn.LSTMStateTuple
        
      
    word_seq_lens           = tf.placeholder(tf.int32, shape = [batch_size], name='word-sequence-length')
    tag_seq_lens            = tf.placeholder(tf.int32, shape = [None,], name='word-sequence-length')
   
    with tf.name_scope('input'):
        word_inputs                 = tf.placeholder(tf.int32, shape=[None, None], name = "word-input")
        tag_inputs                  = tf.placeholder(tf.int32, shape=[None, None, None], name= "tag-input")
        y                           = tf.placeholder(tf.int32, shape=[None, None], name = "y-input")

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
        pred            = tf.tanh(tf.matmul(outputs_reshape,W_out) + b_out, name='pred')
   

    with tf.name_scope("train") as scope:
        with tf.name_scope("loss") as scope:
            cross_entropy               = tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)
            loss                        = tf.reduce_mean(cross_entropy)

        optimizer                   = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    
    loss_sum = tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Initialized")
        
    logs_path = '/Users/katia.patkin/Berkeley/Research/BiRNN/tmp'    
    writer = tf.summary.FileWriter(logdir=logs_path, graph= sess.graph)    

    training_epochs = 10
    num_examples    = 40
    display_step    = 1

    
    for epoch in range(training_epochs):
    
        avg_loss = 0.
        total_batch = num_examples/batch_size

        for i in range(total_batch):
            t_s_len, w_s_len, t_in, w_in, labels = gen_toy_data(batch_size,tag_vocabulary_size,word_vocabulary_size, max_len = 3)  
                             
            feed_dict =  {word_inputs : w_in, tag_inputs : t_in, word_seq_lens : w_s_len, tag_seq_lens: t_s_len, y : labels}

            [ap, l] = sess.run([summary_op, loss], feed_dict = feed_dict)
            

            avg_loss += l/total_batch
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
  

def main():
    
     
    
  

if __name__ == "__main__":

    main()
    
    
  #  build_model(batch_size              = 2, 
  #                   word_embedding_size     = 4, 
  #                   tag_embedding_size      = 4, 
  #                   n_hidden_fw             = 7, 
  #                   n_hidden_bw             = 7, 
  #                   n_hidden_lstm           = 14,
  #                   word_vocabulary_size    = 10,  
  #                   tag_vocabulary_size     = 10,
  #                   num_steps               = 10)





