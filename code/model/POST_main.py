from __future__ import print_function

import copy
import json
import time
import tensorflow as tf

from POST_model import POSTModel

def get_model(session, config, graph, mode='decode'):

    """ Creates new model for restores existing model """
    start_time = time.time()

    model = POSTModel(config.batch_size, config.word_embedding_size,
                        config.tag_embedding_size, config.n_hidden_fw,
                        config.n_hidden_bw, config.n_hidden_lstm,
                        config.word_vocabulary_size,
                        config.tag_vocabulary_size, config.learning_rate,
                        config.learning_rate_decay_factor, config.add_pos_in,
                        config.add_w_pos_in, config.w_attn, mode,
                        config.reg_loss,)

    model.build_graph(graph)

    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Time to restore model: %.2f" % (time.time() - start_time))
    elif mode == 'train':
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print("Time to create model: %.2f" % (time.time() - start_time))
    else:
        raise ValueError('Model not found to restore.')
        return None
    return model
