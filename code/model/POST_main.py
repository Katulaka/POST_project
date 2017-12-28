from __future__ import print_function

import copy
import json
import time
import tensorflow as tf

from POST_model import POSTModel

def get_model(session, config, graph, mode='decode'):

    """ Creates new model for restores existing model """
    start_time = time.time()

    model = POSTModel(config.ModelParms, mode)
    model.build_graph(graph)
    checkpoint_path = config.checkpoint_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
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
