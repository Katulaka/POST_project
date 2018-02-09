from __future__ import print_function

import copy
import json
import time
import tensorflow as tf

from POST_model import POSTModel

def get_model(session, config, graph, mode='decode'):
    print('==========================================================')
    """ Creates new model or restores existing model """
    start_time = time.time()

    model = POSTModel(config.ModelParms, mode)
    model.build_graph(graph)
    ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("[[get_model:]] Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        print("[[get_model:]] %.3fs to restore model" % (time.time() - start_time))
    elif mode == 'train':
        print("[[get_model:]] Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print("[[get_model:]] %.3fs to create model" % (time.time() - start_time))
    else:
        raise ValueError('[[get_model:]] Model not found to restore.')
        return None
    return model
