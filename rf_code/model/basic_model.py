from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import os
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import json


class BasicModel(object):
    __metaclass__ = ABCMeta
    __slots__ = ()

    def __init__ (self, config):

        # if config['best']:
        #     config.update(self.get_best_config(config['env_name']))
        self.config = copy.deepcopy(config)

        # if config['debug']: # This is a personal check i like to do
        #     print('config', self.config)
        if self.config['mode'] == 'train':
            self.num_epochs = self.config['num_epochs']
        # self.random_seed = self.config['random_seed']
        self.dtype = tf.float32
        self.initializer = tf.contrib.layers.xavier_initializer()

        import pdb; pdb.set_trace()
        if self.config['opt_fn'] == 'adam':
            self.optimizer_fn = tf.train.AdamOptimizer
        else:
            self.optimizer_fn = tf.train.GradientDescentOptimizer

        self.activation_fn = tf.nn.relu

        self.model_name = self.config['model_name']
        self.result_dir = self.config['result_dir']
        self.ckpt_dir = self.config['ckpt_dir']

        self.graph = self.build_graph()

        with self.graph.as_default():
            # all_variables = [k for k in tf.global_variables()
            #                 if k.name.startswith(self.config['scope_name'])]
            # self.saver = tf.train.Saver(all_variables, max_to_keep=4)
            self.saver = tf.train.Saver(max_to_keep=4)
            self.init_op = tf.global_variables_initializer()

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allocator_type = 'BFC'
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.40
        self.sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.sess_config, graph=self.graph)
        # self.sw = tf.summary.FileWriter(self.ckpt_dir, self.sess.graph)

        self.init()

        # def set_agent_props(self):
        # # This function is here to be overriden completely.
        # # When you look at your model, you want to know exactly which custom options it needs.
        #     pass
        #
        # def get_best_config(self):
        # # This function is here to be overriden completely.
        # # It returns a dictionary used to update the initial configuration (see __init__)
        #     return {}

        # @staticmethod
        # def get_random_config(fixed_params={}):
        # # Why static? Because you want to be able to pass this function to other processes
        # # so they can independently generate random configuration of the current model
        #     raise Exception('The get_random_config function must be overriden by the agent')

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        checkpoint = tf.train.get_checkpoint_state(self.ckpt_dir)
        if checkpoint is None:
            if self.config['mode'] == 'train':
                self.sess.run(self.init_op)
            else:
                raise ValueError('Model not found to restore.')
        else:
            # if self.config['debug']:
            print('[[basic_model.init]]Loading the model from folder: %s' % self.ckpt_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)


    @abstractmethod
    def build_graph(self, graph):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def decode(self):
        raise NotImplementedError

    def save(self):
    # This function is usually common to all your models, Here is an example:
        global_step = self.sess.run(self.global_step)
        # if self.config['debug']:
        #     print('Saving to %s with global_step %d' % (self.ckpt_dir, global_step))
        if not os.path.exists(self.ckpt_dir):
            try:
                os.makedirs(os.path.abspath(self.ckpt_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, self.model_name),
                        global_step)

        # I always keep the configuration that
        # if not os.path.isfile(self.ckpt_dir + '/config.json'):
        with open(os.path.join(self.result_dir,'config.json'), 'w') as f:
            json.dump({self.config['mode']: self.config}, f)


    def freeze_graph(self, output_node_names):
        if not tf.gfile.Exists(self.ckpt_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % model_dir)

        if not output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1

        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(self.ckpt_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/frozen_model.pb"

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess, # The session is used to retrieve the weights
            self.graph.as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def

    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph
