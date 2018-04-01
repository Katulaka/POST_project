import tensorflow as tf
import numpy as np
import time
import os
import json
import datetime

from model.pos_model import POSModel
from model.stag_model import STAGModel
from utils.batcher import Batcher
from utils.dataset import Dataset
from utils.parse_cmdline import parse_cmdline
from utils.tags.tree_t import trees_to_ptb

def main(_):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = parse_cmdline()
    print('==================================================================')
    # create vocabulary and array of dataset from train file
    ds = Dataset(config['ds'])
    data = ds.gen_dataset()
    print('==================================================================')
    # initializing batcher class
    batcher = Batcher(**config['btch'])

    for k in ds.vocab.keys():
        config['n'+k] = ds.nsize[k]
    config['npos'] = config['ntags']
    print('==================================================================')
    for k,v in config.items():
        print ('[[Model Params]] %s: %s' % (k, v))

    if (config['mode'] in ['train', 'decode', 'stats']):
        model = POSModel(config) if config['pos'] else STAGModel(config)

    if (config['mode'] == 'train'):
        print('==================================================================')
        print("[[POST]] Starting model training.")
        model.train(batcher, ds.dataset)

    elif (config['mode'] == 'decode'):
        print('==================================================================')
        print("[[POST]] Starting model decodeing")

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        test_min, test_max = config['ds']['ds_range']['test']
        fname = '_'.join(['ds', str(test_min), str(test_max), now])
        dir_name = os.path.join('results', config['model_name'], config['mode'])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        batcher.use_data(ds.dataset['test'])
        decoded = model.decode(ds.vocab, batcher, ds.t_op)
        # import pdb; pdb.set_trace()
        pattern = np.array(decoded)[:,1].tolist()
        import ipdb; ipdb.set_trace()
        pattern_file = os.path.join(dir_name, fname + '.ptrn')
        with open(pattern_file, 'w') as outfile:
            json.dump(pattern, outfile)

        decode_trees = np.array(decoded)[:,0].tolist()
        decoded_tags = trees_to_ptb(decode_trees)
        dec_file = os.path.join(dir_name, fname + '.test')
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)

        gold = ds.gen_gold()
        gold_file = os.path.join(dir_name, fname + '.gold')
        with open(gold_file, 'w') as outfile:
            json.dump(gold, outfile)

    elif (config['mode'] == 'stats'):

        batcher.use_data(ds.dataset['test'])
        stats, stats_mod, stats_out = model.stats(batcher, ds.vocab)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        dir_name = os.path.join('code', 'plot')
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        fname = os.path.join(dir_name, '_'.join(['data', now]))
        with open(fname, 'w') as outfile:
            data = {'stats':stats, 'stats_mod':stats_mod, 'stats_out': stats_out}
            json.dump(data, outfile)
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
