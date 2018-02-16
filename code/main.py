import tensorflow as tf
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

    model = POSModel(config) if config['pos'] else STAGModel(config)

    if (config['mode'] == 'train'):
        print('==================================================================')
        print("[[POST]] Starting model training.")
        for k,v in config.items():
            print ('[[Model Params]] %s: %s' % (k, v))
        model.train(batcher, ds.dataset)

    elif (config['mode'] == 'decode'):
        print('==================================================================')
        print("[[POST]] Starting model decodeing")

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        test_min, test_max = config['ds']['ds_range']['test']
        fname = '_'.join(['ds', str(test_min), str(test_max), now])

        batcher.use_data(ds.dataset['test'])
        decode_trees = model.decode(ds.vocab, batcher, ds.t_op)
        decoded_tags = trees_to_ptb(decode_tags)
        dec_file = os.path.join('decode', fname + '.test')
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)

        gold = ds.gen_gold()
        gold_file = os.path.join('decode', fname + '.gold')
        with open(gold_file, 'w') as outfile:
            json.dump(gold, outfile)

    elif (config['mode'] == 'stats'):

        batcher.use_data(ds.dataset['test'])
        stats, stats_mod, stats_out = model.stats(batcher, ds.vocab)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # import pdb; pdb.set_trace()
        fname = os.path.join('code', 'plot', '_'.join(['data', now]))
        with open(fname, 'w') as outfile:
            data = {'stats':stats, 'stats_mod':stats_mod, 'stats_out': stats_out}
            json.dump(data, outfile)
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
