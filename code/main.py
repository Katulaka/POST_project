import tensorflow as tf
import time
import os
import json

from model.refactor.pos_model import POSModel
from model.refactor.stag_model import STAGModel
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
    gold = ds.gen_gold()
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
            print '[[Model Params]] %s: %s' % (k, v)
        model.train(batcher, ds.dataset)

    elif (config['mode'] == 'decode'):
        print('==================================================================')
        print("[[POST]] Starting model decodeing")
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        test_min, test_max = config['ds']['ds_range']['test']
        fname = '_'.join(['ds', str(test_min), str(test_max), now])
        dec_file = os.path.join('decode', fname + '.test')
        gold_file = os.path.join('decode', fname + '.gold')
        batcher.use_data(ds.dataset['test'])
        decode_trees = model.decode(ds.vocab, batcher, ds.t_op)
        decoded_tags = trees_to_ptb(decode_tags)
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)
        with open(gold_file, 'w') as outfile:
            json.dump(gold, outfile)

    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
