import tensorflow as tf
import time
import os
import json

from model.refactor.pos_model import POSModel
from model.refactor.stag_model import STAGModel
from utils.batcher import Batcher
from utils.dataset import _gen_dataset
from utils.parse_cmdline import parse_cmdline
# from utils.tags.ptb_tags_convert import trees_to_ptb
from utils.tags.tree_t import trees_to_ptb

def main(_):
    config = parse_cmdline()
    # create vocabulary and array of dataset from train file
    print('==================================================================')
    print("[[POST:]] Generating dataset and vocabulary")
    start_t = time.time()
    vocab, dataset, t_op, tags, gold = _gen_dataset(config)
    config['npos'] = config['ntags'] = vocab['tags'].vocab_size()
    config['nwords'] = vocab['words'].vocab_size()
    config['nchars'] = vocab['chars'].vocab_size()
    print ("[[POST]] %.3f to get dataset and vocabulary" %(time.time()-start_t))

    model = POSModel(config) if config['pos'] else STAGModel(config)
    # initializing batcher class
    batcher_train = Batcher(dataset['train'], config['batch_size'], config['reverse'])
    batcher_dev = Batcher(dataset['dev'], config['batch_size'], config['reverse'])
    batcher_test = Batcher(dataset['test'], config['batch_size'], config['reverse'])

    if (config['mode'] == 'train'):
        print('==================================================================')
        print("[[POST]] Starting model training.")
        for k,v in config.items():
            print '[[Model Params]] %s: %s' % (k, v)
        model.train(batcher_train, batcher_dev)

    elif (config['mode'] == 'decode'):
        print('==================================================================')
        print("[[POST]] Starting model decodeing")
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        test_min, test_max = config['ds_range']['test']
        fname = '_'.join(['ds', str(test_min), str(test_max), now])
        dec_file = os.path.join('decode', fname + '.test')
        gold_file = os.path.join('decode', fname + '.gold')
        decode_trees = model.decode(vocab, batcher_test, t_op)
        decoded_tags = trees_to_ptb(decode_tags)
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)
        with open(gold_file, 'w') as outfile:
            json.dump(gold, outfile)

    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
