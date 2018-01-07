import numpy as np
import tensorflow as tf
import time
import os
import json

from utils.dataset import gen_dataset
import model.POST_decode as POST_decode
import model.POST_train as POST_train
from utils.conf import Config
from utils.batcher import Batcher
from utils.tags.tree_t import trees_to_ptb

def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', help='')
    parser.add_argument('--ckpt_dir', type=str, default='stags', help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--use_pos', action='store_true', help='')
    parser.add_argument('--use_c_embed', action='store_true', help='')
    parser.add_argument('--attn', action='store_true', help='')
    parser.add_argument('--test_min', default=0, type=int)
    parser.add_argument('--test_max', default=np.inf, type=int)
    parser.add_argument('--dev_min', default=0, type=int)
    parser.add_argument('--dev_max', default=np.inf, type=int)
    parser.add_argument('--train_min', default=0, type=int)
    parser.add_argument('--train_max', default=np.inf, type=int)
    parser.add_argument('--beam', type=int, default=5, help='')
    parser.add_argument('--only_pos', action='store_true', help='')
    parser.add_argument('--keep_direction', action='store_true', help='')
    parser.add_argument('--no_val_gap', action='store_true', help='')
    parser.add_argument('--reverse', action='store_true', help='')
    parser.add_argument('--num_goals', type=int, default=1, help='')
    parser.add_argument('--comb_loss', action='store_true', help='')
    parser.add_argument('--time_out', type=float, default=100., help='')
    # parser.add_argument('--stat_file', type=str, default='stats', help='')

    args = parser.parse_args()

    tags_type = (args.only_pos, args.keep_direction, args.no_val_gap)
    ds_range = {'train': (args.train_min, args.train_max),
                'dev': (args.dev_min, args.dev_max),
                'test': (args.test_min, args.test_max)}
    batch_prop = {'batch_size' : args.batch, 'reverse' : args.reverse}

    # create vocabulary and array of dataset from train file
    print('==================================================================')
    print("[[POST:]] Generating dataset and vocabulary")
    start_time = time.time()
    vocab, dataset, t_op, tags, gold = gen_dataset(Config, tags_type, ds_range)
    print ("[[POST:]] %.3fs to get dataset and vocabulary " %
                        (time.time()-start_time))

    # initializing batcher class
    batcher = Batcher(**batch_prop)

    # Update config variables
    Config.ModelParms.batch_size = args.batch
    Config.ModelParms.ntags = vocab['tags'].vocab_size()
    Config.ModelParms.nwords = vocab['words'].vocab_size()
    Config.ModelParms.nchars = vocab['chars'].vocab_size()
    Config.ModelParms.attn = args.attn
    Config.ModelParms.comb_loss = args.comb_loss
    Config.ModelParms.pos = args.pos
    Config.ModelParms.use_c_embed = args.use_c_embed
    Config.ModelParms.use_pos = args.use_pos

    Config.time_out = args.time_out
    Config.num_goals = args.num_goals
    Config.no_val_gap = args.no_val_gap

    Config.ckpt_dir = args.ckpt_dir
    Config.ckpt_path = os.path.join(os.getcwd(), 'checkpoints', args.ckpt_dir)

    Config.beam_size = args.beam

    if (args.action == 'train'):
        print('==================================================================')
        print("[[POST:]] Starting model training. Model Parameters:")
        for k in sorted(Config.ModelParms.__dict__.iterkeys()):
            if not k.startswith('__'):
                print '[[POST:]] %s: %s' % (k, Config.ModelParms.__dict__[k])
        POST_train.train(Config, batcher, ds.dataset)

    elif (args.action == 'decode'):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        fname = '_'.join(['ds', str(args.test_min), str(args.test_max), now])
        dec_file = os.path.join('decode', fname + '.test')
        gold_file = os.path.join('decode', fname + '.gold')
        batcher.use_data(ds.dataset['test'])
        decoded_trees = POST_decode.decode(Config, ds.vocab, batcher, ds.t_op)
        decoded_tags = trees_to_ptb(decoded_trees)
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)
        with open(gold_file, 'w') as outfile:
            json.dump(gold, outfile)

    # elif(args.action == 'stats'):
    #     stats = POST_decode.stats(Config, vocab, batcher_test, t_op,
    #                                 args.stat_file)

    # elif(args.action == 'verify'):
    #     verify_tags = POST_decode.verify(t_vocab, batcher, t_op)
    #     import collections
    #     compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    #     res = map(lambda to, tn: all(map(lambda x,y: compare(x,y), to, tn)),
    #                                     tags, verify_tags)
    #     idx = np.where(np.logical_not(res))[0]
    #     verif_tags_miss = np.array(tags)[idx]
    #     if verif_tags_miss == []:
    #         print ("Search function works")
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
