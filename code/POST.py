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

PAD = ['PAD', 0]
GO = ['GO', 1]
EOS = ['EOS', 2]
UNK = ['UNK', 3]


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', help='')
    parser.add_argument('--cp_dir', type=str, default='stags', help='')
    parser.add_argument('--stat_file', type=str, default='stats', help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--use_pos', action='store_true', help='')
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
    parser.add_argument('--reg_loss', action='store_true', help='')
    parser.add_argument('--time_out', type=float, default=100., help='')

    args = parser.parse_args()

    # ds_file = os.path.join(os.getcwd(), Config.dataset_dir, Config.dataset_fname)
    ds_dir = os.path.join(os.getcwd(), Config.ds_dir)
    Config.gold_file = os.path.join(ds_dir, Config.gold_fname)
    Config.ds_file = os.path.join(ds_dir, Config.ds_fname)
    Config.tags_type = (args.only_pos, args.keep_direction, args.no_val_gap)
    Config.ds_range = {'train': (args.train_min, args.train_max),
                        'dev': (args.dev_min, args.dev_max),
                        'test': (args.test_min, args.test_max)}
    # create vocabulary and array of dataset from train file
    print("Generating dataset and vocabulary")
    start_time = time.time()
    w_vocab, t_vocab, dataset, t_op, tags, gold_data = gen_dataset(Config)
    print ("Time to generate dataset and vocabulary %f" %
                    (time.time()-start_time))
    # initializing batcher class
    batcher_train = Batcher(dataset['train'], args.batch, args.reverse)
    batcher_dev = Batcher(dataset['dev'], args.batch, args.reverse)
    batcher_test = Batcher(dataset['test'], args.batch, args.reverse)

    # Update config variables
    Config.batch_size = args.batch
    Config.beam_size = args.beam
    Config.tag_vocabulary_size = t_vocab.vocab_size()
    Config.word_vocabulary_size = w_vocab.vocab_size()
    Config.cp_dir = args.cp_dir
    Config.checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', args.cp_dir)
    Config.w_attn = args.attn
    Config.reg_loss = args.reg_loss
    Config.time_out = args.time_out
    Config.num_goals = args.num_goals
    Config.no_val_gap = args.no_val_gap
    Config.pos = args.pos
    Config.use_pos = args.use_pos

    if (args.action == 'train'):
        POST_train.train_eval(Config, batcher_train, batcher_dev)

    elif (args.action == 'decode'):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        fname = '_'.join(['ds', str(args.test_min), str(args.test_max), now])
        dec_file = os.path.join('decode', fname + '.test')
        gold_file = os.path.join('decode', fname + '.gold')
        decode_tags = POST_decode.decode(Config, w_vocab, t_vocab,
                                        batcher_test, t_op,)
        with open(dec_file, 'w') as outfile:
            json.dump(decode_tags, outfile)
        with open(gold_file, 'w') as outfile:
            json.dump(gold_data, outfile)

    elif(args.action == 'stats'):
        stats = POST_decode.stats(Config, w_vocab, t_vocab, batcher_test, t_op,
                                    args.stat_file)
    elif(args.action == 'verify'):
        verify_tags = POST_decode.verify(t_vocab, batcher, t_op)
        import collections
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        res = map(lambda to, tn: all(map(lambda x,y: compare(x,y), to, tn)),
                                        tags, verify_tags)
        idx = np.where(np.logical_not(res))[0]
        verif_tags_miss = np.array(tags)[idx]
        if verif_tags_miss == []:
            print ("Search function works")
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
