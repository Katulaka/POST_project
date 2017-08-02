import numpy as np
import tensorflow as tf
import time
import os

import utils.dataset as ds
import model.POST_main as POST_main
from utils.conf import Config
from utils.batcher import Batcher


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--tags_type', type=str, default='stags')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    # # Download raw data for training #TODO
    # if not os.path.exists(Config.src_dir):
    #      ds.get_raw_data(Config.src_dir)

    data_dir = os.path.join(os.getcwd(), Config.train_dir)
    w_file = os.path.join(data_dir, 'words')
    t_file = os.path.join(data_dir, args.tags_type)
    # Convert raw data into words and tags files
    if not os.path.exists(t_file) or os.path.getsize(t_file) == 0:
        ds.convert_data_flat(Config.src_dir, w_file, t_file, args.tags_type)
    else:
        print ("Words data in %s \nTags data in %s" % (w_file, t_file))

    # create vocabulary and array of dataset from train file
    w_vocab, t_vocab, train_set = ds.gen_dataset(w_file, t_file, max_len=0)
    batcher = Batcher(train_set, t_vocab.vocab_size(), args.batch)

    Config.batch_size = args.batch
    Config.tag_vocabulary_size = t_vocab.vocab_size()
    Config.word_vocabulary_size = w_vocab.vocab_size()
    Config.checkpoint_path = os.path.join(os.getcwd(),
                                            'checkpoints',
                                            args.tags_type)

    if (args.action == 'train'):
        POST_main.train(Config, batcher, args.tags_type)
    elif (args.action == 'decode'):
        orig_tags, dec_tags = POST_main.decode(Config, t_vocab, batcher)
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
