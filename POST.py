import numpy as np
import tensorflow as tf
import time
import os

import utils.dataset as ds
import model.POST_main as POST_main
from utils.conf import Config
from utils.batcher import Batcher
from utils.data import *


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--tags_type', type=str, default='stags')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), 'data')
    w_file = os.path.join(data_dir, 'words')
    t_file = os.path.join(data_dir, args.tags_type)

    if not os.path.exists(t_file) or os.path.getsize(t_file) == 0:
        ds.convert_data_flat(Config.src_dir, w_file, t_file, args.tags_type)
    else:
        print ("Found word data in %s \nFound tag data in %s" % (w_file, t_file))

    w_vocab, words = textfile_to_vocab(w_file)
    t_vocab, tags = textfile_to_vocab(t_file, is_tag=True)

    train_set = gen_dataset(words, w_vocab, tags, t_vocab)

    Config.batch_size = args.batch
    Config.tag_vocabulary_size = t_vocab.vocab_size()
    Config.word_vocabulary_size = w_vocab.vocab_size()
    Config.checkpoint_path = os.path.join(os.getcwd(),
                                            'checkpoints',
                                            args.tags_type)

    batcher = Batcher(train_set, Config.tag_vocabulary_size, Config.batch_size)

    if (args.action == 'train'):
        POST_main.train(Config, batcher, args.tags_type)

    elif (args.action == 'decode'):
        orig_tags, dec_tags = POST_main.decode(Config, t_vocab, batcher)

    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
