import numpy as np
import tensorflow as tf
import time
import os

import utils.dataset as gd
from utils.batcher import *
from utils.data import *
import utils.conf
import model.NN_main as NN_main


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    config = utils.conf.Config

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
        gd.convert_data_flat(config.src_dir, w_file, t_file, args.tags_type)
    else:
        print ("Found word data in %s \nFound tag data in %s" % (w_file, t_file))

    w_vocab, words = textfile_to_vocab(w_file)
    t_vocab, tags = textfile_to_vocab(t_file, is_tag=True)

    train_set = gd.gen_dataset(words, w_vocab, tags, t_vocab)

    config.batch_size = args.batch
    config.tag_vocabulary_size = t_vocab.vocab_size()
    config.word_vocabulary_size = w_vocab.vocab_size()
    config.checkpoint_path = os.path.join(os.getcwd(),
                                            'checkpoints',
                                            args.tags_type)

    batcher = Batcher(config.tag_vocabulary_size, config.batch_size)

    if (args.action == 'train'):
        NN_main.train(config, train_set, args.tags_type)

    elif (args.action == 'decode'):
        orig_tags, dec_tags = NN_main.decode(config, train_set, t_vocab, batcher)
        import pdb; pdb.set_trace()

    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
