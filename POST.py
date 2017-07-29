import numpy as np
import tensorflow as tf
import time
import os

import utils.dataset as gd
from utils.batcher import *
import utils.conf
import model.NN_main as NN_main


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    Config = utils.conf.Config

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--tags_type', type=str, default='stags')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), 'data')
    w_file = os.path.join(data_dir, 'words')
    t_file = os.path.join(data_dir, args.tags_type)
    gen_tags_fn = gd.gen_tags if args.tags_type == 'tags' else gd.gen_stags

    if not os.path.exists(t_file):
        gd.generate_data_flat(Config.src_dir, data_dir, args.tags_type)

    #TODO maybe fix gen_dataset to get config values
    word_vocab, tag_vocab, train_set = gd.gen_dataset(w_file, t_file)
    Config.tag_vocabulary_size = tag_vocab.vocab_size()
    Config.word_vocabulary_size = word_vocab.vocab_size()
    Config.checkpoint_path = os.path.join(os.getcwd(), 'checkpoints',
                                            args.tags_type)
    Config.batch_size = args.batch

    if (args.action == 'train'):
        NN_main.train(Config, train_set, args.tags_type)

    elif (args.action == 'decode'):
        orig_tags, decode_tags = NN_main.decode(Config, train_set,
                                            tag_vocab._id_to_token)

        import pdb; pdb.set_trace()
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
