import numpy as np
import tensorflow as tf
import time
import os

from utils.dataset import gen_dataset
import model.POST_main as POST_main
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
    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--tags_type', type=str, default='stags')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--ds_len', type=int, default=np.inf)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--delim', help='', action='store_true')
    parser.add_argument('--only_pos', help='', action='store_true')
    parser.add_argument('--tag_split', help='', action='store_true')
    parser.add_argument('--tag_sides', help='', action='store_true')

    args = parser.parse_args()

    data_file = os.path.join(os.getcwd(), Config.train_dir, 'data.pkl')
    # data_file = os.path.join(data_dir, 'data.pkl')
    # create vocabulary and array of dataset from train file
    print("Generating dataset and vocabulary")
    start_time = time.time()

    w_vocab, t_vocab, train_set = gen_dataset(Config.src_dir,
                                            data_file,
                                            (args.tag_split,
                                            args.tag_sides,
                                            args.only_pos),
                                            max_len=args.ds_len)
    print ("Time to generate dataset and vocabulary %f" % (time.time()-start_time))
    batcher = Batcher(train_set, t_vocab.vocab_size(), args.batch, args.delim)

    Config.batch_size = args.batch
    Config.beam_size = args.beam
    Config.tag_vocabulary_size = t_vocab.vocab_size()
    Config.word_vocabulary_size = w_vocab.vocab_size()
    Config.checkpoint_path = os.path.join(os.getcwd(),
                                            'checkpoints',
                                            args.tags_type)

    # special_tokens = w_vocab.get_ctrl_tokens()
    if (args.action == 'train'):
        POST_main.train(Config, batcher, args.tags_type,
                                        w_vocab.get_ctrl_tokens())
    elif (args.action == 'decode'):
        orig_tags, dec_tags = POST_main.decode(Config, w_vocab, t_vocab,
                                                batcher,)
                                                # special_tokens)
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
