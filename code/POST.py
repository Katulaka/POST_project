import numpy as np
import tensorflow as tf
import time
import os

from utils.dataset import gen_dataset, split_dataset
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
    parser.add_argument('--action', type=str, default='train', help='')
    parser.add_argument('--cp_dir', type=str, default='stags', help='')
    parser.add_argument('--stat_file', type=str, default='stats', help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--add_pos_in', action='store_true', help='')
    parser.add_argument('--add_w_pos_in', action='store_true', help='')
    parser.add_argument('--attn', action='store_true', help='')
    parser.add_argument('--ds_len', type=int, default=np.inf, help='')
    parser.add_argument('--beam', type=int, default=5, help='')
    parser.add_argument('--only_pos', action='store_true', help='')
    parser.add_argument('--keep_direction', action='store_true', help='')
    parser.add_argument('--no_val_gap', action='store_true', help='')
    parser.add_argument('--reverse', action='store_true', help='')

    # parser.add_argument('--tag_split', action='store_true', help='')
    # parser.add_argument('--slash_split', action='store_true', help='')

    args = parser.parse_args()

    data_file = os.path.join(os.getcwd(), Config.train_dir, 'new_data.txt')

    # create vocabulary and array of dataset from train file
    print("Generating dataset and vocabulary")
    start_time = time.time()
    w_vocab, t_vocab, dataset, t_op, tags = gen_dataset(Config.src_dir,
                                            data_file,
                                            (args.only_pos,
                                            args.keep_direction,
                                            # args.tag_split,
                                            # args.slash_split,
                                            # args.reverse,
                                            args.no_val_gap),
                                            max_len=args.ds_len)
    print ("Time to generate dataset and vocabulary %f" % (time.time()-start_time))
    ratio = 0.1
    train_set, test_set = split_dataset(dataset, ratio)
    # initializing batcher class
    batcher_train = Batcher(train_set, args.batch, args.reverse)
    batcher_test = Batcher(test_set, args.batch, args.reverse)


    # Update config variables
    Config.batch_size = args.batch
    Config.beam_size = args.beam
    Config.tag_vocabulary_size = t_vocab.vocab_size()
    Config.word_vocabulary_size = w_vocab.vocab_size()
    Config.checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', args.cp_dir)

    if (args.action == 'train'):
        # POST_main.train(Config, batcher, args.cp_dir, w_vocab.get_ctrl_tokens(),
        #                 args.add_pos_in, args.attn)
        th_loss = 0.1
        POST_main.train_eval(Config, batcher_train, batcher_test, args.cp_dir,
                            w_vocab.get_ctrl_tokens(), args.add_pos_in,
                            args.add_w_pos_in, args.attn, th_loss)

    elif (args.action == 'decode'):
        mrg_tags, decoded_tags = POST_main.decode(Config, w_vocab, t_vocab,
                                                    batcher_test, t_op,
                                                    args.add_pos_in,
                                                    args.add_w_pos_in,
                                                    args.attn)
        import pdb; pdb.set_trace()

    elif(args.action == 'stats'):
        stats = POST_main.stats(Config, w_vocab, t_vocab, batcher, t_op,
                                args.add_pos_in, args.add_w_pos_in,
                                args.attn, args.stat_file)
    elif(args.action == 'verify'):
        verify_tags = POST_main.verify(t_vocab, batcher, t_op)
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
