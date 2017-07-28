import numpy as np
import tensorflow as tf
import time
import os

import utils.gen_dataset as gd
import utils.data_preproc as dp
import utils.conf
import model.NN_main as NN_main



def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    Config = utils.conf.Config

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
    parser.add_argument('--tags_type', type=str, default='stags')
    args = parser.parse_args()


    data_dir = os.path.join(os.getcwd(), 'data')
    w_file = os.path.join(data_dir, 'words')
    t_file = os.path.join(data_dir, args.tags_type)
    gen_tags_fn = gd.gen_tags if args.tags_type == 'tags' else gd.gen_stags

    if not os.path.exists(t_file):
        gd.generate_data_flat(Config.src_dir, data_dir, gen_tags_fn)

    _, dictionary, reverse_dictionary, train_set = dp.gen_dataset(w_file, t_file)

    #TODO maybe fix gen_dataset to get config values
    Config.tag_vocabulary_size = max(dictionary['tag'].values()) + 1
    Config.word_vocabulary_size = max(dictionary['word'].values()) + 1

    Config.checkpoint_path = os.path.join(os.getcwd(), 'checkpoints',
                                                                args.tags_type)
    if (args.action == 'train'):

        NN_main.train(Config, train_set, args.tags_type)
    elif (args.action == 'decode'):
        # Config.dec_timesteps = 3 if args.tags_type == 'tags' else 20
        words_in, tags_in, best_beam = NN_main.decode(Config, train_set)

        tmp = [[[(map(lambda x: reverse_dict['tag'][x],g[0][1:-1]), g[1])
                for g in bi] for bi in b]
                    for b in best_beam ]
    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
