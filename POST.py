import numpy as np
import tensorflow as tf
import time

import utils.gen_dataset as gd
import utils.conf
import model.NN_main as NN_main
import utils.data_preproc as dp


def main(_):
    seed = int(time.time())
    np.random.seed(seed)

    Config = utils.conf.Config

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
  #  parser.add_argument('--gen_file', type=str)
  #  parser.add_argument('--disc_file', type=str)
  #  parser.add_argument('--gen_prob', type=float)
  #  parser.add_argument('--disc_prob', type=float)

    args = parser.parse_args()

   # if (args.gen_file):
   #     conf.gen_config.train_data_file = args.gen_file

  #  if(args.gen_prob):
  #      conf.gen_config.keep_prob = args.gen_prob

  #  if (args.disc_file):
  #      conf.disc_config.train_data_file = args.disc_file

  #  if(args.disc_prob):
  #      conf.disc_config.keep_prob = args.disc_prob

  #  if args.train_type == 'disc':
  #      print ("Runinng Discriminator Pre-Train")
  #      disc_pre_train()
  #  elif args.train_type == 'gen':
  #      print ("Runinng Generator Pre-Train")
  #      gen_pre_train()
  #  elif args.train_type == 'gen2':
  #      print ("Runinng Generator Pre-Train 2")
  #      gen_pre_train2()
  #  else:
  #      print ("Runinng Adversarial")
  #      al_train()


    #gd.generate_data(Config.src_dir, Config.dest_dir)
    _, dictionary, reverse_dictionary, train_set = dp.gen_dataset(w_file = 'data/words', t_file = 'data/stags')
    #TODO maybe fix gen_dataset to get config values
    Config.tag_vocabulary_size = max(dictionary['tag'].values())
    Config.word_vocabulary_size = max(dictionary['word'].values())
    if (args.action == 'train'):
        NN_main.train(Config, train_set) #TODO Maybe make it read trainset from file
    elif (args.action == 'decode'):
        NN_main.decode(Config, train_set)
    else:
        print("Nothing to do!!")
    # NN_main.evaluate(Config)


if __name__ == "__main__":
    tf.app.run()
