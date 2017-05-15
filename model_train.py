import numpy as np

import conf as conf
import main_NN as main_NN

import utils.gen_dataset as gd


def main(_):
    seed = int(time.time())
    np.random.seed(seed)  

    import argparse
    parser = argparse.ArgumentParser()
  #  parser.add_argument('train_type', type=str)
  #  parser.add_argument('--gen_file', type=str)
  #  parser.add_argument('--disc_file', type=str)
  #  parser.add_argument('--gen_prob', type=float)
  #  parser.add_argument('--disc_prob', type=float)

    args = parser.parse_args()

  #  if (args.gen_file):
  #      conf.gen_config.train_data_file = args.gen_file

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

    gd.generate_data(config.src_dir, config.dest_dir)
  
    main_NN.train(conf.config)
    



if __name__ == "__main__":
    tf.app.run()
