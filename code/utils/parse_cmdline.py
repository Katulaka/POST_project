import os
import argparse
import numpy as np
import time
import tensorflow as tf


def parse_cmdline():
    seed = int(time.time())
    np.random.seed(seed) #TODO tf random seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', help='')
    parser.add_argument('--model_name', type=str, default='stags', help='')
    parser.add_argument('--pos_model', type=str, default='pos', help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--use_pos', action='store_true', help='')
    parser.add_argument('--use_c_embed', action='store_true', help='')
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
    parser.add_argument('--comb_loss', action='store_true', help='')
    parser.add_argument('--time_out', type=float, default=100., help='')

    args = parser.parse_args()

    config = dict()
    config['tags_type'] = (args.only_pos, args.keep_direction, args.no_val_gap)
    config['ds_range'] = {'train': (args.train_min, args.train_max),
                            'dev': (args.dev_min, args.dev_max),
                            'test': (args.test_min, args.test_max)}

    config['src_dir'] = '../gold_data'
    config['ds_dir'] = 'data'
    config['ds_fname'] = 'data.txt'
    config['gold_fname'] = 'gold'
    config['gold_file'] = os.path.join(os.getcwd(), config['ds_dir'], config['gold_fname'])
    config['ds_file'] = os.path.join(os.getcwd(), config['ds_dir'], config['ds_fname'])

    config['model_name'] = args.model_name
    config['result_dir'] = os.path.join(os.getcwd(), 'results', args.model_name)
    config['ckpt_dir'] = os.path.join(config['result_dir'], 'checkpoints')
    config['steps_per_ckpt'] = 10
    # Update config variables
    config['batch_size'] = args.batch
    config['reverse'] = args.reverse
    config['attn'] = args.attn
    config['comb_loss'] = args.comb_loss
    config['pos'] = args.pos
    config['use_c_embed'] = args.use_c_embed
    config['use_pretrained_pos'] = args.use_pos
    config['mode'] = args.action

    #embedding size
    config['dim_word'] = 128
    config['dim_tag'] = 64
    config['dim_char'] = 32
    config['dim_pos'] = 64
    #NN dims
    config['hidden_char'] = 32
    config['hidden_pos'] = 128
    config['hidden_word'] = 128
    config['hidden_tag'] = 256

    config['lr'] = 0.01
    config['th_loss'] = 0.1
    #training num epochs before evaluting the dev loss
    config['num_epochs'] = 1
    config['debug'] = True
    config['pos_model'] = args.pos_model
    pos_model_path = os.path.join(os.getcwd(), 'results', args.pos_model, 'checkpoints')
    config['pos_ckpt'] = tf.train.latest_checkpoint(pos_model_path)
    config['frozen_graph_fname'] = os.path.join(pos_model_path,'frozen_model.pb')
    config['time_out'] = args.time_out
    config['num_goals'] = args.num_goals
    config['beam_size'] = args.beam
    config['dec_timesteps'] = 20
    config['scope_name'] = 'pos_model' if args.pos else 'stag_model'

    return config