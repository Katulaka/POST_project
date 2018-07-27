import argparse
import os
import numpy as np

# import time
# import tensorflow as tf


def parse_cmdline():
    # np.random.seed(int(time.time()))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', type=str, default='debug', help='')

    parser.add_argument('--model_name', type=str, default='stags', help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--pos_model_name', type=str, default=None, help='')

    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--steps_per_ckpt', default=2000, type=int)
    parser.add_argument('--grad_clip', action='store_true', help='')
    parser.add_argument('--grad_norm', default=5.0, type=float)
    parser.add_argument('--opt_fn', type=str, default='sgd', help='')


    parser.add_argument('--dim_word', default=64, type=int) # maybe too small: 256
    parser.add_argument('--dim_tag', default=64, type=int)  # plenty
    parser.add_argument('--dim_char', default=32, type=int) # plenty
    parser.add_argument('--dim_pos', default=64, type=int)  # plenty

    parser.add_argument('--h_word', default=64, type=int)
    parser.add_argument('--h_tag', default=64, type=int)    # maybe too small: 256
    parser.add_argument('--h_char', default=32, type=int)
    parser.add_argument('--h_pos', default=64, type=int)

    parser.add_argument('--use_c_embed', action='store_true', help='')
    parser.add_argument('--attn', action='store_true', help='')
    parser.add_argument('--use_subset', action='store_true', help='')
    parser.add_argument('--batch', type=int, default=32, help='')

    parser.add_argument('--no_val_gap', action='store_true', help='')
    parser.add_argument('--reverse', action='store_true', help='')

    parser.add_argument('--beam_size', type=int, default=5, help='')
    parser.add_argument('--beam_timesteps', type=int, default=30, help='')
    parser.add_argument('--time_out', type=float, default=100., help='')
    parser.add_argument('--num_goals', type=int, default=1, help='')

    # parser.add_argument('--load_from_file', type=str, default=None, help='')

    # parser.add_argument('--tesmin', default=0, type=int)
    # parser.add_argument('--tesmax', default=np.inf, type=int)
    # parser.add_argument('--dev_min', default=0, type=int)
    # parser.add_argument('--dev_max', default=np.inf, type=int)

    config = dict()

    args = parser.parse_args()
    config['mode'] = args.mode
    config['pos'] = args.pos
    config['scope_name'] = 'pos_model' if args.pos else 'stag_model'
    config['model_name'] = args.model_name

    config['result_dir'] = os.path.join(os.getcwd(), 'results', args.model_name)
    config['ckpt_dir'] = os.path.join(config['result_dir'], 'checkpoints')

    config['use_pretrained_pos'] = args.pos_model_name != None

    if not config['pos'] and config['use_pretrained_pos']:
        pos_model_path = os.path.join(os.getcwd(), 'results', args.pos_model, 'checkpoints')
        config['frozen_graph_fname'] = os.path.join(pos_model_path,'frozen_model.pb')
        # config['pos_ckpt'] = tf.train.latest_checkpoint(pos_model_path)

    config['btch'] = {}
    config['btch']['batch_size'] = args.batch
    config['btch']['tags_type'] = {'reverse' : args.reverse, 'no_val_gap': args.no_val_gap}

    config['btch']['dir_range'] = {'train': (2,22), 'dev': (22,23), 'test': (23,24)}

    config['btch']['nsize'] = {'tags':0, 'words': 0, 'chars':0}
    config['btch']['src_dir'] = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
    config['btch']['data_file'] = 'at_data.out'
    # config['batch_file'] = os.path.join(config['result_dir'],'batch.pickle')
    # config['subset_file'] = os.path.join(config['result_dir'],'sub_batch.json')
    config['batch_file'] = os.path.join(os.getcwd(), 'batcher', 'batch_nvg_r.pickle')
    config['use_subset'] = args.use_subset
    config['subset_file'] = os.path.join(os.getcwd(), 'batcher','sub_batch.json')

    #embedding size
    config['dim_word'] = args.dim_word
    config['dim_tag'] = args.dim_tag
    config['dim_char'] = args.dim_char
    config['dim_pos'] = args.dim_pos
    #NN dims
    config['hidden_char'] = args.h_char
    config['hidden_pos'] = args.h_pos
    config['hidden_word'] = args.h_word
    config['hidden_tag'] = args.h_tag

    #model arch
    config['use_c_embed'] = args.use_c_embed
    config['attn'] = args.attn

    if config['mode'] == 'train':
        config['lr'] = args.lr
        config['opt_fn'] = args.opt_fn
        config['num_epochs'] = args.num_epochs
        config['steps_per_ckpt'] = args.steps_per_ckpt
        config['grad_clip'] = args.grad_clip
        config['grad_norm'] = args.grad_norm

    elif config['mode'] == 'test' or config['mode'] == 'dev':
        config['beam_size'] = args.beam_size
        config['beam_timesteps'] = args.beam_timesteps
        config['num_goals'] = args.num_goals
        config['time_out'] = args.time_out

    return config
