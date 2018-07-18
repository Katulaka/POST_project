import argparse
import os
import numpy as np

# import json
# import time
# import tensorflow as tf


def parse_cmdline():
    # np.random.seed(int(time.time()))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default='stags', help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--pos_model_name', type=str, default=None, help='')

    # parser.add_argument('--load_from_file', type=str, default=None, help='')

    # parser.add_argument('--test_min', default=0, type=int)
    # parser.add_argument('--test_max', default=np.inf, type=int)
    # parser.add_argument('--dev_min', default=0, type=int)
    # parser.add_argument('--dev_max', default=np.inf, type=int)


    subparsers = parser.add_subparsers(help='types of action')

    t_parser = subparsers.add_parser('train', parents=[parser])
    t_parser.add_argument('--num_epochs', default=1, type=int)
    t_parser.add_argument('--lr', default=0.01, type=float)
    t_parser.add_argument('--steps_per_ckpt', default=10, type=int)

    t_parser.add_argument('--dim_word', default=64, type=int)
    t_parser.add_argument('--dim_tag', default=64, type=int)
    t_parser.add_argument('--dim_char', default=32, type=int)
    t_parser.add_argument('--dim_pos', default=64, type=int)

    t_parser.add_argument('--h_word', default=64, type=int)
    t_parser.add_argument('--h_tag', default=64, type=int)
    t_parser.add_argument('--h_char', default=32, type=int)
    t_parser.add_argument('--h_pos', default=64, type=int)

    t_parser.add_argument('--use_c_embed', action='store_true', help='')
    t_parser.add_argument('--attn', action='store_true', help='')
    t_parser.add_argument('--batch', type=int, default=32, help='')

    t_parser.add_argument('--no_val_gap', action='store_true', help='')
    t_parser.add_argument('--reverse', action='store_true', help='')

    # t_parser.add_argument('--train_min', default=0, type=int)
    # t_parser.add_argument('--train_max', default=np.inf, type=int)

    d_parser = subparsers.add_parser('decode', parents=[parser])
    d_parser.add_argument('--no_val_gap', action='store_true', help='')
    d_parser.add_argument('--reverse', action='store_true', help='')
    d_parser.add_argument('--batch', type=int, default=32, help='')

    d_parser.add_argument('--dim_word', default=64, type=int)
    d_parser.add_argument('--dim_tag', default=64, type=int)
    d_parser.add_argument('--dim_char', default=32, type=int)
    d_parser.add_argument('--dim_pos', default=64, type=int)

    d_parser.add_argument('--h_word', default=64, type=int)
    d_parser.add_argument('--h_tag', default=64, type=int)
    d_parser.add_argument('--h_char', default=32, type=int)
    d_parser.add_argument('--h_pos', default=64, type=int)

    d_parser.add_argument('--use_c_embed', action='store_true', help='')
    d_parser.add_argument('--attn', action='store_true', help='')
    d_parser.add_argument('--beam_size', type=int, default=5, help='')
    d_parser.add_argument('--beam_timesteps', type=int, default=25, help='')
    d_parser.add_argument('--time_out', type=float, default=100., help='')
    d_parser.add_argument('--num_goals', type=int, default=1, help='')

    b_parser = subparsers.add_parser('evalb', parents=[parser])

    dbg_parser = subparsers.add_parser('debug', parents=[parser])
    dbg_parser.add_argument('--no_val_gap', action='store_true', help='')
    dbg_parser.add_argument('--reverse', action='store_true', help='')



    config = dict()

    import sys
    config['mode'] = sys.argv[1]
    if config['mode'] == 'train':
        current_parser =  t_parser
    elif config['mode'] == 'decode':
        current_parser =  d_parser
    elif config['mode'] == 'evalb':
        current_parser =  b_parser
    else:
        current_parser =  dbg_parser

    args = current_parser.parse_args()
    config['tags_type'] = {'reverse' : args.reverse,
                            'no_val_gap': args.no_val_gap}

    if config['mode'] == 'train':
        config['pos'] = args.pos
        config['num_epochs'] = args.num_epochs
        config['model_name'] = args.model_name
        config['result_dir'] = os.path.join(os.getcwd(), 'results', args.model_name)
        config['ckpt_dir'] = os.path.join(config['result_dir'], 'checkpoints')
        config['steps_per_ckpt'] = args.steps_per_ckpt
        config['scope_name'] = 'pos_model' if args.pos else 'stag_model'
        config['lr'] = args.lr

        config['use_pretrained_pos'] = args.pos_model_name != None

        if not config['pos'] and config['use_pretrained_pos']:
            pos_model_path = os.path.join(os.getcwd(), 'results', args.pos_model, 'checkpoints')
        # config['pos_ckpt'] = tf.train.latest_checkpoint(pos_model_path)
            config['frozen_graph_fname'] = os.path.join(pos_model_path,'frozen_model.pb')

        config['btch'] = {}
        config['btch']['batch_size'] = args.batch
        config['btch']['tags_type'] = {'reverse' : args.reverse,
                                'no_val_gap': args.no_val_gap}

        config['btch']['dir_range'] = {'train': (2,22),
                                'dev': (22,23),
                                'test': (23,24)}

        config['btch']['nsize'] = {'tags':0, 'words': 0, 'chars':0}

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

        config['src_dir'] = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
        config['at_fout'] = 'at_data.out'

    elif config['mode'] == 'decode':

        config['model_name'] = args.model_name
        config['result_dir'] = os.path.join(os.getcwd(), 'results', args.model_name)
        config['ckpt_dir'] = os.path.join(config['result_dir'], 'checkpoints')
        config['pos'] = args.pos
        config['scope_name'] = 'pos_model' if args.pos else 'stag_model'

        config['use_pretrained_pos'] = args.pos_model_name != None

        if not config['pos'] and config['use_pretrained_pos']:
            pos_model_path = os.path.join(os.getcwd(), 'results', args.pos_model, 'checkpoints')
        # config['pos_ckpt'] = tf.train.latest_checkpoint(pos_model_path)
            config['frozen_graph_fname'] = os.path.join(pos_model_path,'frozen_model.pb')

        config['btch'] = {}
        config['btch']['batch_size'] = args.batch
        config['btch']['tags_type'] = {'reverse' : args.reverse,
                                'no_val_gap': args.no_val_gap}

        config['btch']['dir_range'] = {'train': (2,22),
                                'dev': (22,23),
                                'test': (23,24)}

        config['btch']['nsize'] = {'tags':0, 'words': 0, 'chars':0}

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

        config['src_dir'] = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
        config['at_fout'] = 'at_data.out'

        config['beam_size'] = args.beam_size
        config['beam_timesteps'] = args.beam_timesteps
        config['num_goals'] = args.num_goals
        config['time_out'] = args.time_out


    # ds_dir = '../data'
    # ds_fname = 'data.txt'
    # gold_fname = 'gold'
    # config['ds']['ds_file'] = os.path.join(os.getcwd(), ds_dir, ds_fname)
    # config['ds']['gold_file'] = os.path.join(os.getcwd(), ds_dir, gold_fname)



    return config
