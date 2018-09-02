import argparse
# import os
# import numpy as np

def parse_cmdline():
    # np.random.seed(int(time.time()))

    parser = argparse.ArgumentParser(add_help=False)
    # subparsers = parser.add_subparsers()
    #
    # subparser = subparsers.add_parser("train")
    # subparser.set_defaults(callback=run_train)

    parser.add_argument('mode', type=str, default='debug')

    parser.add_argument('--train_path', type=str, default='../data/02-21.clean')
    parser.add_argument('--dev_path', type=str, default='../data/22-22.clean')
    parser.add_argument('--test_path', type=str, default='../data/23-23.clean')
    parser.add_argument('--batch_path', type=str, default='../batcher/batch.pkl')
    parser.add_argument('--no_val_gap', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--model_name', type=str, default='stags')
    parser.add_argument('--scope_name', type=str, default='stag_model')
    parser.add_argument('--result_dir', type=str, default='../results')

    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--learn_rate', default=0.001, type=float)
    parser.add_argument('--steps_per_ckpt', default=0, type=int)
    parser.add_argument('--grad_clip', action='store_true')
    parser.add_argument('--grad_norm', default=5.0, type=float)
    parser.add_argument('--opt_fn', type=str, default='adam')

    parser.add_argument('--gpu_n', type=int, default=0)

    parser.add_argument('--dim_word', default=64, type=int) # maybe too small: 256
    parser.add_argument('--dim_tag', default=32, type=int)  # plenty
    parser.add_argument('--dim_char', default=32, type=int) # plenty
    parser.add_argument('--dim_pos', default=32, type=int)  # plenty

    parser.add_argument('--h_word', default=128, type=int)
    parser.add_argument('--h_tag', default=128, type=int)    # maybe too small: 256
    parser.add_argument('--h_char', default=32, type=int)
    parser.add_argument('--h_pos', default=64, type=int)

    parser.add_argument('--nwords', default=0, type=int)
    parser.add_argument('--ntags', default=0, type=int)
    parser.add_argument('--nchars', default=0, type=int)
    parser.add_argument('--npos', default=0, type=int)

    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--no_c_embed', action='store_true')
    parser.add_argument('--no_attn', action='store_true')
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--is_stack', action='store_true')
    parser.add_argument('--kp_bidi', type=float, default=0.8)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--is_add_elmo', action='store_true')
    parser.add_argument('--elmo_dim', type=int, default=256)

    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--beam_timesteps', type=int, default=30)
    parser.add_argument('--time_out', type=float, default=100.)
    parser.add_argument('--time_th', type=float, default=10.)
    parser.add_argument('--cost_coeff_rate', type=float, default=0.5)
    parser.add_argument('--num_goals', type=int, default=1)

    # parser.add_argument('--load_from_file', type=str, default=None)

    args = parser.parse_args()
    # mode = args.mode
    # config = dict()
    # config['gpu_n'] =args.gpu_n
    # config['scope_name'] = 'stag_model'
    # config['model_name'] = args.model_name
    # #embedding size
    # config['dim_word'] = args.dim_word
    # config['dim_tag'] = args.dim_tag
    # config['dim_char'] = args.dim_char
    # config['dim_pos'] = args.dim_pos
    # #NN dims
    # config['hidden_char'] = args.h_char
    # config['hidden_pos'] = args.h_pos
    # config['hidden_word'] = args.h_word
    # config['hidden_tag'] = args.h_tag
    # #model arch
    # config['drop_rate'] = args.drop_rate
    # config['no_c_embed'] = args.no_c_embed
    # config['no_attn'] = args.no_attn
    # config['layer_norm'] = args.layer_norm
    # config['is_stack'] = args.is_stack
    # config['kp_bidi'] = args.kp_bidi
    # config['n_layers'] = args.n_layers
    # config['is_add_elmo'] = args.is_add_elmo
    # config['elmo_dim'] = args.elmo_dim
    #
    # if  mode == 'train':
    #     config['lr'] = args.learn_rate
    #     config['opt_fn'] = args.opt_fn
    #     config['num_epochs'] = args.num_epochs
    #     config['steps_per_ckpt'] = args.steps_per_ckpt
    #     config['grad_clip'] = args.grad_clip
    #     config['grad_norm'] = args.grad_norm
    # else:
    #     config['beam_size'] = args.beam_size
    #     config['beam_timesteps'] = args.beam_timesteps
    #     config['num_goals'] = args.num_goals
    #     config['time_out'] = args.time_out
    #     config['time_th'] = args.time_th
    #     config['cost_coeff_rate'] = args.cost_coeff_rate

    return args
