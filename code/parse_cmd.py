import argparse
import os
import numpy as np

def parse_cmdline():
    # np.random.seed(int(time.time()))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', type=str, default='debug', help='')

    parser.add_argument('--model_name', type=str, default='stags', help='')
    parser.add_argument('--scope_name', type=str, default='stag_model', help='')
    parser.add_argument('--pos', action='store_true', help='')
    parser.add_argument('--pos_model_name', type=str, default=None, help='')

    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--steps_per_ckpt', default=0, type=int)
    parser.add_argument('--grad_clip', action='store_true', help='')
    parser.add_argument('--grad_norm', default=5.0, type=float)
    parser.add_argument('--opt_fn', type=str, default='adam', help='')

    parser.add_argument('--gpu_n', type=int, default=0)

    parser.add_argument('--dim_word', default=64, type=int) # maybe too small: 256
    parser.add_argument('--dim_tag', default=32, type=int)  # plenty
    parser.add_argument('--dim_char', default=32, type=int) # plenty
    parser.add_argument('--dim_pos', default=32, type=int)  # plenty

    parser.add_argument('--h_word', default=128, type=int)
    parser.add_argument('--h_tag', default=128, type=int)    # maybe too small: 256
    parser.add_argument('--h_char', default=32, type=int)
    parser.add_argument('--h_pos', default=64, type=int)

    parser.add_argument('--drop_rate', default=0.5, type=float)
    # parser.add_argument('--keep_prob', default=0.8, type=float)

    parser.add_argument('--no_c_embed', action='store_true', help='')
    parser.add_argument('--no_attn', action='store_true', help='')
    parser.add_argument('--layer_norm', action='store_true', help='')
    parser.add_argument('--is_stack', action='store_true', help='')
    parser.add_argument('--kp_bidi', type=float, default=0.8)
    parser.add_argument('--n_layers', type=int, default=1)
    # parser.add_argument('--affine', action='store_true', help='')
    parser.add_argument('--use_subset', action='store_true', help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--batch_file', type=str, default='batch.pkl', help='')
    parser.add_argument('--is_add_elmo', action='store_true', help='')
    parser.add_argument('--elmo_dim', type=int, default=256, help='')

    parser.add_argument('--no_val_gap', action='store_true', help='')
    parser.add_argument('--reverse', action='store_true', help='')

    parser.add_argument('--beam_size', type=int, default=5, help='')
    parser.add_argument('--beam_timesteps', type=int, default=30, help='')
    parser.add_argument('--time_out', type=float, default=100., help='')
    parser.add_argument('--time_th', type=float, default=10., help='')
    parser.add_argument('--cost_coeff_rate', type=float, default=0.5, help='')
    parser.add_argument('--num_goals', type=int, default=1, help='')

    parser.add_argument('--s_idx', type=int, default=0, help='')
    parser.add_argument('--e_idx', type=str, default=None, help='')
    # parser.add_argument('--load_from_file', type=str, default=None, help='')

    config = dict()

    args = parser.parse_args()
    config['mode'] = args.mode

    config['model_name'] = args.model_name
    config['path'] = {}
    config['path']['result'] = os.path.join(os.getcwd(), 'results', args.model_name)
    config['path']['ckpt'] = os.path.join(config['path']['result'], 'checkpoints')
    config['path']['sw'] = os.path.join(config['path']['result'], 'summary')
    config['path']['decode'] = os.path.join(config['path']['result'],'decode')
    config['path']['batcher'] = os.path.join(os.getcwd(), 'batcher')
    config['path']['src_dir'] = '~/Berkeley/Research/Tagger/'
    config['path']['gold_dir'] = os.path.join(config['path']['src_dir'], 'gold_data')
    config['path']['lb_dir'] = os.path.join(config['path']['src_dir'], 'loop_back')
    config['path']['evalb_dir'] = os.path.join(config['path']['src_dir'], 'EVALB')

    config['arch'] = {}
    config['arch']['scope_name'] = args.scope_name
    config['arch']['dim_word'] = args.dim_word
    config['arch']['dim_tag'] = args.dim_tag
    config['arch']['dim_char'] = args.dim_char
    config['arch']['dim_pos'] = args.dim_pos
    config['arch']['hidden_char'] = args.h_char
    config['arch']['hidden_pos'] = args.h_pos
    config['arch']['hidden_word'] = args.h_word
    config['arch']['hidden_tag'] = args.h_tag
    config['arch']['drop_rate'] = args.drop_rate
    config['arch']['no_c_embed'] = args.no_c_embed
    config['arch']['no_attn'] = args.no_attn
    config['arch']['layer_norm'] = args.layer_norm
    config['arch']['is_stack'] = args.is_stack
    config['arch']['kp_bidi'] = args.kp_bidi
    config['arch']['n_layers'] = args.n_layers
    config['arch']['is_add_elmo'] = args.is_add_elmo
    config['arch']['elmo_dim'] = args.elmo_dim
    config['arch']['gpu_n'] = args.gpu_n
    config['arch']['use_pretrained_pos'] = args.pos_model_name != None
    config['pos'] = args.pos
    # if not args.pos and config['arch']['use_pretrained_pos']:
    #     pos_model_path = os.path.join(os.getcwd(), 'results', args.pos_model, 'checkpoints')
    #     config['frozen_graph_fname'] = os.path.join(pos_model_path,'frozen_model.pb')

    config['btch'] = {}
    config['btch']['tags_type'] = {'reverse' : args.reverse, 'no_val_gap': args.no_val_gap}
    config['btch']['dir_range'] = {'train': (2,22), 'dev': (22,23), 'test': (23,24)}
    config['btch']['src_dir'] = config['path']['gold_data']
    config['btch']['data_f'] = os.path.join(config['path']['batcher'], 'at_data.out')

    config['btch']['batch_size'] = args.batch_size
    config['batcher_f'] = os.path.join(config['path']['batcher'], args.batch_file)

    config['btch']['s_idx'] = args.s_idx
    config['btch']['e_idx'] = args.e_idx

    config['btch']['use_subset'] = args.use_subset
    config['btch']['precentage'] = 0.1
    config['subset_f'] = os.path.join(config['path']['batch'], 'sub_batch.json')
    config['subset_dev_f'] = os.path.join(config['path']['batch'], 'sub_batch_dev.json')

    if config['mode'] == 'train':
        config['train'] = {}
        config['train']['lr'] = args.lr
        config['train']['opt_fn'] = args.opt_fn
        config['train']['num_epochs'] = args.num_epochs
        config['train']['steps_per_ckpt'] = args.steps_per_ckpt
        config['train']['grad_clip'] = args.grad_clip
        config['train']['grad_norm'] = args.grad_norm
    elif config['mode'] == 'test':
        config['beam']['size'] = args.beam_size
        config['beam']['timesteps'] = args.beam_timesteps
        config['beam']['beams_f'] = os.path.join(config['path']['decode'] ,'beams.p')
        config['beam']['ranks_f'] = os.path.join(cconfig['path']['decode'] ,'ranks.p')

        config['astar']['num_goals'] = args.num_goals
        config['astar']['time_out'] = args.time_out
        config['astar']['time_th'] = args.time_th
        config['astar']['cost_coeff_rate'] = args.cost_coeff_rate

        dec_tree_f = 'to_{:.2f}_tt_{:.2f}_ccr_{:.2f}_rng_{}_{}.p'.format(args.time_out, \
                        args.time_th, args.cost_coeff_rate, args.s_idx, args.e_idx)
        config['astar']['decode_trees_f'] = os.path.join(config['path']['decode'], dec_tree_f)
        config['astar']['ranks_f'] = os.path.join(config['path']['decode'] ,'astar_ranks.p')

    else:
        config['evalb']['gold_f'] = os.path.join(os.getcwd(),'gold.p')
        config['evalb']['script_f'] = os.path.join(config['path']['evalb_dir'],'evalb')
        config['evalb']['parms_f'] = os.path.join(config['path']['evalb_dir'],'COLLINS.prm')
        config['evalb']['res_f'] = os.path.join(config['path']['decode'], 'res.eval')
