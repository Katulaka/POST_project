import argparse

def parse_cmdline():

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('mode', type=str, default='debug')

    parser.add_argument('--train_path', type=str, default='data/02-21.clean')
    parser.add_argument('--dev_path', type=str, default='data/22-22.clean')
    parser.add_argument('--test_path', type=str, default='data/23-23.clean')
    parser.add_argument('--batch_path', type=str, default='batcher/batch.pkl')
    parser.add_argument('--no_val_gap', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--precentage',  default=1., type=float)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--model_name', type=str, default='stags')
    parser.add_argument('--scope_name', type=str, default='stag_model')
    parser.add_argument('--result_dir', type=str, default='results')

    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--learn_rate', default=0.001, type=float)
    parser.add_argument('--steps_per_ckpt', default=0, type=int)
    parser.add_argument('--grad_clip', action='store_true')
    parser.add_argument('--grad_norm', default=5.0, type=float)
    parser.add_argument('--opt_fn', type=str, default='adam')

    parser.add_argument('--gpu_n', type=int, default=0)

    parser.add_argument('--dim_word', default=64, type=int) # maybe too small: 256
    parser.add_argument('--dim_label', default=32, type=int)  # plenty
    parser.add_argument('--dim_char', default=32, type=int) # plenty
    parser.add_argument('--dim_tag', default=32, type=int)  # plenty

    parser.add_argument('--h_word', default=128, type=int)
    parser.add_argument('--h_label', default=128, type=int)    # maybe too small: 256
    parser.add_argument('--h_char', default=32, type=int)

    parser.add_argument('--nwords', default=0, type=int)
    parser.add_argument('--npos', default=0, type=int)
    parser.add_argument('--nchars', default=0, type=int)
    parser.add_argument('--nlabels', default=0, type=int)

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
    return args
