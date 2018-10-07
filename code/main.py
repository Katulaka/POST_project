import tensorflow as tf
import os
import pickle

from parse_cmdline import parse_cmdline
from batcher import Batcher

from model.stag_model import STAGModel

def load_batcher(args):
    import time
    start_time = time.clock()
    parms = {}
    parms['data_file'] = 'data/new.processed'

    parms['d_files'] = {'train': args.train_path,
                        'dev': args.dev_path,
                        'test': args.test_path}

    parms['tags_type'] = {'reverse' : args.reverse,
                            'no_val_gap': args.no_val_gap}

    batch_file = args.batch_path
    path = '/'.join(batch_file.split('/')[:-1])
    if not os.path.exists(batch_file) or os.path.getsize(batch_file) == 0:
        if not os.path.exists(path):
            os.makedirs(path)
        print ("Couldn't find {}\n Creating new batcher".format(batch_file))
        batcher = Batcher(**parms)
        with open(batch_file, 'wb') as output:
            pickle.dump(batcher, output, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Loading batcher from {}".format(batch_file))
        with open(batch_file, 'rb') as input:
            batcher = pickle.load(input)
    current_parms = {'batch_size': args.batch_size,
                        'precentage': args.precentage}
    batcher(**current_parms)
    print ("{:.3f} to get batcher".format(time.clock()-start_time))

    for k in batcher._vocab.keys():
        vars(args)['n'+k] = batcher._vocab[k].vocab_size()

    return batcher

def run_train(args):
    batcher = load_batcher(args)
    model = STAGModel(args)
    model.train(batcher)

def run_test(config):
    config['decode_dir'] = os.path.join(config['result_dir'],'decode')
    dec_tree_f = 'dec_to_{:.2f}_tt_{:.2f}_ccr_{:.2f}_b_{}.p'.format(args.time_out, \
                    args.time_th, args.cost_coeff_rate, args.beam_size)
    config['decode_trees_file'] = os.path.join(config['decode_dir'], dec_tree_f)
    config['astar_ranks_file'] = os.path.join(config['decode_dir'] ,'astar_ranks.p')
    config['beams_file'] = os.path.join(config['decode_dir'] ,'beams.p')
    config['beams_rank_file'] = os.path.join(config['decode_dir'] ,'ranks.p')
    config['tags_file'] = os.path.join(config['decode_dir'] ,'tags.p')
    batcher, config = load_batcher(config)
    model = STAGModel(config)
    decode_trees = model.decode('test', batcher)
    # evaluate

def run_stats(config):
    atcher, config = load_batcher(config)
    model = STAGModel(config)
    import cProfile
    profile = cProfile.Profile()

    profile.enable()
    beams, tags, beams_rank = model.stats('test', batcher)
    profile.disable()
    profile.print_stats()

    decode_trees, astar_ranks = model._decode(batcher)
    with open(config['beams_rank_file'], 'rb') as f:
        ranks = pickle.load(f)
    ranks_m1 = [[r-1 for r in rank] for rank in ranks]

def main(_):
    args = parse_cmdline()

    if (args.mode == 'train'):
        run_train(args)

    elif (args.mode == 'test'):
        run_test(config)

    elif (args.mode == 'loop_back'):
        pass

    elif (args.mode == 'stats'):
        run_stats(config)

    else:
        pass


if __name__ == "__main__":
    tf.app.run()
