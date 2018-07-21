import tensorflow as tf
import os

from parse_cmdline import parse_cmdline
# from dataset import Dataset
from batcher import Batcher

from tree_t import gen_aug_tags
from tag_ops import TagOp
from astar.search import solve_tree_search

from model.pos_model import POSModel
from model.stag_model import STAGModel


def main(_):
    config = parse_cmdline()
    import os
    if (config['mode'] == 'debug'):
        # data = dict()
        src_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
        lb_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/loop_back'

        pdir = '~/Berkeley/Research/Tagger'
        evalb = os.path.join(pdir, 'EVALB', 'evalb')
        pfile = os.path.join(pdir, 'EVALB', 'COLLINS.prm')
        t_op = TagOp(**config['tags_type'])
        for directory, dirnames, filenames in os.walk(src_dir):
            # if directory[-1].isdigit() and directory[-2:] not in ['00','01','24']:
            if directory[-1].isdigit() and directory[-2:] in ['15']:

                dir_idx = directory.split('/')[-1]
                lb_c_dir = os.path.join(lb_dir, dir_idx)
                try:
                    if not os.path.exists(lb_c_dir):
                        os.makedirs(lb_c_dir)
                except OSError:
                    print ('Error: Creating directory. ' +  lb_c_dir)

                import cProfile

                pr = cProfile.Profile()
                pr.enable()
                for fname in sorted(filenames):
                    fin = os.path.join(directory, fname)
                    fout_lb = os.path.join(lb_c_dir, fname.split('.')[0]+'.lb')
                    res = gen_aug_tags(fin)
                    reconst = []
                    for tags, words in zip(res['tags'], res['words']):
                        mod_tags = [[t] for t in t_op.combine_fn(t_op.modify_fn(tags))]
                        score = [[1.]]*len(tags)
                        tag_score_mat = map(lambda x, y: zip(x, y), mod_tags, score)
                        trees_res,_ = solve_tree_search(tag_score_mat, words,
                        config['tags_type']['no_val_gap'], 1, 100)
                        try:
                            reconst.append(trees_res[0].from_tree_to_ptb())
                        except:
                            reconst.append('')
                    pr.disable()
                    pr.dump_stats(os.path.join(lb_c_dir, fname.split('.')[0]+'.pr'))
                    with open(fout_lb, 'w') as fout:
                        for rc in reconst:
                            fout.write('%s\n' % rc)
                    rfile = os.path.join(lb_c_dir, fname.split('.')[0]+'.evalb')
                    os.popen('%s -p %s %s %s > %s' % (evalb, pfile, fin, fout_lb, rfile))

    elif (config['mode'] == 'train'):
        batcher = Batcher(**config['btch'])
        data = batcher.load_fn()
        batcher.create_dataset(data, config['mode'])
        for k in batcher._vocab.keys():
            config['n'+k] = batcher._nsize[k]
        model = POSModel(config) if config['pos'] else STAGModel(config)
        model.train(batcher)

    elif (config['mode'] == 'dev'):
        pass


        for k in batcher._vocab.keys():
            config['n'+k] = batcher._nsize[k]
        model = POSModel(config) if config['pos'] else STAGModel(config)


if __name__ == "__main__":
    tf.app.run()
