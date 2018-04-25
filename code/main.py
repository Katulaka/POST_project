import tensorflow as tf
import numpy as np
import time
import os
import json
import datetime

from model.pos_model import POSModel
from model.stag_model import STAGModel
from utils.batcher import Batcher
from utils.dataset import Dataset
from utils.parse_cmdline import parse_cmdline
from utils.tags.tree_t import trees_to_ptb

def main(_):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = parse_cmdline()
    print('==================================================================')
    if (config['mode'] in ['train', 'decode', 'stats']):
        # create vocabulary and array of dataset from train file
        ds = Dataset(config['ds'])
        data = ds.gen_dataset()
        print('==================================================================')
        # initializing batcher class
        batcher = Batcher(**config['btch'])

        for k in ds.vocab.keys():
            config['n'+k] = ds.nsize[k]
        config['npos'] = config['ntags']
        model = POSModel(config) if config['pos'] else STAGModel(config)
        print('==================================================================')

    for k,v in config.items():
        print ('[[Model Params]] %s: %s' % (k, v))

    if (config['mode'] == 'train'):
        print('==================================================================')
        print("[[POST]] Starting model training.")
        model.train(batcher, ds.dataset)

    elif (config['mode'] == 'decode'):
        print('==================================================================')
        print("[[POST]] Starting model decodeing")
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        test_min, test_max = config['ds']['ds_range']['test']
        # fname = '_'.join(['ds', str(test_min), str(test_max), now])
        fname = now
        d_name = '_'.join(['ds', str(test_min), str(test_max)])
        dir_name = os.path.join('results', config['model_name'], config['mode'], d_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        batcher.use_data(ds.dataset['test'])
        decoded = model.decode(ds.vocab, batcher, ds.t_op)
        # import pdb; pdb.set_trace()
        pattern = np.array(decoded)[:,1].tolist()
        pattern = [[p.tolist() for p in pp] for pp in pattern]
        pattern_file = os.path.join(dir_name, fname + '.ptrn')
        with open(pattern_file, 'w') as outfile:
            json.dump(pattern, outfile)

        decode_trees = np.array(decoded)[:,0].tolist()
        decode_tags = trees_to_ptb(decode_trees)
        dec_file = os.path.join(dir_name, fname + '.test')
        with open(dec_file, 'w') as outfile:
            # json.dump(decode_tags, outfile)
            for dtag in decode_tags:
                outfile.write("%s\n" % dtag)

        gold = ds.gen_gold()
        gold_file = os.path.join(dir_name, fname + '.gold')
        with open(gold_file, 'w') as outfile:
            # json.dump(gold, outfile)
            for g in gold:
                outfile.write("%s\n" % g)

    elif (config['mode'] == 'stats'):

        batcher.use_data(ds.dataset['test'])
        stats, stats_mod, stats_out = model.stats(batcher, ds.vocab)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        dir_name = os.path.join('code', 'plot')
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        fname = os.path.join(dir_name, '_'.join(['data', now]))
        with open(fname, 'w') as outfile:
            data = {'stats':stats, 'stats_mod':stats_mod, 'stats_out': stats_out}
            json.dump(data, outfile)

    elif(config['mode'] == 'evalb'):
        pdir = '~/Berkeley/Research/Tagger'
        evalb = os.path.join(pdir, 'EVALB', 'evalb')
        pfile = os.path.join(pdir, 'EVALB', 'COLLINS.prm')
        ddir = os.path.join('results', '27_3_model', 'decode')
        gfile = os.path.join(ddir, 'ds_51_60', '2018-04-01_06_35.gold')
        tfile = os.path.join(ddir, 'ds_51_60', '2018-04-01_06_35.test')

        ggfile = os.path.join(ddir, 'ds_51_60', '51_60.gld')
        ttfile = os.path.join(ddir, 'ds_51_60', '51_60.tst')
        efile = os.path.join(ddir, 'ds_51_60', '51_60.eval')

        ptfile = os.path.join(ddir, 'ds_51_60', '2018-04-01_06_19.ptrn')

        # with open(gfile, 'r') as outfile:
        #     gold = json.load(outfile)
        # with open(ggfile, 'w') as outfile:
        #     for g in gold:
        #         outfile.write("%s\n" % g)
        # with open(tfile, 'r') as outfile:
        #     test = json.load(outfile)
        # with open(ttfile, 'w') as outfile:
        #     for t in test:
        #         try:
        #             outfile.write("%s\n" % t[0])
        #         except:
        #             outfile.write("\n")

        os.popen('%s -p %s %s %s > %s' % (evalb, pfile, ggfile, ttfile, efile))

    else:
        print("Nothing to do!!")


if __name__ == "__main__":
    tf.app.run()
