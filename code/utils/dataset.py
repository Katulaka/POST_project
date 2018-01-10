from __future__ import print_function

import os
import json
import time

from tags.tree_t import gen_tags
from tags.tag_ops import TagOp
from utils import flatten_to_1D, _select
from vocab import Vocab


class Dataset(object):
    """docstring for ."""
    def __init__(self, config):
        for name, value in config.items():
            setattr(self, name, config[name])
        import pdb; pdb.set_trace()
        for name in ['dataset', 'vocab', 'size', 'idx']:
            setattr(self, name, dict())


    def load_fn(self, data_file, get_fn):
        """ """
        if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
            data = get_fn(data_file)
        else:
            print ("[[Dataset.load_fn]] File used: %s" % data_file)
            with open (data_file, 'r') as outfile:
                data = json.load(outfile)
        return data

    def src_to_tags_words(self, data_file):
        """ If src dir is empty or not a file will result in empty file """
        all_data = dict()
        for directory, dirnames, filenames in os.walk(self.src_dir):
            if directory[-1].isdigit():
                data = {'words' : [], 'tags' : []}
                for fname in sorted(filenames):
                    fin = os.path.join(directory, fname)
                    for tags, words in gen_tags(fin):
                        data['words'].append(words)
                        data['tags'].append(tags)
                all_data[int(directory[-2:])] = data
        with open(data_file, 'w') as outfile:
            json.dump(all_data, outfile)
        return all_data

    def src_to_gold(self, data_file):
        """ """
        # all_data = dict()
        data = []
        for directory, dirnames, filenames in os.walk(self.src_dir):
            if directory[-1].isdigit() and int(directory[-2:])==23:
                for fname in sorted(filenames):
                    fin = os.path.join(directory, fname)
                    with open(fin) as f:
                        data += [x.strip('\n') for x in f.readlines()]
            # all_data[int(directory[-2:])] = data
        with open(data_file, 'w') as outfile:
            json.dump(data, outfile)
        return data

    def load_gold(self, data_file):
        return self.load_fn(data_file, self.src_to_gold)

    def slice_gold(self, data, ds_range, ds):
        indeces = [ds_range[0] <= len(d) <= ds_range[1] for d in ds]
        return _select(data, indeces)

    def load_data(self, data_file):
        start_time = time.time()
        data = self.load_fn(data_file, self.src_to_tags_words)
        print("[[Dataset.load_data]] %.3f to load data" % (time.time()-start_time))
        return data

    def split(self, data, dir_range):
        """ """
        for k, rng in dir_range.items():
            start_time = time.time()
            self.dataset[k] = {'words': [], 'tags': []}
            for idx in xrange(*rng):
                self.dataset[k]['words'].extend(data[str(idx)]['words'])
                self.dataset[k]['tags'].extend(data[str(idx)]['tags'])
            self.size[k] = len(self.dataset[k]['words'])
            print ("[[Dataset.split]] %.5f to split into %s: %s entries" %
                            (time.time()-start_time, k, self.size[k]))

    def slice(self, ds_range):
        for k, ds in self.dataset.items():
            start_time = time.time()
            self.idx[k] = [ds_range[k][0] <= len(w) <= ds_range[k][1] for w in ds['words']]
            for d in ds.values():
                d = _select(d, self.idx[k])
            print ("[[Dataset.slice]] %.5f to keep %s data in range %s" %
                            (time.time()-start_time, k, ds_range[k]))

    def modify(self, tags_type):
        start_time = time.time()
        self.t_op = TagOp(**tags_type)
        for d in self.dataset.values():
            d['tags'] = self.t_op.modify_fn(d['tags'])
            d['chars'] = [[list(w) for w in s] for s in d['words']]
        print ("[[Dataset.modify]] Tags properties are %s" % (tags_type))
        print ("[[Dataset.modify]] %.3f to modify dataset" % (time.time()-start_time))

    def invert_dataset(self):
        #mode: train/dev/test
        #elm: words/chars/tags
        inv_ds = dict()
        for mode, ds in self.dataset.items():
            for elm in ds.keys():
                if elm in inv_ds.keys():
                    inv_ds[elm].update({mode : ds[elm]})
                else:
                    inv_ds[elm] = {mode : ds[elm]}
        return inv_ds

    def get_vocab(self, nsize):

        for elm, ids in self.invert_dataset().items():
            start_time = time.time()
            d = [d for d in ids.values()] if elm == 'tags' else ids['train']
            self.vocab[elm] = Vocab(flatten_to_1D(d), nsize[elm])
            self.nsize[elm] = self.vocab[elm].vocab_size()
            print ("[[Dataset.get_vocab]] %.3f for %s vocab (size: %s)" %
                                (time.time()-start_time, elm, self.nsize[elm]))

    def to_ids(self, nsize):
        self.get_vocab(nsize)
        for kds, ds in self.dataset.items():
            start_time = time.time()
            for k in ds.keys():
                ds[k] = self.vocab[k].to_ids(ds[k])
            print ("[[Dataset.to_ids]] %.3f for %s" % (time.time()-start_time, kds))

    def gen_dataset(self):
        start_time = time.time()
        data = self.load_data(self.ds_file)
        self.split(data, self.dir_range)
        self.slice(self.ds_range)
        self.modify(self.tags_type)
        self.to_ids(self.nsize)
        print("[[Dataset.gen_dataset]] Total time %.3f"  % (time.time()-start_time))
        return data

    def gen_gold(self):
        data = self.load_gold(self.gold_file)
        return _select(data, self.idx['test'])
