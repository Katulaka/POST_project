from __future__ import print_function

import os
import numpy as np
from itertools import chain
import json
import time

from vocab import Vocab
from tags.trees_to_tags import gen_tags
from tags.tag_ops import  TagOp
from utils import flatten_to_1D, _select

def get_raw_data(data_path):
    """ """
    print("Getting raw data from corpora")
    if not os.path.exists(data_path):
        try:
            os.makedirs(os.path.abspath(data_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        scp_path = ("scp -r login.eecs.berkeley.edu:" +
        "/project/eecs/nlp/corpora/EnglishTreebank/wsj/* ")
        os.system(scp_path + data_path)

def get_data_from_file(src_dir, data_file, get_fn):
    """ """
    start_time = time.time()
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        data = get_fn(src_dir, data_file)
    else:
        print ("Data file used: %s" % data_file)
        with open (data_file, 'r') as outfile:
            data = json.load(outfile)
    print("Total time to load data: %f" % (time.time()-start_time))
    return data

def data_to_dict(src_dir, data_file):
    """ If src dir is empty or not a file will result in empty file """
    all_data = dict()
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit():
            data = {'words' : [], 'tags' : []}
            for fname in sorted(filenames):
                data_in = os.path.join(directory, fname)
                print("Reading file %s" %(data_in))
                for tags, words in gen_tags(data_in):
                    data['words'].append(words)
                    data['tags'].append(tags)
            all_data[int(directory[-2:])] = data
    with open(data_file, 'w') as outfile:
        json.dump(all_data, outfile)
    return all_data

def gold_to_list(gold_src_dir, gold_file):
    """ """
    gold_data = []
    for directory, dirnames, filenames in os.walk(gold_src_dir):
        if directory[-1].isdigit() and int(directory[-2:])==23:
            for fname in sorted(filenames):
                data_file = os.path.join(directory, fname)
                with open(data_file) as f:
                    gold_data += [x.strip('\n') for x in f.readlines()]
    with open(gold_file, 'w') as outfile:
        json.dump(gold_data, outfile)
    return gold_data

def split_dataset(data):
    """ """
    start_time = time.time()
    dataset = {'train' : {'words': [], 'tags': []},
                'dev': data[str(22)],
                'test': data[str(23)]}
    for idx in xrange(2,22):
        dataset['train']['words'].extend(data[str(idx)]['words'])
        dataset['train']['tags'].extend(data[str(idx)]['tags'])
    print("Total time to split data into train, dev, test: %f" %
                                        (time.time()-start_time))
    return dataset

def get_vocab(dataset, w_vocab_size, t_vocab_size):
    """ """
    start_time = time.time()
    w_vocab = Vocab(flatten_to_1D(dataset['train']['words']), w_vocab_size)
    print ("Total time for word vocab %f" % (time.time()-start_time))
    _tags = []
    for ds in dataset.values(): #TODO
        _tags.extend(flatten_to_1D(ds['tags']))
    t_vocab = Vocab(_tags, t_vocab_size)
    print ("Total time for tag vocab %f" % (time.time()-start_time))
    return w_vocab, t_vocab

def slice_dataset(dataset, ds_range):
    """ """
    start_time = time.time()
    indeces = [ds_range[0] <= len(w) <= ds_range[1] for w in dataset['words']]
    dataset['words'] = _select(dataset['words'], indeces)
    dataset['tags'] = _select(dataset['tags'], indeces)
    print("Total time to slice sentences : %f" % (time.time()-start_time))
    return dataset, indeces


def gen_dataset(config, w_vocab_size=0, t_vocab_size=0,):
    """ """
    data = get_data_from_file(config.src_dir, config.ds_file, data_to_dict)
    ds = split_dataset(data)
    gd =  get_data_from_file(config.src_dir, config.gold_file, gold_to_list)
    tags = dict()
    indeces = dict()
    for key in ds.keys():
        ds[key], indeces[key] = slice_dataset(ds[key], config.ds_range[key])
        tags[key] = ds[key]['tags']
    gd = _select(gd, indeces['test'])
    start_time = time.time()
    t_op = TagOp(*config.tags_type)
    for d in ds.values():
        d['tags'] = t_op.modify_fn(d['tags'])
    print ("Total time to modify tags %f" % (time.time()-start_time))

    w_vocab, t_vocab = get_vocab(ds, w_vocab_size, t_vocab_size)

    start_time = time.time()
    for d in ds.values():
        d['words'] = w_vocab.to_ids(d['words'])
        d['tags'] = t_vocab.to_ids(d['tags'])
    print ("Total time to convert data from tokens to ids %f" %
            (time.time()-start_time))
    return w_vocab, t_vocab, ds, t_op, tags, gd


# def get_dataset(src_dir, data_file):
#
#     start_time = time.time()
#     if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
#         data = data_to_dict(src_dir, data_file)
#     else:
#         print ("Data file used: %s" % data_file)
#         with open (data_file, 'r') as outfile:
#             data = json.load(outfile)
#     print("Total time to load data: %f" % (time.time()-start_time))
#     return data
