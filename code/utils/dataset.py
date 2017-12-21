from __future__ import print_function

import os
import numpy as np
from itertools import chain
import json
import time

from vocab import Vocab
from tags.trees_to_tags import gen_tags
from tags.tag_ops import  TagOp
from utils import flatten_to_1D

def get_raw_data(data_path):

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


def get_dataset(src_dir, data_file):

    start_time = time.time()
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        data = data_to_dict(src_dir, data_file)
    else:
        print ("Data file used: %s" % data_file)
        with open (data_file, 'r') as outfile:
            data = json.load(outfile)
    print("Total time to load data: %f" % (time.time()-start_time))
    return data

def split_dataset(data):

    start_time = time.time()
    dataset = {'train' : {'words': [], 'tags': []},
                'dev': data[str(22)],
                'test': data[str(23)]}
    for idx in xrange(2,22):
        dataset['train']['words'].extend(data[str(idx)]['words'])
        dataset['train']['tags'].extend(data[str(idx)]['tags'])
    print("Total time to split data into train, dev,test: %f" %
                                        (time.time()-start_time))
    return dataset

def get_vocab(dataset, w_vocab_size, t_vocab_size):

    start_time = time.time()
    w_vocab = Vocab(flatten_to_1D(dataset['train']['words']), w_vocab_size)
    print ("Total time for word vocab %f" % (time.time()-start_time))
    _tags = []
    for ds in dataset.values(): #TODO
        _tags.extend(flatten_to_1D(ds['tags']))
    t_vocab = Vocab(_tags, t_vocab_size)
    print ("Total time for tag vocab %f" % (time.time()-start_time))
    return w_vocab, t_vocab

def _select(A,i):
    return list(np.array(A)[i])

def _slice_dataset(dataset, min_len, max_len):

    start_time = time.time()
    indeces = [min_len <= len(w) <= max_len for w in dataset['words']]
    dataset['words'] = _select(dataset['words'], indeces)
    dataset['tags'] = _select(dataset['tags'], indeces)
    print("Total time to slice sentences : %f" % (time.time()-start_time))
    return dataset, indeces


def gen_dataset(src_dir, data_file, tags_type, min_len, max_len,
                w_vocab_size=0, t_vocab_size=0,):

    # dataset = split_dataset(get_dataset(src_dir, data_file))
    # gold_data = get_gold('../gold_data', 'data/gold') #TODO mv the gold file to github
    dataset = split_dataset(get_data_from_file(src_dir, data_file, data_to_dict))
    gold_data =  get_data_from_file('../gold_data', 'data/gold', gold_to_list)
    tags = dict()
    indeces = dict()
    for key in dataset.keys():
        dataset[key], indeces[key] = _slice_dataset(dataset[key], min_len[key], max_len[key])
        tags[key] = dataset[key]['tags']
    gold_data = _select(gold_data, indeces['test']) #TODO expose this to upper levels
    start_time = time.time()
    t_op = TagOp(*tags_type)
    for ds in dataset.values():
        ds['tags'] = t_op.modify_fn(ds['tags'])
    print ("Total time to modify tags %f" % (time.time()-start_time))

    w_vocab, t_vocab = get_vocab(dataset, w_vocab_size, t_vocab_size)

    start_time = time.time()
    for ds in dataset.values():
        ds['words'] = w_vocab.to_ids(ds['words'])
        ds['tags'] = t_vocab.to_ids(ds['tags'])
    print ("Total time to convert data from tokens to ids %f" %
            (time.time()-start_time))
    return w_vocab, t_vocab, dataset, t_op, tags

def gold_to_list(gold_src_dir, gold_file):
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

def get_data_from_file(src_dir, data_file, get_fn):

    start_time = time.time()
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        data = get_fn(src_dir, data_file)
    else:
        print ("Data file used: %s" % data_file)
        with open (data_file, 'r') as outfile:
            data = json.load(outfile)
    print("Total time to load data: %f" % (time.time()-start_time))
    return data
