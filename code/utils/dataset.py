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
            # data = dict()
            # data['words'] =[]
            # data['tags'] = []
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

def slice_dataset(all_dataset, max_len):

    start_time = time.time()
    dataset = {'train': dict(), 'dev': dict(), 'test': dict()}
    _select = lambda A, i: list(np.array(A)[i])
    for key, ds in all_dataset.items():
        indeces = [len(w) <= max_len for w in ds['words']]
        dataset[key]['words'] = _select(ds['words'], indeces)
        dataset[key]['tags'] = _select(ds['tags'], indeces)
    print("Total time to slice sentences : %f" % (time.time()-start_time))
    return dataset

def _slice_dataset(dataset, min_len, max_len):

    start_time = time.time()
    _select = lambda A, i: list(np.array(A)[i])
    indeces = [min_len <= len(w) <= max_len for w in dataset['words']]
    dataset['words'] = _select(dataset['words'], indeces)
    dataset['tags'] = _select(dataset['tags'], indeces)
    print("Total time to slice sentences : %f" % (time.time()-start_time))
    return dataset


def gen_dataset(src_dir, data_file, tags_type, min_len, max_len,
                w_vocab_size=0, t_vocab_size=0,):

    dataset = split_dataset(get_dataset(src_dir, data_file))
    tags = dict()
    for key in dataset.keys():
        dataset[key] = _slice_dataset(dataset[key], min_len[key], max_len[key])
        tags[key] = dataset[key]['tags']
    import pdb; pdb.set_trace()
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

# def _split_dataset(dataset, ratio):
#     train_len = int(len(dataset[dataset.keys()[0]])*(1 - ratio))
#     train = dict((k, v[:train_len]) for k, v in dataset.items())
#     test = dict((k, v[train_len:]) for k, v in dataset.items())
#     return train, test
#
# def split_to_tags_words(src_file, dst_file):
#     data = dict()
#     data['words'] =[]
#     data['tags'] = []
#     for tags, words in gen_tags(src_file):
#         data['words'].append(words)
#         data['tags'].append(tags)
#     with open(dst_file, 'w') as outfile:
#         jason.dump(data, outfile)
#     # return data
#
# def split_all(src_dir, dst_root_dir):
#     for directory, dirnames, filenames in os.walk(src_dir):
#         if directory[-1].isdigit():
#             dst_dir = os.path.join(dst_root_dir, directory[-2:])
#             if not os.path.exists(dst_dir):
#                 os.mkdir(dst_dir)
#             for fname in sorted(filenames):
#                 src_file = os.path.join(directory, fname)
#                 dst_file = os.path.join(dst_dir, fname)
#                 split_to_tags_words(src_file, dst_file)
