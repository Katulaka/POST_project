from __future__ import print_function

import os
import json
import time

from vocab import Vocab
from tags.ptb_tags_convert import gen_tags
# from utils.tags.tree_t import TreeT
# from nltk.tree import Tree as Tree_
from tags.tag_ops import TagOp
from utils import flatten_to_1D, _select


def load_data(src_dir, data_file, get_fn):
    """ """
    start_time = time.time()
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        data = get_fn(src_dir, data_file)
    else:
        print ("[[gen_dataset/load_data:]] File used: %s" % data_file)
        with open (data_file, 'r') as outfile:
            data = json.load(outfile)
    print("[[gen_dataset/load_data:]] %.3fs to load data" % (time.time()-start_time))
    return data

# def get_dependancies(fin, penn_path='code/utils/tags/pennconverter.jar'):
#
#     dep_dict_file = []
#     dep_dict_tree = {}
#     lines = os.popen("java -jar "+penn_path+"< "+fin+" -splitSlash=false").read().split('\n')
#
#     for line in lines:
#         words = line.split()
#         if words:
#             dep_dict_tree[int(words[0])] = int(words[6])
#         else:
#             dep_dict_file.append(dep_dict_tree)
#             dep_dict_tree = {}
#
#     return dep_dict_file
#
# def gen_tags(fin):
#
#     tree_deps = get_dependancies(fin)
#     with open(fin) as f:
#         # lines = [x.strip('\n') for x in f.readlines()]
#         for i, line in enumerate(f.readlines()):
#
#             line = line.strip('\n')
#             #max_id is the number of words in line + 1.
#             # This is index kept in order to number words from 1 to num of words
#             max_id = len(Tree_.fromstring(line).leaves()) + 1
#             line = line.replace('(', ' ( ').replace(')', ' ) ').split()
#             tree = TreeT()
#             tree.from_ptb_to_tree(line, max_id)
#             tree.add_height(tree_deps[i])
#             return tree.from_tree_to_tag()

def data_to_dict(src_dir, data_file):
    """ If src dir is empty or not a file will result in empty file """
    all_data = dict()
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit():
            data = {'words' : [], 'tags' : []}
            for fname in sorted(filenames):
                data_in = os.path.join(directory, fname)
                print("[[load_data/data_to_dict:]] Reading file %s" %(data_in))
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
                print("[[load_data/gold_to_list:]] Reading file %s" %(data_file))
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
    print("[[gen_dataset/split_dataset:]] %.3fs to split into train/dev/test" %
                                        (time.time()-start_time))
    return dataset

def slice_dataset(dataset, ds_range, key):
    """ """
    start_time = time.time()
    indeces = [ds_range[0] <= len(w) <= ds_range[1] for w in dataset['words']]
    dataset['words'] = _select(dataset['words'], indeces)
    dataset['tags'] = _select(dataset['tags'], indeces)
    print("[[gen_dataset/slice_dataset(%s):]] %.3fs to keep sentences in range (%s, %s)" %
                 (key, time.time()-start_time, ds_range[0], ds_range[1]))
    return dataset, indeces

def gen_dataset(config, tags_type, ds_range, nwords=0, ntags=0, nchars=0):
    """ """
    data = load_data(config.src_dir, config.ds_file, data_to_dict)
    #split dataset into train/dev/test
    ds = split_dataset(data)
    tags = dict()
    indeces = dict()
    for key in ds.keys():
        ds[key], indeces[key] = slice_dataset(ds[key], ds_range[key], key)
        tags[key] = ds[key]['tags']

    gd =  load_data(config.src_dir, config.gold_file, gold_to_list)
    gd = _select(gd, indeces['test'])

    start_time = time.time()
    t_op = TagOp(*tags_type)
    for d in ds.values():
        d['tags'] = t_op.modify_fn(d['tags'])
        d['chars'] = [[list(w) for w in s] for s in d['words'] ]
    print ("[[gen_dataset:]] %.3fs to modify tags" % (time.time()-start_time))

    vocab = get_vocab(ds, nwords, ntags, nchars)

    start_time = time.time()
    for d in ds.values():
        for k in d.keys():
            d[k] = vocab[k].to_ids(d[k])
    print ("[[gen_dataset:]] %.3fs to convert data from tokens to ids" %
            (time.time()-start_time))
    return vocab, ds, t_op, tags, gd

def _gen_dataset(config, nwords=0, ntags=0, nchars=0):
    """ """
    data = load_data(config['src_dir'], config['ds_file'], data_to_dict)
    #split dataset into train/dev/test
    ds = split_dataset(data)
    tags = dict()
    indeces = dict()
    for key in ds.keys():
        ds[key], indeces[key] = slice_dataset(ds[key], config['ds_range'][key], key)
        tags[key] = ds[key]['tags']

    gd = load_data(config['src_dir'], config['gold_file'], gold_to_list)
    gd = _select(gd, indeces['test'])

    start_time = time.time()
    t_op = TagOp(*config['tags_type'])
    for d in ds.values():
        d['tags'] = t_op.modify_fn(d['tags'])
        d['chars'] = [[list(w) for w in s] for s in d['words'] ]
    print ("[[gen_dataset:]] %.3fs to modify tags" % (time.time()-start_time))

    vocab = get_vocab(ds, nwords, ntags, nchars)

    start_time = time.time()
    for d in ds.values():
        for k in d.keys():
            d[k] = vocab[k].to_ids(d[k])
    print ("[[gen_dataset:]] %.3fs to convert data from tokens to ids" %
            (time.time()-start_time))
    return vocab, ds, t_op, tags, gd

def get_vocab(ds, nwords, ntags, nchars):
    """ """
    vocab = dict()
    start_time = time.time()
    words = flatten_to_1D(ds['train']['words'])
    vocab['words'] = Vocab(words, nwords)
    print ("[[gen_dataset/get_vocab:]] %.3fs for words vocab" %
                            (time.time()-start_time))

    start_time = time.time()
    tags = flatten_to_1D([d['tags'] for d in ds.values()])
    vocab['tags'] = Vocab(tags, ntags)
    print ("[[gen_dataset/get_vocab:]] %.3fs fors tags vocab" %
                            (time.time()-start_time))

    start_time = time.time()
    chars = flatten_to_1D(ds['train']['chars'])
    vocab['chars'] = Vocab(chars, nchars)
    print ("[[gen_dataset/get_vocab:]] %.3fs for chars vocab" %
                                (time.time()-start_time))

    return vocab
