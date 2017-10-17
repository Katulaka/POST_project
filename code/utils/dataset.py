from __future__ import print_function

import os
import numpy as np
from itertools import chain
import json
import time

from vocab import Vocab
from gen_tags import gen_tags, TagOp
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

def convert_data_flat(src_dir, data_file):
    """ If src dir is empty or not a file will result in empty file """
    # Download raw data for training #TODO
    # if not os.path.exists(src_dir):
    #      get_raw_data(src_dir)

    data = dict()
    data['words'] =[]
    data['tags'] = []
    for directory, _, filenames in os.walk(src_dir):
        for fname in filenames:
            data_in = os.path.join(directory, fname)
            print("Reading file %s" %(data_in))
            for tags, words in gen_tags(data_in):
                data['words'].append(words)
                data['tags'].append(tags)

    with open(data_file, 'w') as outfile:
        json.dump(data, outfile)

    return data


def gen_dataset(src_dir, data_file, tags_type, w_vocab_size=0, t_vocab_size=0,
                max_len=np.inf):

    start_time = time.time()
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        data = convert_data_flat(src_dir, data_file)
    else:
        print ("Data file used: %s" % data_file)
        with open (data_file, 'r') as outfile:
            data = json.load(outfile)
    print("Total time to load data: %f" % (time.time()-start_time))
    dataset = dict()

    _select = lambda A, i: list(np.array(A)[i])
    words = data['words']
    w_vocab = Vocab(flatten_to_1D(words), w_vocab_size)
    indeces = [len(w) <= max_len for w in words]
    dataset['words'] = w_vocab.to_ids(_select(words, indeces))
    print ("Time to get word data %f" % (time.time()-start_time))
    t_op = TagOp(*tags_type)
    tags = _select(data['tags'], indeces)
    _tags = t_op.modify_fn(data['tags'])
    print ("Time to modify tags %f" % (time.time()-start_time))
    t_vocab = Vocab(flatten_to_1D(_tags), t_vocab_size)
    print ("Time for tag vocab %f" % (time.time()-start_time))
    dataset['tags'] = t_vocab.to_ids(_select(_tags, indeces))
    print ("Time to get tag data %f" % (time.time()-start_time))
    return w_vocab, t_vocab, dataset, t_op, tags
