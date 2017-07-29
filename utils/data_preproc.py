from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join

import collections
import numpy as np

import shutil
import copy

from utils import *


def build_dictionary(words, vocabulary_size = -1):
    count = [['PAD', -1],['GO', -1],['EOS', -1], ['UNK', -1]]
    if vocabulary_size == -1:
        count.extend(collections.Counter(words).most_common())
    else:
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in words:
        if word not in dictionary:
            unk_count += 1
    count[1][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary

def lookup_fn(sentences, dictionary):
    vector = []
    for sentence in sentences:
        vector.append(list(map(lambda word: dictionary[word], sentence)))
    return vector

def _generate_batch(data, batch_size=32):
    dn = np.array(data)
    dn = np.array_split(dn, len(dn)/batch_size)
    return dn


def generate_batch(data, batch_size=32):
    batch = dict()
    d_keys = data.keys()
    d_size = len(data[d_keys[0]])
    cond = True
    while cond: #TODO fix dataset to avoid mismatching from word and tag
        d_index = np.random.randint(d_size, size=batch_size)
        for key in d_keys:
            batch[key] = np.array(data[key])[d_index]
        lw = map(lambda x: len(x), batch['word'])
        lt = map(lambda x: len(x), batch['tag'])
        cond = len([i for i in range(len(lw)) if lw[i]!=lt[i]])>0
    return batch

def data_padding(data, mlen = 0, pad_sym=0):

    max_len = mlen if mlen>0 else len(max(data, key=len))
    for i,el in enumerate(data):
        pad_len = max_len-len(el)
        data[i] = np.lib.pad(el, (0,pad_len), 'constant', constant_values=(pad_sym))
    return data

def add_xos(data, start_token = 1, end_token = 2):
    return map(lambda x: [start_token]+x+[end_token], data)

def add_go(data, start_token = 1):
    return map(lambda x: [start_token]+x, data)

def add_eos(data, end_token = 2):
    return map(lambda x: x+[end_token], data)

def _to_onehot(vec_in, max_len, size):
    vec_out = np.zeros((max_len, size))
    vec_out[np.arange(len(vec_in)), vec_in] = 1
    return vec_out

def gen_sourcefile(text_w, text_t, max_len=10):

    indeces = [i for i in range(len(text_w)) if len(text_w[i])<=max_len]
    word_list = np.ndarray.tolist(np.array(text_w)[indeces])
    tag_list = np.ndarray.tolist(np.array(text_t)[indeces])

    with open('word_file', 'w') as f:
        f.write("\n".join([' '.join(s) for s in word_list]))

    with open('tag_file', 'w') as f:
        f.write("\n".join([' '.join(s) for s in tag_list]))

    return word_list, tag_list

def gen_dataset(w_file = 'words', t_file = 'tags', max_len=10, tvocab_size=-1, wvocab_size=-1):

    count = dict()
    dictionary = dict()
    reverse_dictionary = dict()

    text = read_file(w_file)
    text_flat = list(flatten(text))
    count['word'], dictionary['word'], reverse_dictionary['word'] = build_dictionary(text_flat, wvocab_size)

    tags = read_file(t_file)
    tags_flat = list(flatten([tag.split('+') for tree in tags for tag in tree]))
    count['tag'], dictionary['tag'], reverse_dictionary['tag'] = build_dictionary(tags_flat, tvocab_size)

    vectors = dict()
    word_list, tag_list = gen_sourcefile(text, tags, max_len)
    vectors['word'] = lookup_fn(word_list, dictionary['word'])
    tag_list_mod = [[tag.split('+') for tag in tree] for tree in tag_list]
    vectors['tag'] = map(lambda tag_list: lookup_fn(tag_list, dictionary['tag']), tag_list_mod)

    return count, dictionary, reverse_dictionary, vectors

def get_batch(train_data, tag_vocabulary_size, batch_size=32):

    def arr_dim(a): return 1 + arr_dim(a[0]) if (type(a) == list) else 0

    bv = generate_batch(train_data, batch_size)
    bv_w = copy.copy(bv['word'])
    bv_t = copy.copy(bv['tag'])
    if arr_dim(bv_t.tolist()) == 3:
        bv_t = [x for y in bv_t for x in y]

    seq_len_w = map(lambda x: len(x), bv_w)
    data_padding(bv_w)
    bv_w = np.vstack([np.expand_dims(x, 0) for x in bv_w])

    bv_t_go = add_go(bv_t)
    bv_t_eos = add_eos(bv_t)
    seq_len_t = map(lambda x: len(x), bv_t_go)
    bv_t_1hot = map(lambda x: _to_onehot(x, max(seq_len_t),
                                        tag_vocabulary_size), bv_t_eos)
    bv_t_1hot = np.vstack([np.expand_dims(x, 0) for x in bv_t_1hot])
    data_padding(bv_t_go)
    bv_t_go = np.vstack([np.expand_dims(x, 0) for x in bv_t_go])

    return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_go, bv_t_1hot
