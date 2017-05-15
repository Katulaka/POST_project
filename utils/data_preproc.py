from __future__ import print_function

from os import listdir
from os.path import isfile, join

import collections
import numpy as np

import os
import shutil


def make_dir(path_dir):
    
    if not os.path.exists(path_dir):
        try:
            os.makedirs(os.path.abspath(path_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# TODO: hidden files
def del_extra_files(root_dir):

    print("[del_extra_files] Deleting mismatching files between dirs")
    
    for f in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, f)) and f not in ["{:02d}".format(x) for x in xrange(25)]:
            shutil.rmtree(os.path.join(root_dir, f))
        if os.path.isfile(os.path.join(root_dir, f)):
            os.remove(os.path.join(root_dir, f))


def get_data(data_path):

    print("[get_data] Getting RAW data from corpora")

    make_dir(data_path)
    
    if not os.path.exists(os.path.join(data_path, 'wsj')):
        os.system("scp -r login.eecs.berkeley.edu:/project/eecs/nlp/corpora/'EnglishTreebank/wsj/ " + data_path)
        del_extra_files(os.path.join(data_path, 'wsj'))


def read_file(fname):

    with open(fname) as f:
        text = f.read().splitlines()
    
    return list(map(lambda x: x.split(), text))


def flatten_list(list_):
    return sum(list_, [])
        

def build_dataset(words, vocabulary_size = 100000):
    
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
        unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data, count, dictionary, reverse_dictionary


def lookup_fn(sentences, dictionary):

    vector = []
    for sentence in sentences:
        vector.append(list(map(lambda word: dictionary[word], sentence)))
    return vector


def gen_data(path, r_min=0, r_max=1):
    
    data = {}
    count = {}
    dictionary = {}
    reverse_dictionary = {}
    sentences = {}
    vectors = {}
               
    for el in ['tags', 'words']: 
        in_list = []
        sentence_list = []
        for subdir in map(lambda n: str(n).zfill(2), range(r_min, r_max)):
            current_path = join(path, subdir, el)
            files_list = list(filter(lambda f: isfile(join(current_path, f)), listdir(current_path)))
            for f in files_list:
                text = read_file(join(current_path, f))
                sentence_list.extend(text)
                in_list.extend(flatten_list(text))
        sentences[el] = sentence_list
        data[el], count[el], dictionary[el], reverse_dictionary[el] = build_dataset(in_list)
        vectors[el] = lookup_fn(sentences[el], dictionary[el])

    del in_list    # Hint to reduce memory.
    del sentence_list

    return vectors 


def generate_batch(data, batch_size=32):
    dn = np.array(data)
    dn = np.array_split(dn, len(dn)/batch_size)
    return dn


def data_padding(data, pad_sym=0):
    
    max_len = len(max(data, key=len))
    for i,el in enumerate(data):
        pad_len = max_len-len(el)
        # import pdb; pdb.set_trace()
        data[i] = np.lib.pad(el, (0,pad_len), 'constant', constant_values=(pad_sym))
    return data
