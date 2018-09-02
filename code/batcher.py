from __future__ import print_function

import os
import time
import numpy as np
from itertools import islice

from vocab import PAD, GO, EOS, UNK, Vocab
from utils import operate_on_Narray, _operate_on_Narray, flatten_to_1D
from tag_ops import TagOp

from nltk.tree import Tree
from tree_t import get_dependancies, TreeT
import collections
import pickle


class Batcher(object):

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, '_'+k, v)

        self._t_op = TagOp(**self._tags_type)
        self._data = self.load_data()
        self._vocab = self.create_vocab()
        self.convert_to_ids()

    def __call__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, '_'+k, v)

    def create_data(self, data_file):
        """ """
        data = {}
        types = ['train', 'dev', 'test']
        fields = ['gold', 'words', 'pos', 'chars', 'tags']
        Entry = collections.namedtuple('entry', fields)
        for type, d_file in self._d_files.items():
            tree_deps = get_dependancies(d_file)
            gold, words, pos, chars, tags = [], [], [], [], []
            with open(d_file) as f:
                for line, dep in zip(f.readlines(), tree_deps):
                    line = line.strip('\n')
                    gold.append(line)
                    word_pos = Tree.fromstring(line).pos()
                    max_id = len(word_pos) + 1
                    tup_words, tup_pos = zip(*word_pos)
                    words.append(list(tup_words))
                    pos.append(list(tup_pos))
                    chars.append([list(w) for w in tup_words])
                    line = line.replace('(', ' ( ').replace(')', ' ) ').split()
                    tags.append(TreeT().from_ptb_to_tag(line, max_id, dep))
            data_dict = Entry(gold, words, pos, chars, tags)._asdict()
            data.setdefault(type,{}).update(data_dict)
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        return data

    def load_data(self):

        data_file = self._data_file
        if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
            print ("[[Batcher]] Couldn't find {}".format(data_file))
            data = self.create_data(data_file)
        else:
            print ("[[Batcher]] Loading data from {}".format(data_file))
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        for k in data.keys():
            data[k]['tags'] = [[self._t_op.modify_tag(t) for t in ts] for ts in data[k]['tags']]
        return data

    def create_vocab(self):
        """ """
        vocab_data = {}
        _vocab = {}
        for d_k, d_v in self._data.items():
            for k, v in d_v.items():
                if k != 'gold':
                    vocab_data.setdefault(k,[]).extend(v)

        for k, v in vocab_data.items():
            start_time = time.clock()
            _vocab[k] = Vocab(flatten_to_1D(v))
            print ("{:.3f} to creat {} vocab (size: {})".format(
                        time.clock()-start_time, k, _vocab[k].vocab_size()))
        return _vocab

    def convert_to_ids(self):
        """ """
        for k, v in self._vocab.items():
            start_time = time.clock()
            for d_k, d_v in self._data.items():
                self._data[d_k][k] = v.to_ids(d_v[k])
            print ("{:.3f} to convert to ids {} vocab".format(
                                                    time.clock()-start_time, k))
        return self

    def get_batch(self, mode, permute=False):

        _d_size = len(self._data[mode]['gold'])
        if self._use_subset:
            precentage = 0.1
            skip_size = int(np.ceil(1/precentage))
            sub_idx = range(_d_size)[0::skip_size]
            _d_size = len(sub_idx)

        n_batches = int(np.ceil(float(_d_size)/self._batch_size))

        batched = {}

        d_keys = list(self._data[mode].keys())
        d_keys.remove('gold')
        for k in d_keys:
            v = self._data[mode][k]
            data = np.array(v)[sub_idx].tolist(v) if self._use_subset else v
            batched.setdefault(k, np.array_split(data, n_batches))

        b_idx = np.random.permutation(n_batches) if permute else range(n_batches)
        return [{k: batched[k][i].tolist() for k in d_keys} for i in b_idx]

    def seq_len(self, batch_data):
        return operate_on_Narray(batch_data, len)

    def max_seq_len(self, batch_data):
        return operate_on_Narray(batch_data, max)

    def _pad(self, data, max_len, pad_token, l_pad_len):
        r_pad_len = max_len - len(data) - l_pad_len
        return np.lib.pad(data, (l_pad_len, r_pad_len),
                        'constant', constant_values=(pad_token)).tolist()

    def pad(self, data, max_len, pad_token=PAD[1], l_pad_len=0):
        return operate_on_Narray(data, self._pad, max_len, pad_token, l_pad_len)

    def add_pad_vec(self, data, pad_token=PAD[1]):
        return [[[pad_token]] + bv for bv in data]

    def pad_matrix(self, data, dim_1, dim_2):
        data_pad = []
        for d in data:
            d = np.array(d)
            d.resize(dim_1, dim_2)
            data_pad.append(d.tolist())
        return data_pad

    def _add_go(self, data, go_token):
        return [go_token] + data

    def add_go(self, data, go_token=GO[1]):
        return operate_on_Narray(data, self._add_go, go_token)

    def _add_eos(self, data, eos_token):
        return data+[eos_token]

    def add_eos(self, data, eos_token=EOS[1]):
        return operate_on_Narray(data, self._add_eos, eos_token)

    def _remove_go(self, data, go_token):
        idx = data.index(go_token)
        return data[idx+1:]

    def remove_go(self, data, go_token=GO[1]):
        return operate_on_Narray(data, self._remove_go, go_token)

    def _remove_eos(self, data, eos_token):
        idx = data.index(eos_token)
        return data[:idx]

    def remove_eos(self, data, eos_token=EOS[1]):
        return operate_on_Narray(data, self._remove_eos, eos_token)

    def remove_delim(self, data, go_token, eos_token):
        return remove_eos(remove_go(data, go_token), eos_token)

    def remove_delim_len(self, data):
        return [d[1:l-1].tolist() for d, l in zip(data['in'], data['len'])]

    def remove_len(self, data):
        return [d[1:l].tolist() for d, l in zip(data['in'], data['len'])]

    def remove_pad(self, data):
        return list(filter(None, self.remove_len(data)))

    def id_to_unk(self, _id, vocab):
        rnd_val = np.random.random()
        prob = 1./(1+vocab.get_count(_id))
        unk_id = vocab.get_unk_id()
        return _id if rnd_val > prob else unk_id

    def to_unks(self, bv, vocab):
        return _operate_on_Narray(bv, self.id_to_unk, vocab)

    def process_words(self, bv_w, add_unk):

        #Process words input batch
        if self._vocab != [] and add_unk:
            bv_w = self.to_unks(bv_w, self._vocab['words'])
        #Add GO tken and EOS token
        bv_w_delim = self.add_eos(self.add_go(bv_w))
        #Find lenght of each sentence in batch
        seq_len_w = self.seq_len(bv_w_delim)
        #find max length
        max_len_w = self.max_seq_len(seq_len_w)
        #Pad all sentences to max lenght
        bv_w_pad = self.pad(bv_w_delim, max_len_w)
        #stack
        bv_w_in = np.vstack(bv_w_pad)

        return seq_len_w, bv_w_in, max_len_w

    def process_tags(self, bv_t, max_len_w):

        #Process tags input batch
        #add go token to tags
        bv_t_go = self.add_go(bv_t)
        #find each tag sequence lenght
        seq_len_t_go = self.seq_len(bv_t_go)
        #Pad sequence lenght to max lenght first elemnt is 0
        #b/c words start with GO token
        seq_len_t_pad = self.pad(seq_len_t_go, max_len_w, l_pad_len=1)
        #Flantten
        seq_len_t = np.reshape(seq_len_t_pad, [-1])
        #Find max len
        max_len_t = max(seq_len_t)

        #Pad on a sentence level (first vector is 0)
        bv_t_go = self.pad(self.add_pad_vec(bv_t_go), max_len_t)
        bv_t_in = self.pad_matrix(bv_t_go, max_len_w, max_len_t)
        bv_t_in = np.reshape(bv_t_in, (-1, max_len_t))

        bv_t_eos = flatten_to_1D(self.add_eos(bv_t))

        return seq_len_t, bv_t_in, bv_t_eos

    def process_pos(self, bv_pos, max_len_w):

        bv_pos_delim = self.add_eos(self.add_go(bv_pos))
        bv_pos_pad = self.pad(bv_pos_delim, max_len_w)
        bv_pos_in = np.vstack(bv_pos_pad)

        return bv_pos_in

    def process_chars(self, bv_c, max_len_w):

        bv_c_delim = self.add_eos(self.add_go(bv_c))
        seq_len_c = self.seq_len(bv_c_delim)
        seq_len_c = self.pad(seq_len_c, max_len_w, l_pad_len=1)
        max_len_c = max(flatten_to_1D(seq_len_c))
        bv_c_delim = self.pad(self.add_pad_vec(bv_c_delim), max_len_c)
        bv_c_in = self.pad_matrix(bv_c_delim, max_len_w, max_len_c)
        bv_c_in = np.vstack(bv_c_in)
        seq_len_c = np.reshape(seq_len_c, [-1])
        return seq_len_c, bv_c_in

    def process(self, bv, add_unk=False):
        batch = dict()
        self._seq_len = self.seq_len(bv['words'])
        seq_len_w, bv_w_in, max_len_w = self.process_words(bv['words'], add_unk)
        batch.update({'word': {'in': bv_w_in, 'len': seq_len_w}})

        seq_len_t, bv_t_in, bv_t_eos = self.process_tags(bv['tags'], max_len_w)
        batch.update({'tag': {'in': bv_t_in, 'len': seq_len_t, 'out': bv_t_eos}})

        bv_pos_in = self.process_pos(bv['pos'], max_len_w)
        batch.update({'pos': {'in': bv_pos_in, 'out': bv_pos_in}})

        seq_len_c, bv_c_in = self.process_chars(bv['chars'], max_len_w)
        batch.update({'char': {'in': bv_c_in, 'len': seq_len_c}})
        bv_w_t = self.pad(bv['words'], max(self._seq_len))
        bv_w_t_in = self._vocab['words'].to_tokens(bv_w_t)
        batch.update({'word_t': {'in': bv_w_t_in, 'len': self._seq_len}})
        return batch

    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
