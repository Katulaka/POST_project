from __future__ import print_function

import os
import json
import time
import numpy as np
from itertools import islice

from data_preproc import create_data_file
from vocab import PAD, GO, EOS, UNK, Vocab
from utils import operate_on_Narray, _operate_on_Narray, flatten_to_1D
from tag_ops import TagOp


class Batcher(object):

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, '_'+k, v)

        self._t_op = TagOp(**self._tags_type)
        self._data = self.create_data()
        self._vocab = self.create_vocab()
        self.convert_to_ids()


    def load_fn(self, src_dir=None, data_file=None):
        """ """
        start_time = time.clock()

        if src_dir is None:
            src_dir = self._src_dir
        if data_file is None:
            data_file = self._data_file

        if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
            print ("[[Batcher.load_fn]] Couldn't find data file: %s" % data_file)
            data = create_data_file(src_dir, data_file)
        else:
            print ("[[Batcher.load_fn]] Loading data file: %s" % data_file)
            with open (data_file, 'r') as outfile:
                data = json.load(outfile)
        print ("[[Batcher.load_fn]] %.3f to load data" % (time.clock()-start_time))
        return data

    def create_data(self):
        """ """
        _data = self.load_fn()
        start_time = time.clock()
        for k in _data.keys():
            _data[k].setdefault('pos',[]).extend([[t[-1] for t in ts] for ts in _data[k]['tags']])
            _data[k].setdefault('chars',[]).extend([[list(w) for w in s] for s in _data[k]['words']])
            _data[k]['tags'] = [[self._t_op.modify_tag(t) for t in ts] for ts in _data[k]['tags']]
        print ("[[Batcher.create_data]] %.3f to create dataset" % (time.clock()-start_time))
        return _data

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
            print ("[[Batcher.create_vocab]] %.3f for %s vocab (size: %s)" %
                    (time.clock()-start_time, k, _vocab[k].vocab_size()))
        return _vocab

    def convert_to_ids(self):
        """ """
        for k, v in self._vocab.items():
            start_time = time.clock()
            for d_k, d_v in self._data.items():
                self._data[d_k][k] = v.to_ids(d_v[k])
            print ("[[Batcher.convert_to_ids]] %.3f for %s vocab" %
                    (time.clock()-start_time, k))
        return self

    def create_dataset(self, mode):
        """ """
        start_time = time.clock()
        self._ds = {}
        for k, v in self._vocab.items():
            for d_k in sorted(self._data.keys()):
                if int(d_k[:2]) in range(*self._dir_range[mode]):
                    self._ds.setdefault(k,[]).extend(self._data[d_k][k])
        self._d_size = len(self._ds[k])
        print ("[[Batcher.create_dataset]] %.3f to create ds" % (time.clock()-start_time))
        return self

    def get_subset_idx(self, src_file, precentage):
        """ """

        if os.path.exists(src_file):
            with open(src_file, 'r') as f:
                    subset_idx = json.load(f)
        else:
            import random
            size_subset = int(np.ceil(self._d_size * precentage))
            subset_idx = random.sample(range(1, self._d_size), size_subset)
            with open(src_file, 'w') as f:
                json.dump(subset_idx, f)
        return subset_idx

    def get_batch(self, permute=False, subset_idx=None):

        if not subset_idx is None:
            self._d_size = len(subset_idx)

        n_batches = int(np.ceil(float(self._d_size)/self._batch_size))
        if permute:
            batch_idx = np.random.permutation(n_batches)
        else:
            batch_idx = range(n_batches)

        batched = {}
        for k in self._vocab.keys():
            if not subset_idx is None:
                data = np.array(self._ds[k])[subset_idx].tolist()
            else:
                data = self._ds[k]
            batched.setdefault(k, np.array_split(data, n_batches))

        return [{k: batched[k][i].tolist() for k in self._vocab.keys()} for i in batch_idx]

    def update_vars(self):
        import sys
        if '--batch' in sys.argv[1:]:
            self._batch_size = int(sys.argv[1:][sys.argv[1:].index('--batch')+1])
        return self

    def get_batch_size(self):
        return self._batch_size

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

        return batch

    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
