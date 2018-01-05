from __future__ import print_function

import copy
import numpy as np
from itertools import islice

from vocab import PAD, GO, EOS, UNK
from utils import operate_on_Narray, _operate_on_Narray, flatten_to_1D


class Batcher(object):

    def __init__(self, data, batch_size, reverse):
        self._data = data
        self._batch_size = batch_size
        self._revese = reverse
        self._d_size = len(data.values()[0])

    def get_batch_size(self):
        return self._batch_size

    def _get_pos(self, data, pos_id):
        return data[pos_id]

    def get_pos(self, batch_data, pos_id):
        return operate_on_Narray(batch_data, self._get_pos, pos_id)

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

    # def remove_delim_len(self, data, d_len):
    #     return [d[1:l-1].tolist() for d, l in zip(data, d_len)]

    def remove_delim_len(self, data):
        return [d[1:l-1].tolist() for d, l in zip(data['in'], data['len'])]

    # def remove_delim_len(self, *args):
    #     if len(args) == 1:
    #         return remove_delim_len_batch(self, *args)



    def get_random_batch(self):
        batch = dict()
        d_index = np.random.randint(self._d_size, size=self._batch_size)
        for k in self._data.keys():
            batch[k] = np.array(self._data[k])[d_index]
        return batch

    def get_permute_batch(self):
        batch = dict()
        num_batches = np.ceil(float(self._d_size)/self._batch_size)
        batch_permute = np.random.permutation(int(num_batches))
        batch = {k : np.array_split(self._data[k], num_batches)
                    for k in self._data.keys()}
        return [{k: batch[k][i] for k in self._data.keys()}
                    for i in batch_permute]

    def get_batch(self):
        num_batches = np.ceil(float(self._d_size)/self._batch_size)
        words = np.array_split(self._data['words'], num_batches)
        tags = np.array_split(self._data['tags'], num_batches)
        chars = np.array_split(self._data['chars'], num_batches)
        return [dict(zip(('words','tags','chars'),(w,t,c)))
                        for w,t,c in zip(words, tags, chars)]

    def process_words(self, bv_w):

        #Process words input batch
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

        bv_pos_in = self.process_pos(bv_t, max_len_w)
        return seq_len_t, bv_t_in, bv_t_eos, bv_pos_in

    def process_pos(self, bv_t, max_len_w):

        pos_id = 0 if self._revese else -1
        bv_pos = self.get_pos(bv_t, pos_id)
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

    def process(self, bv):
        #TODO do i really need the deepcopy?
        bv_w = copy.deepcopy(bv['words'].tolist())
        self._seq_len = self.seq_len(bv_w)
        seq_len_w, bv_w_in, max_len_w = self.process_words(bv_w)

        bv_t = copy.deepcopy(bv['tags'].tolist())
        seq_len_t, bv_t_in, bv_t_eos, bv_pos_in = self.process_tags(bv_t, max_len_w)
        # bv_pos_in = self.process_pos(bv_t, max_len_w)

        bv_c = copy.deepcopy(bv['chars'].tolist())
        seq_len_c, bv_c_in = self.process_chars(bv_c, max_len_w)

        return bv_w_in, seq_len_w, bv_c_in, seq_len_c, bv_pos_in, bv_t, bv_t_in, seq_len_t, bv_t_eos

    def _process(self, bv):
        batch = dict()
        #TODO do i really need the deepcopy?
        bv_w = copy.deepcopy(bv['words'].tolist())
        self._seq_len = self.seq_len(bv_w)
        seq_len_w, bv_w_in, max_len_w = self.process_words(bv_w)
        batch.update({'word': {'in': bv_w_in, 'len': seq_len_w}})

        bv_t = copy.deepcopy(bv['tags'].tolist())
        seq_len_t, bv_t_in, bv_t_eos, bv_pos_in = self.process_tags(bv_t, max_len_w)
        # bv_pos_in = self.process_pos(bv_t, max_len_w)
        batch.update({'tag': {'in': bv_t_in, 'len': seq_len_t, 'out': bv_t_eos}})
        batch.update({'pos': {'in': bv_pos_in, 'out': bv_pos_in}})

        bv_c = copy.deepcopy(bv['chars'].tolist())
        seq_len_c, bv_c_in = self.process_chars(bv_c, max_len_w)
        batch.update({'char': {'in': bv_c_in, 'len': seq_len_c}})

        return batch


    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
