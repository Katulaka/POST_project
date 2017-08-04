from __future__ import print_function

import collections
import numpy as np


PAD = ['PAD', 0]
GO = ['GO', 1]
EOS = ['EOS', 2]
UNK = ['UNK', 3]
GO_TAG = ['<GO>', 4]
EOS_TAG = ['<EOS>', 5]

class Vocab(object):

    def __init__(self, _text, _size=0):
        self._token_to_id = dict()
        self._id_to_token = dict()

        special_tokens = dict([PAD, GO, EOS, UNK, GO_TAG, EOS_TAG])
        s_tokens_sort = sorted(special_tokens.items(), key=lambda x: x[1])
        self._count = map(lambda t: [t[0], -1], s_tokens_sort)

        if _size > 0 :
            common_tokens = collections.Counter(_text).most_common(_size - 1)
        else:
            common_tokens = collections.Counter(_text).most_common()
        self._count.extend(common_tokens)
        for token, _ in self._count:
            self._token_to_id[token] = len(self._token_to_id)
        unk_count = 0
        for token in _text:
            if token not in self._token_to_id:
                unk_count += 1
        self._count[special_tokens['UNK']][1] = unk_count
        self._id_to_token = dict(zip(self._token_to_id.values(),
                                self._token_to_id.keys()))

    def check_vocab(self, token):
        if token not in self._token_to_id:
            return None
        return self._token_to_id[token]

    def vocab_size(self):
        return max(self._token_to_id.values()) + 1

    def token_to_id(self, token):
        if token not in self._token_to_id:
            return self._token_to_id[UNK[1]]
        return self._token_to_id[token]

    def to_ids(self, sentences):
        return map(lambda s: map(lambda w: self.token_to_id(w), s), sentences)

    def id_to_token(self, token_id):
        if token_id not in self._id_to_token:
            raise ValueError('id not found in vocab: %d.' % token_id)
        return self._id_to_token[token_id]

    def to_tokens(self, ids):
        return map(lambda s: map(lambda w: self.id_to_token(w), s), ids)

#TODO change pad function to avoid in place replacment
def pad(data, mlen=0, pad_token=PAD[1]):

    max_len = mlen if mlen > 0 else len(max(data, key=len))
    for i,el in enumerate(data):
        pad_len = max_len-len(el)
        data[i] = np.lib.pad(el, (0, pad_len), 'constant', constant_values=(pad_token))
    return data

def add_go(data, start_token=GO[1]):
    return map(lambda x: [start_token] + x, data)

def add_eos(data, end_token=EOS[1]):
    return map(lambda x: x + [end_token], data)

def add_go_tag(data, start_token=GO_TAG[1]):
    return map(lambda x: [[start_token]] + x, data)

def add_eos_tag(data, end_token=EOS_TAG[1]):
    return map(lambda x: x + [[end_token]], data)

def to_onehot(vec_in, max_len, size):
    vec_out = np.zeros((max_len, size))
    vec_out[np.arange(len(vec_in)), vec_in] = 1
    return vec_out
