from __future__ import print_function

import os
import numpy as np
from gen_tags import gen_stags, gen_tags
from utils import make_dir

def get_rawdata(data_path):

    print("Getting RAW data from corpora")
    make_dir(data_path)
    if not os.path.exists(os.path.join(data_path, 'wsj')):
        scp_path = ("scp -r login.eecs.berkeley.edu:" +
                    "/project/eecs/nlp/corpora/EnglishTreebank/wsj/ ")
        os.system(scp_path + data_path)


def convert_data_hier(src_dir, dest_dir, tag_type='stags'):

    gen_tags_fn = gen_stags if tag_type == 'stags' else gen_tags
    for sub_dir in os.listdir(src_dir):
        out_path = os.path.join(dest_dir, sub_dir)
        make_dir(os.path.join(out_path, "tags"))
        make_dir(os.path.join(out_path, "words"))

        for f_in in os.listdir(os.path.join(src_dir, sub_dir)):
            data_in = os.path.join(src_dir, sub_dir, f_in)
            tags_out = os.path.join(out_path, "tags", f_in+'.tg')
            words_out = os.path.join(out_path, "words", f_in+'.wrd')

            with open(tags_out, 'w') as t_file:
                with open(words_out, 'w') as w_file:
                    for s_tags, s_words in gen_tags_fn(data_in):
                        print(s_tags, file=t_file)
                        print(s_words, file=w_file)


def convert_data_flat(src_dir, words_out, tags_out, tag_type='stags'):
    """ If src dir is empty or not a file will result in empty file """

    gen_tags_fn = gen_stags if tag_type == 'stags' else gen_tags
    with open(tags_out, 'w') as t_file:
        with open(words_out, 'w') as w_file:
            for directory, _, filenames in os.walk(src_dir):
                for fname in filenames:
                    data_in = os.path.join(directory, fname)
                    print("Reading file %s" %(data_in))
                    for s_tags, s_words in gen_tags_fn(data_in):
                        print(s_tags, file=t_file)
                        print(s_words, file=w_file)
