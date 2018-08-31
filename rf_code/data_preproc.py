from nltk.corpus import BracketParseCorpusReader as reader
import os
from tree_t import gen_aug_tags


def get_ptb(data_path):
    """ """
    print("[[get_raw_data:]] Getting raw data from corpora")
    if not os.path.exists(data_path):
        try:
            os.makedirs(os.path.abspath(data_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        scp_path = ("scp -r login.eecs.berkeley.edu:" +
        "/project/eecs/nlp/corpora/EnglishTreebank/wsj/* ")
        os.system(scp_path + data_path)

def remove_traces(ts): # Remove traces and null elements
    for t in ts:
        for ind, leaf in reversed(list(enumerate(t.leaves()))):
            postn = t.leaf_treeposition(ind)
            parentpos = postn[:-1]
            if leaf.startswith("*") or t[parentpos].label() == '-NONE-':
                while parentpos and len(t[parentpos]) == 1:
                    postn = parentpos
                    parentpos = postn[:-1]
                print(t[postn], "will be deleted")
                del t[postn]
    return ts

def simplify(ts): # Simplify tags
    for t in ts:
        for s in t.subtrees():
            tag = s.label()
            if tag not in ['-LRB-', '-RRB-', '-LCB-', '-RCB-', '-NONE-']:
                if '-' in tag or '=' in tag or '|' in tag:
                    simple = tag.split('-')[0].split('=')[0].split('|')[0]
                    s.set_label(simple)
                    print('substituting', simple, 'for', tag)
    return ts


def clean_up_file(fin, fout):
    rfin = fin.split('/')
    r = reader('/'.join(rfin[:-2]), '/'.join(rfin[-2:]))
    trees = simplify(remove_traces(list(r.parsed_sents())))
    with open(fout, 'w') as outfile:
        for i, t in enumerate(trees):
            line = ' '.join(t.pformat().replace('\n', '').split())
            outfile.write("%s\n" % line)

def clean_up(src_dir, dst_root_dir):

    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit():
            dst_dir = os.path.join(dst_root_dir, directory[-2:])
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            for fname in sorted(filenames):
                fout = os.path.join(dst_dir, fname)
                fin = os.path.join(directory, fname)
                clean_up_file(fin, fout)

def create_data_file(src_dir, fout):
    print ("[[Batcher.create_data_file]] Creating data file: \
                %s from source dir %s" % (fout, src_dir))
    data = {}
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit() and directory[-2:] not in ['00','01','24']:
            for fname in sorted(filenames):
                k = fname.split('_')[-1].split('.')[0]
                fin = os.path.join(directory, fname)
                data.setdefault(k, {}).update(gen_aug_tags(fin))
    with open(fout, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    src_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/raw_data/wsj'
    dst_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
    clean_up(src_dir, dst_dir)
