from nltk.corpus import BracketParseCorpusReader as reader
import os

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
            if tag not in ['-LRB-', '-RRB-', '-NONE-']:
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

    all_data = dict()
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit():
            dst_dir = os.path.join(dst_root_dir, directory[-2:])
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            for fname in sorted(filenames):
                fout = os.path.join(dst_dir, fname)
                fin = os.path.join(directory, fname)
                clean_up_file(fin, fout)


if __name__ == '__main__':
    src_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/raw_data/wsj'
    dst_dir = '/Users/katia.patkin/Berkeley/Research/Tagger/gold_data'
    clean_up(src_dir, dst_dir)
