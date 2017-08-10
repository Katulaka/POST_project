from __future__ import print_function

import os
import pickle
import re
from treelib import Node, Tree

from utils import operate_on_Narray
from nltk.corpus import BracketParseCorpusReader as reader

RIGHT = '/'
LEFT = '\\'
UP = '+'
NONE = '|'

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

def simplify(ts):# Simplify tags
    for t in ts:
        for s in t.subtrees():
            tag = s.label()
            if tag not in ['-LRB-', '-RRB-', '-NONE-']:
                #  and '-' in tag:
                if '-' in tag or '=' in tag or '|' in tag:
                    simple = tag.split('-')[0].split('=')[0].split('|')[0]
                    s.set_label(simple)
                    print('substituting', simple, 'for', tag)
    return ts

class TreeAux(object):

    def __init__(self, min_range=0, max_range=0, height=0):
        self.n_range = (min_range, max_range)
        self.height = height

def get_tree(tree, line, max_id=0, leaf_id=1, parent_id=None):
    # starts by ['(', 'pos']
    pos_tag = line[1]
    if parent_id is None:
        pos_id = 0
    else:
        pos_id = max_id
        max_id += 1

    tree.create_node(pos_tag, pos_id, parent_id, TreeAux())

    parent_id = pos_id
    total_offset = 2

    if line[2] != '(':
        # sub-tree is leaf
        # line[0:3] = ['(', 'pos', 'word', ')']
        word_tag = line[2]
        tree.create_node(word_tag, leaf_id, parent_id, TreeAux())
        return 4, max_id, leaf_id+1

    line = line[2:]

    while line[0] != ')':
        offset, max_id, leaf_id = get_tree(tree, line, max_id, leaf_id, parent_id)
        total_offset += offset
        line = line[offset:]

    return total_offset+1, max_id, leaf_id

def gen_range(tree, nid, min_range, max_range):

    if tree[nid].is_leaf():
        tree[nid].data.n_range = (min_range, max_range)
        return (max_range, max_range + 1)

    for child in tree.children(nid):
        cid = child.identifier
        (min_range, max_range) = gen_range(tree, cid, min_range, max_range)

    _min_range = min(tree.children(nid),
                    key=lambda c: c.data.n_range[0]).data.n_range[0]
    _max_range = max(tree.children(nid),
                    key=lambda c: c.data.n_range[1]).data.n_range[1]

    tree[nid].data.n_range = (_min_range, _max_range)

    return (min_range, max_range)

def penn_converter(fin, penn_path='code/utils/pennconverter.jar'):

    dep_dict_file = []
    dep_dict_tree = {}

    # try:
    lines = os.popen("java -jar "+penn_path+"< "+fin).read().split('\n')
    # except Exception as ex:
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     import pdb; pdb.set_trace()
    #     print message

    for line in lines:
        words = line.split()
        if words:
            dep_dict_tree[int(words[0])] = int(words[6])
        else:
            dep_dict_file.append(dep_dict_tree)
            dep_dict_tree = {}

    return dep_dict_file

def gen_height(tree, tree_dep):

    for leaf in tree.leaves(tree.root):
        lid = leaf.identifier
        depid = tree_dep[lid]
        if depid == tree.root:
            tree[lid].data.height = tree.depth(leaf)
        else:
            min_range = min(tree[depid].data.n_range[0], tree[lid].data.n_range[0])
            max_range = max(tree[depid].data.n_range[1], tree[lid].data.n_range[1])
            height = 0
            pid = tree.parent(lid).identifier

            while tree[pid].data.n_range[0] > min_range or tree[pid].data.n_range[1] < max_range:
                height += 1
                pid = tree.parent(pid).identifier

            tree[lid].data.height = height
    return True

def get_trees(fin):

    tree_deps = penn_converter(fin)

    rfin = fin.split('/')
    r = reader('/'.join(rfin[:-2]), '/'.join(rfin[-2:]))
    trees = simplify(remove_traces(list(r.parsed_sents())))

    for i, t in enumerate(trees):
        line = str(t.pformat().replace('\n', ''))
        line = line.replace('(', ' ( ').replace(')', ' ) ').split()
        tree = Tree()
        #max_id is the number of words in line + 1.
        # This is index kept in order to number words from 1 to num of words
        max_id = len(t.leaves()) + 1
        get_tree(tree, line, max_id)
        gen_range(tree, tree.root, 0, 1)
        gen_height(tree, tree_deps[i])
        yield tree


def extend_path(tree, current_id, leaf_id, path_dict):

    path_tag = tree.parent(current_id).tag
    if tree.siblings(current_id):
        siblings = [RIGHT+s.tag if s.identifier > current_id else LEFT+s.tag for s in tree.siblings(current_id)]
        siblings.insert(0, path_tag)
        path_tag = "".join(siblings)

    path_dict[leaf_id] = UP.join([path_tag, path_dict[leaf_id]])


def gen_tag(tree, nid, path):

    # Stop condition
    if tree[nid].is_leaf():

        pid = tree.parent(nid).identifier
        path[nid] = tree[pid].tag

        if tree[nid].data.height > 1 :
            extend_path(tree, pid, nid, path)

        return nid, tree[nid].data.height - 1

    # Recursion
    for child in tree.children(nid):
        cid = child.identifier
        leaf_id, height = gen_tag(tree, cid, path)

    if height == 1:
        return None, 1

    elif height > 1:
        pid = tree.parent(nid).identifier
        extend_path(tree, pid, leaf_id, path)

    return leaf_id, height - 1

def gen_tags(fin):
    for tree in get_trees(fin):
        path = {}
        gen_tag(tree, tree.root, path)
        yield (path.values(), [tree[key].tag for key in path.keys()])


# def _gen_tags(fin):
#
#     rfin = fin.split('/')
#     r = reader('/'.join(rfin[:-2]), '/'.join(rfin[-2:]))
#     trees = simplify(remove_traces(list(r.parsed_sents())))
#     for t in trees:
#         res = [str("+ ".join(r)).split("+ ") for r in map(list, zip(*t.pos()))]
#         yield res[::-1]
#
#     # for tree in get_trees(fin):
#     #     rng = xrange(1, len(tree.leaves(tree.root))+1)
#     #     words = ' '.join(map(lambda lid: tree[lid].tag, rng))
#     #     tags = ' '.join(map(lambda lid: tree.parent(lid).tag, rng))
#     #     yield (tags, words)


class TagOp(object):

    def __init__(self, sub_split, direction, pos): #TODO
        self.sub_split = sub_split
        self.direction = direction
        self.pos = pos


    def split_tag(self, tag, sym):
            return tag.replace(LEFT, sym+LEFT).replace(RIGHT, sym+RIGHT).split(sym)

    def _split_fn(self, tag):
        if self.pos: #if just pos no need to deal with the  whole sequence
            return [tag.split(UP)[-1]]
        if not self.direction: #if don't want to keep left/right indication.
                                # default: all left
            tag = tag.replace(RIGHT, LEFT)
        if self.sub_split: #split on the individuale parts (including missing)
            return self.split_tag(tag, UP)
        return tag.split(UP) #splip just between levels

    def split_fn(self, tags):
        return operate_on_Narray(tags, self._split_fn)

    def combine_tag(self, tag):
        res = []
        for t in tag:
            if (t.startswith(LEFT) or t.startswith(RIGHT)) and res:
                res[-1] += t
            else:
                res.append(t)
        return res

    def combine_fn(self, tags):
        return operate_on_Narray(tags, self.combine_tag)

    def add_right(self, tag):
        return RIGHT + tag

    def add_left(self, tag):
        return LEFT + tag
