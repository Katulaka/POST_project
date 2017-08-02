from __future__ import print_function

import os
import re
from treelib import Node, Tree


def is_balanced(string):
    iparens = iter('(){}[]<>')
    parens = dict(zip(iparens, iparens))
    closing = parens.values()
    stack = []
    for ch in string:
        delim = parens.get(ch, None)
        if delim:
            stack.append(delim)
        elif ch in closing:
            if not stack or ch != stack.pop():
                return False
    return not stack

def read_balanced_line(fin):
    s = ""
    lines = iter(open(fin, 'r'))
    for line in lines:
        if line.strip():
            s = s + line.split('\n')[0]
            if is_balanced(s) and s != "":
                yield s
                s = ""

class NRange(object):

    def __init__(self, min_range, max_range):
        self.n_range = (min_range, max_range)


def get_tree(tree, line, max_id=0, leaf_id=1, parent_id=None):

    # starts by ['(', 'pos']
    pos_tag = line[1].split('-')[0].split('=')[0].split('|')[0]
    if parent_id is None:
        pos_id = 0
    else:
        pos_id = max_id
        max_id += 1

    tree.create_node(pos_tag, pos_id, parent_id, NRange(0,0))

    parent_id = pos_id
    total_offset = 2

    if line[2] != '(':
        # sub-tree is leaf
        # line[0:3] = ['(', 'pos', 'word', ')']
        word_tag = line[2]
        tree.create_node(word_tag, leaf_id, parent_id, NRange(0,0))
        return 4, max_id, leaf_id+1

    line = line[2:]

    while line[0] != ')':
        offset, max_id, leaf_id = get_tree(tree, line, max_id, leaf_id, parent_id)
        total_offset += offset
        line = line[offset:]

    return total_offset+1, max_id, leaf_id


def get_trees(fin):

    for line in read_balanced_line(fin):

        tree = Tree()
        line = re.sub(r'\([\s]*-NONE-[^\)]+\)', '', line)
        p = re.compile('\([\S]+[\s]+\)')
        while p.findall(line):
            line = re.sub(p, '', line)

        # import pdb; pdb.set_trace()
        #max_id is the number of words in line + 1
        end_with_paren = re.compile('[\S]+\)').findall(line)
        max_id = len([w for w in end_with_paren if not w.endswith('))')]) + 1
        line = line.replace('(', ' ( ').replace(')', ' ) ').split()[1:]
        get_tree(tree, line, max_id)
        yield tree

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


def extend_path(tree, pid, leaf_id, path_dict):

    path_tag = tree.parent(pid).tag
    if tree.siblings(pid):
        siblings = map(lambda sibling: sibling.tag,  tree.siblings(pid))
        siblings.insert(0, path_tag)
        path_tag = "\\".join(siblings)

    path_dict[leaf_id] = "+".join([path_tag, path_dict[leaf_id]])


def gen_height_list(tree, tree_dep_dict):

    hieght_dict = {}

    for leaf in tree.leaves(tree.root):
        lid = leaf.identifier
        depid = tree_dep_dict[lid]
        if depid == tree.root:
            hieght_dict[lid] = tree.depth(leaf)

        else:
            min_range = min(tree[depid].data.n_range[0], tree[lid].data.n_range[0])
            max_range = max(tree[depid].data.n_range[1], tree[lid].data.n_range[1])
            height = 0
            pid = tree.parent(lid).identifier

            while tree[pid].data.n_range[0] > min_range or tree[pid].data.n_range[1] < max_range:
                height += 1
                pid = tree.parent(pid).identifier

            hieght_dict[lid] = height
    return hieght_dict

def gen_tree_dep_dict(data_in, penn_path='code/utils/pennconverter.jar'):

    fin = "dep_treebank"
    os.system("java -jar "+penn_path+"< "+data_in+"> "+fin) #TODO add error msg

    dep_dict_list = []
    dep_dict = {}
    lines = iter(open(fin, 'r'))
    for line in lines:
        words = line.split()
        if words:
            dep_dict[int(words[0])] = int(words[6])
        else:
            dep_dict_list.append(dep_dict)
            dep_dict = {}

    os.remove(fin)
    return dep_dict_list

def gen_stag(tree, nid, height_dict, path_dict):

    # Stop condition
    if tree[nid].is_leaf():

        pid = tree.parent(nid).identifier
        path_dict[nid] = tree[pid].tag

        if height_dict[nid] > 1 :
            extend_path(tree, pid, nid, path_dict)

        return nid, height_dict[nid] - 1

    # Recursion
    for child in tree.children(nid):
        cid = child.identifier
        leaf_id, height = gen_stag(tree, cid, height_dict, path_dict)

    if height == 1:
        return None, 1

    elif height > 1:
        pid = tree.parent(nid).identifier
        extend_path(tree, pid, leaf_id, path_dict)

    return leaf_id, height - 1

def gen_stags(fin):

    tree_dep_dict = gen_tree_dep_dict(fin)

    for i, tree in enumerate(get_trees(fin)):
        gen_range(tree, tree.root, 0, 1)
        try:
            height_dict = gen_height_list(tree, tree_dep_dict[i])
        except ValueError:
            print ("Error")
        path_dict = {}
        gen_stag(tree, tree.root, height_dict, path_dict)
        yield (' '.join(path_dict.values()), ' '.join(map(lambda key: tree[key].tag, path_dict.keys())))

def gen_tags(fin):
    for tree in get_trees(fin):
        rng = xrange(1, len(tree.leaves(tree.root))+1)
        words = ' '.join(map(lambda lid: tree[lid].tag, rng))
        tags = ' '.join(map(lambda lid: tree.parent(lid).tag, rng))
        yield (tags, words)
