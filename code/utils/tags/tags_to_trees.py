from nltk.tree import Tree as Tree_
from treelib import Node, Tree
import os

class TreeAux(object):

    def __init__(self, height=0, lids=[]):
        self.height = height
        self.leaves = lids

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

def penn_converter(fin, penn_path='code/utils/tags/pennconverter.jar'):

    dep_dict_file = []
    dep_dict_tree = {}
    lines = os.popen("java -jar "+penn_path+"< "+fin+" -splitSlash=false").read().split('\n')

    for line in lines:
        words = line.split()
        if words:
            dep_dict_tree[int(words[0])] = int(words[6])
        else:
            dep_dict_file.append(dep_dict_tree)
            dep_dict_tree = {}

    return dep_dict_file

def gen_height(tree, tree_dep):

    for n in tree.all_nodes():
        n.data.leaves = []

    for leaf in tree.leaves():
        lid = leaf.identifier
        hid = tree_dep[lid]
        if hid == tree.root:
            tree[lid].data.height = tree.depth(tree[lid])
            for cid in [p for p in tree.paths_to_leaves() if lid in p][0]:
                tree[cid].data.leaves += [lid]
        else:
            height = -1
            cid = lid
            cond = True
            while cond:
                tree[cid].data.leaves += [lid]
                height += 1
                cid = tree.parent(cid).identifier
                cid_leaves = [l.identifier for l in tree.leaves(cid)]
                cid_l_dep = [tree_dep[l] for l in cid_leaves if l != lid]
                cond = set(cid_l_dep).issubset(set(cid_leaves))
            tree[lid].data.height = height

    x_nodes = [n.identifier for n in tree.all_nodes() if n.data.leaves == []]
    for x_node in x_nodes[::-1]:
        min_id = min(tree.children(x_node), key=lambda c: c.data.height)
        _lid = min_id.data.leaves[0]
        tree[_lid].data.height += 1
        tree[x_node].data.leaves += [_lid]

    return True

def get_trees(fin):
    #TODO maybe change code to nltk trees
    tree_deps = penn_converter(fin)
    with open(fin) as f:
        lines = [x.strip('\n') for x in f.readlines()]
    for i, line in enumerate(lines):
        #max_id is the number of words in line + 1.
        # This is index kept in order to number words from 1 to num of words
        max_id = len(Tree_.fromstring(line).leaves()) + 1
        line = line.replace('(', ' ( ').replace(')', ' ) ').split()
        tree = Tree()
        get_tree(tree, line, max_id)
        gen_height(tree, tree_deps[i])
        yield tree
