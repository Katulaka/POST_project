import os
import copy

from treelib import Node, Tree
from nltk.tree import Tree as Tree_
from tag_ops import R, L, CL, CR
from utils.utils import operate_on_Narray, _operate_on_Narray


class TreeAux(object):

    def __init__(self, height=0, lids=[]):
        self.height = height
        self.leaves = lids

def ptb_to_tree(tree, line, max_id=0, leaf_id=1, parent_id=None):
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
        offset, max_id, leaf_id = ptb_to_tree(tree, line, max_id, leaf_id, parent_id)
        total_offset += offset
        line = line[offset:]

    return total_offset+1, max_id, leaf_id

def get_dependancies(fin, penn_path='code/utils/tags/pennconverter.jar'):

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

def add_height(tree, tree_dep):

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

def ptb_to_trees(fin):
    #TODO maybe change code to nltk trees
    tree_deps = get_dependancies(fin)
    with open(fin) as f:
        lines = [x.strip('\n') for x in f.readlines()]
    for i, line in enumerate(lines):
        #max_id is the number of words in line + 1.
        # This is index kept in order to number words from 1 to num of words
        max_id = len(Tree_.fromstring(line).leaves()) + 1
        line = line.replace('(', ' ( ').replace(')', ' ) ').split()
        tree = Tree()
        ptb_to_tree(tree, line, max_id)
        add_height(tree, tree_deps[i])
        yield tree

def path_to_tags(tree, path):
    tags = []
    for p in path:
        _res = []
        _p = copy.copy(p)
        if _p[0] in [CL, CR]:
            _res.append(_p[0])
            _p = _p[1:]
        while _p[:-1]:
            el_p = _p.pop(0)
            _res.append(tree[el_p].tag)
            for c in tree.children(el_p):
                if c.identifier != _p[0]:
                    _res.append(R+c.tag if c.identifier > _p[0] else L+c.tag)
        _res.append(tree[_p[0]].tag)
        tags.append(_res)
    return tags

def tree_to_path(tree, nid, path):

    # Stop condition
    if tree[nid].is_leaf():
        path[nid] = []
        return nid, tree[nid].data.height

    # Recursion
    flag = CR
    for child in tree.children(nid):
        cid = child.identifier
        leaf_id, height = tree_to_path(tree, cid, path)

        if (height == 0):
            # Reached end of path can add flag
            path[leaf_id].insert(0, flag)
            # path[leaf_id].append(flag)

        if height > 0:
            path[leaf_id].insert(0, nid)
            # only single child will have height>0
            # and its value will be the one that is returned
            # to the parent
            ret_leaf_id, ret_height = leaf_id, height-1

            # once we reached a height>0, it means that
            # this path includes the parent, and thus flag
            # direction should flip
            flag = CL

    return ret_leaf_id, ret_height

def gen_tags(fin):
    print ("gen_tags")

    for i, tree in enumerate(ptb_to_trees(fin)):
        path = {}
        try:
            tree_to_path(tree, tree.root, path)
            yield (path_to_tags(tree, path.values()), [tree[k].tag for k in path.keys()])
        except:
            print ("Wrong tree %d in %s" % (i, fin))

def tree_to_ptb(tree):
    nid = tree.root
    if tree[nid].is_leaf():
        return  ' (' + tree[nid].tag + ' ' + tree[nid].data.word + ')'

    res = ' (' + tree[nid].tag

    for c_nid in sorted(tree.children(nid), key=lambda x: x.identifier):
        res += tree_to_ptb(tree.subtree(c_nid.identifier))

    return res + ')'

def trees_to_ptb(trees):
    return _operate_on_Narray(trees, tree_to_ptb)


# def find_tag(tree):
#     sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
#     new_tag = []
#     while (sorted_nodes):
#         _new_tag =[]
#         if tree[sorted_nodes[0]].is_leaf():
#             _new_tag.append(tree[sorted_nodes[0]].tag)
#             del sorted_nodes[0]
#         else:
#             while (not tree[sorted_nodes[0]].is_leaf()):
#                 ch = [R+c.tag if sorted_nodes[1] < c.identifier else L+c.tag
#                             for c in tree.siblings(sorted_nodes[1])]
#                 _new_tag.append(''.join([tree[sorted_nodes[0]].tag] + ch))
#                 del sorted_nodes[0]
#             _new_tag.append(tree[sorted_nodes[0]].tag)
#             del sorted_nodes[0]
#         new_tag.append(_new_tag)
#     return new_tag
#
#
# def _find_tag(tree):
#     sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
#     new_tag = []
#     while (sorted_nodes):
#         sub_tag = []
#         if tree[sorted_nodes[0]].data.comb_side != '':
#             sub_tag = [tree[sorted_nodes[0]].data.comb_side]
#         while (not tree[sorted_nodes[0]].is_leaf()):
#             sub_tag += [sorted_nodes[0]]
#             del sorted_nodes[0]
#         sub_tag += [sorted_nodes[0]]
#         del sorted_nodes[0]
#         new_tag.append(sub_tag)
#     return path_to_tags(tree, new_tag)
