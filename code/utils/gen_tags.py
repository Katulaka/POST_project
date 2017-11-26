from __future__ import print_function

from nltk.corpus import BracketParseCorpusReader as reader
import os
from treelib import Node, Tree
from utils import operate_on_Narray, _operate_on_Narray
import copy as copy

# R = '/'
# L = '\\'
R = '}'
L = '{'
CR = '>'
CL = '<'
UP = '+'
NA = '|'
ANY = '*'

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

def penn_converter(fin, penn_path='code/utils/pennconverter.jar'):

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
        gen_height(tree, tree_deps[i])
        yield tree


def extend_path(tree, path):
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

def gen_tag(tree, nid, path):

    # Stop condition
    if tree[nid].is_leaf():
        path[nid] = []
        return nid, tree[nid].data.height

    # Recursion
    flag = CR
    for child in tree.children(nid):
        cid = child.identifier
        leaf_id, height = gen_tag(tree, cid, path)

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

    for i, tree in enumerate(get_trees(fin)):
        path = {}
        try:
            gen_tag(tree, tree.root, path)
            # tst = extend_path(tree, path.values())
            # res = [t[0] for t in tst if t[0] in ['>', '<']]
            # if len(res) != len(tst) - 1:
            yield (extend_path(tree, path.values()),
                            [tree[key].tag for key in path.keys()])
        except:
            print ("Wrong tree %d in %s" % (i, fin))

def to_mrg(tree, v): #TODO fix function

    nid = tree.root
    if tree[nid].is_leaf():
        return  ' (' + tree[nid].tag + ' ' + v[nid] + ')'

    res = ' (' + tree[nid].tag

    for c_nid in sorted(tree.children(nid), key=lambda x: x.identifier):
        res += to_mrg(tree.subtree(c_nid.identifier), v)

    return res + ')'


class TagOp(object):

    # def __init__(self, pos, direction, sub_split, slash_split, reverse, no_val_gap):
    def __init__(self, pos, direction, no_val_gap):
        # self.sub_split = sub_split
        self.direction = direction
        self.pos = pos
        # self.slash_split = slash_split
        # self.reverse = reverse
        self.no_val_gap = no_val_gap

    def _mod_tag(self, tag, l_sym, r_sym):
        return tag.replace(L, l_sym+L+r_sym).replace(R, l_sym+R+r_sym)

    def _tag_split(self, tag):
        return self._mod_tag(tag, UP, '').split(UP)
        # tag.replace(L, UP+L).replace(R, UP+R).split(UP)

    def _slash_split(self, tag):
        return self._mod_tag(tag, UP, UP).split(UP)

    def _revese(self, tag):
        return UP.join(tag.split(UP)[::-1])

    def _remove_val_gap(self, tag):
        return self._mod_tag(tag, '', ANY+NA).split(NA)[0]


    def modify_tag(self, tag):

        # if self.pos: #if just pos no need to deal with the  whole sequence
        #     return [tag.split(UP)[-1]]

        if not self.direction: #if don't want to keep left/right indication.
            tag = tag.replace(R, L)   # default: all left

        if self.no_val_gap:
            tag = tag.replace(L, L+ANY+NA).replace(R, R+ANY+NA).split(NA)[0]


        # if self.sub_split: #split on the individuale parts (including missing)
        #     # _tag = self._tag_split(tag)
        #     _tag = tag.replace(L, UP+L).replace(R, UP+R).split(UP)
        #     if self.no_val_gap:
        #         _tag = [e.replace(L, L+ANY+NA).replace(R, R+ANY+NA).split(NA)[0]
        #          for e in _tag]
        #     return _tag
        # if self.reverse:
        #     tag = self._revese(tag)
        # if self.slash_split: #split on individuale pars and directionality symbols
        #     return self._slash_split(tag)
        # return tag.split(UP) #splip just between levels
        return tag

    def _modify_fn(self, tag_list):
        return [self.modify_tag(tag) for tag in tag_list]

    def modify_fn(self, tags):
        return _operate_on_Narray(tags, self.modify_tag)
        # return operate_on_Narray(tags, self._modify_fn)

    def combine_tag(self, tag):
        res = []
        _tag = tag
        try:
            if tag[0] in [CL, CR]:
                res.append(tag[:2])
                _tag = tag[2:]
        except:
            return res
        for t in _tag:
            if res and (res[-1][-1].endswith(L) or res[-1][-1].endswith(R)):
                res[-1] += [t]
            elif res and (t.startswith(L) or t.startswith(R)):
                res[-1] += [t]
            else:
                res.append([t])
        return res


    # def combine_tag(self, tag):
    #     res = []
    #     for t in tag:
    #         if res and (res[-1].endswith(L) or res[-1].endswith(R)):
    #             res[-1] += t
    #         elif res and (t.startswith(L) or t.startswith(R)):
    #             res[-1] += t
    #         else:
    #             res.append(t)
    #     return res

    def combine_fn(self, tags):
        return operate_on_Narray(tags, self.combine_tag)

    def add_right(self, tag):
        return R + tag

    def add_left(self, tag):
        return L + tag

# print ("start")
# # fin = '/Users/katia.patkin/Berkeley/Research/Tagger/POST/tmp.mrg'
# fin = '/Users/katia.patkin/Berkeley/Research/Tagger/raw_data/wsj/14/wsj_1493.mrg'
# res = gen_tags(fin)
# for i, r in enumerate(res):
#     ri = r
#     print (i)
