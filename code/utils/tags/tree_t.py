import copy
import os
from nltk.tree import Tree as Tree_
from treelib import Tree
from utils.utils import _getattr_operate_on_Narray
from utils.tags.tag_ops import R, L, CL, CR

from functools import lru_cache
memoize = lru_cache(None)

class TreeData(object):
    def __init__(self, height=0, lids=[], miss_side='', comb_side='', word =''):
        self.height = height
        self.leaves = lids
        self.miss_side = miss_side
        self.comb_side = comb_side
        self.word = word


class TreeT(object):

    def __init__(self, max_id=0):
        self.tree = Tree()

    def from_ptb_to_tree(self, line, max_id=0, leaf_id=1, parent_id=None):
        # starts by ['(', 'pos']
        pos_tag = line[1]
        if parent_id is None:
            pos_id = 0
        else:
            pos_id = max_id
            max_id += 1

        self.tree.create_node(pos_tag, pos_id, parent_id, TreeData())

        parent_id = pos_id
        total_offset = 2

        if line[2] != '(':
            # sub-tree is leaf
            # line[0:3] = ['(', 'pos', 'word', ')']
            word_tag = line[2]
            self.tree.create_node(word_tag, leaf_id, parent_id, TreeData())
            return 4, max_id, leaf_id+1

        line = line[2:]

        while line[0] != ')':
            offset, max_id, leaf_id = self.from_ptb_to_tree(line, max_id, leaf_id, parent_id)
            total_offset += offset
            line = line[offset:]

        return total_offset+1, max_id, leaf_id

    def add_height(self, tree_dep):

        for n in self.tree.all_nodes():
            n.data.leaves = []

        for leaf in self.tree.leaves():
            lid = leaf.identifier
            hid = tree_dep[lid]
            if hid == self.tree.root:
                self.tree[lid].data.height = self.tree.depth(self.tree[lid])
                for cid in [p for p in self.tree.paths_to_leaves() if lid in p][0]:
                    self.tree[cid].data.leaves += [lid]
            else:
                height = -1
                cid = lid
                cond = True
                while cond:
                    self.tree[cid].data.leaves += [lid]
                    height += 1
                    cid = self.tree.parent(cid).identifier
                    cid_leaves = [l.identifier for l in self.tree.leaves(cid)]
                    cid_l_dep = [tree_dep[l] for l in cid_leaves if l != lid]
                    cond = set(cid_l_dep).issubset(set(cid_leaves))
                self.tree[lid].data.height = height

        x_nodes = [n.identifier for n in self.tree.all_nodes() if n.data.leaves == []]
        for x_node in x_nodes[::-1]:
            min_id = min(self.tree.children(x_node), key=lambda c: c.data.height)
            _lid = min_id.data.leaves[0]
            self.tree[_lid].data.height += 1
            self.tree[x_node].data.leaves += [_lid]

        return True

    def _from_tree_to_ptb(self, nid):
        nid = self.tree.subtree(nid).root
        if self.tree[nid].is_leaf():
            return  ' (' + self.tree[nid].tag + ' ' + self.tree[nid].data.word + ')'

        res = ' (' + self.tree[nid].tag

        for c_nid in sorted(self.tree.children(nid), key=lambda x: x.identifier):
            res += self._from_tree_to_ptb(c_nid.identifier)

        return res + ')'

    def from_tree_to_ptb(self):
        return self._from_tree_to_ptb(self.tree.root)

    def from_tag_to_tree(self, tag, word, pos_id=0):
        parent_id = None
        for tag_nodes in tag:
            if tag_nodes[0] in [CL, CR]:
                c_side = tag_nodes[0]
                _tag_nodes = tag_nodes[1:] if len(tag_nodes) > 1 else ['']
            else:
                c_side = ''
                _tag_nodes = tag_nodes
            self.tree.create_node(_tag_nodes[0], pos_id,
                                    parent=parent_id,
                                    data=TreeData(comb_side=c_side))

            parent_id = pos_id
            pos_id += 1
            for tag_node in _tag_nodes[1:]:
                self.tree.create_node(tag_node[1:], pos_id,
                                        parent=parent_id,
                                        data=TreeData(miss_side=tag_node[0]))
                pos_id += 1
        for l in self.tree.leaves():
            if l.data.miss_side == '':
                l.data.word = word
                break
        return pos_id

    @memoize
    def is_combine_to(self, side):
        return self.tree[self.tree.root].data.comb_side == side

    @memoize
    def is_combine_right(self):
        return self.is_combine_to(CR)

    @memoize
    def is_combine_left(self):
        return self.is_combine_to(CL)

    @memoize
    def is_complete_tree(self):
        return all([n.data.miss_side == '' for n in self.tree.all_nodes()])

    @memoize
    def get_missing_leaves_to(self, miss_val, side):
        return [l.identifier for l in self.tree.leaves(self.tree.root)
                if l.data.miss_side == side and l.tag == miss_val]

    @memoize
    def get_missing_leaves_left(self, miss_val):
        return self.get_missing_leaves_to(miss_val, L)

    @memoize
    def get_missing_leaves_right(self, miss_val):
        return self.get_missing_leaves_to(miss_val, R)

    @memoize
    def root_tag(self):
        return self.tree[self.tree.root].tag

    @memoize
    def is_no_missing_leaves(self):
        return all([l.data.miss_side == '' for l in self.tree.leaves(self.tree.root)])

    @memoize
    def combine_tree(self, _tree, comb_leaf):
        self.tree.paste(comb_leaf, _tree.tree)
        self.tree.link_past_node(comb_leaf)

    def tree_to_path(self, nid, path):

        # Stop condition
        if self.tree[nid].is_leaf():
            path[nid] = []
            return nid, self.tree[nid].data.height

        # Recursion
        flag = CR
        for child in self.tree.children(nid):
            cid = child.identifier
            leaf_id, height = self.tree_to_path(cid, path)

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

    def path_to_tags(self, path):
        tags = []
        for p in path:
            _res = []
            _p = copy.copy(p)
            if _p[0] in [CL, CR]:
                _res.append(_p[0])
                _p = _p[1:]
            while _p[:-1]:
                el_p = _p.pop(0)
                _res.append(self.tree[el_p].tag)
                for c in self.tree.children(el_p):
                    if c.identifier != _p[0]:
                        _res.append(R+c.tag if c.identifier > _p[0] else L+c.tag)
            _res.append(self.tree[_p[0]].tag)
            tags.append(_res)
        return tags

    def path_to_words(self, path):
        return [self.tree[k].tag for k in path]

    def from_tree_to_tag(self):
        path = {}
        self.tree_to_path(self.tree.root, path)
        return {'tags': self.path_to_tags(path.values()), 'words': self.path_to_words(path.keys())}

    def from_ptb_to_tag(self, line, max_id, depend):
        self.from_ptb_to_tree(line, max_id)
        self.add_height(depend)
        return self.from_tree_to_tag()

def trees_to_ptb(trees):
    return _getattr_operate_on_Narray(trees, 'from_tree_to_ptb')

def get_dependancies(fin, penn_path='code/utils/pennconverter.jar'):

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

def gen_tags(fin):

    tree_deps = get_dependancies(fin)
    with open(fin) as f:
        # lines = [x.strip('\n') for x in f.readlines()]
        for i, line in enumerate(f.readlines()):

            line = line.strip('\n')
            #max_id is the number of words in line + 1.
            # This is index kept in order to number words from 1 to num of words
            max_id = len(Tree_.fromstring(line).leaves()) + 1
            line = line.replace('(', ' ( ').replace(')', ' ) ').split()
            tree = TreeT()
            return tree.from_ptb_to_tag(line, max_id, tree_deps[i])
