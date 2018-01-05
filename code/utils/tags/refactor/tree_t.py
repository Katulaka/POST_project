import copy
from treelib import Tree
from tag_ops import R, L, CL, CR

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
            offset, max_id, leaf_id = self.from_ptb_to_tree(self, line, max_id, leaf_id, parent_id)
            total_offset += offset
            line = line[offset:]

        return total_offset+1, max_id, leaf_id

    def add_height(self, tree_dep):

        for n in self.tree.all_nodes():
            n.data.leaves = []

        for leaf in self.tree.leaves():
            lid = leaf.identifier
            hid = tree_dep[lid]
            if hid == tree.root:
                self.tree[lid].data.height = self.tree.depth(tree[lid])
                for cid in [p for p in tree.paths_to_leaves() if lid in p][0]:
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
        # nid = tree.root
        nid = self.tree.subtree(nid).root
        if self.tree[nid].is_leaf():
            return  ' (' + self.tree[nid].tag + ' ' + self.tree[nid].data.word + ')'

        res = ' (' + tree[nid].tag

        for c_nid in sorted(self.tree.children(nid), key=lambda x: x.identifier):
            res += self.from_tree_to_ptb(self, c_nid.identifier)

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

    def is_combine_to(self, side):
        return self.tree[self.tree.root].data.comb_side == side

    def is_combine_right(self):
        return self.is_combine_to(CR)

    def is_combine_left(self):
        return self.is_combine_to(CL)

    def is_complete_tree(self):
        return all([n.data.miss_side == '' for n in self.tree.all_nodes()])

    def get_missing_leaves_to(self, miss_val, side):
        return [l.identifier for l in self.tree.leaves(self.tree.root)
                if l.data.miss_side == side and l.tag == miss_val]

    def get_missing_leaves_left(self, miss_val):
        return self.get_missing_leaves_to(miss_val, L)

    def get_missing_leaves_right(self, miss_val):
        return self.get_missing_leaves_to(miss_val, R)

    def root_tag(self):
        return self.tree[self.tree.root].tag

    def is_no_missing_leaves(self):
        all([l.data.miss_side == '' for l in self.tree.leaves(self.tree.root)])

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
            leaf_id, height = self.tree_to_path(self, cid, path)

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
            self.tree_to_path(tree.root, path)
            return (self.path_to_tags(path.values()), self.path_to_words(path.keys()))