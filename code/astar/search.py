import copy
import numpy as np
import re
from sets import Set
from treelib import Node, Tree

from astar import AStar
from utils.gen_tags import extend_path, R, L, CR, CL, ANY

class TagTree(object):

    class Prop(object):
        def __init__(self, miss_side='', comb_side=''):
            self.miss_side = miss_side
            self.comb_side = comb_side

    def __init__(self, tag, score, pos_id=0):
        self.tag = tag
        self.score = score
        self.tree = Tree()
        parent_id = None
        for tag_nodes in self.tag:
            if tag_nodes[0] in [CL, CR]:
                c_side = tag_nodes[0]
                _tag_nodes = tag_nodes[1:] if len(tag_nodes) > 1 else ['']
            else:
                c_side = ''
                _tag_nodes = tag_nodes

            self.tree.create_node(_tag_nodes[0], pos_id, parent=parent_id,
                                data=TagTree.Prop(comb_side=c_side))

            parent_id = pos_id
            pos_id += 1
            for tag_node in _tag_nodes[1:]:
                self.tree.create_node(tag_node[1:], pos_id, parent=parent_id,
                                    data=TagTree.Prop(miss_side=tag_node[0]))
                pos_id += 1
        self.max_pos_id = pos_id


class TagNode(object):

    def __init__(self, rid, lid, rank, tree=[]):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
        self.tree = tree

    def __eq__(self, other):
        return self.idx == other.idx and self.rank == other.rank

    def combine_trees(self, trees):
        ptr = 0

        trees_cp = copy.copy(trees)
        while ptr < len(trees_cp)-1:
            combine = False
            t_l = trees_cp[ptr]
            t_r = trees_cp[ptr+1]
            if t_l[t_l.root].data.comb_side == CR and \
                all([n.data.miss_side == '' for n in t_l.all_nodes()]):
                root = t_l[t_l.root]
                leaves = [l for l in t_r.leaves(t_r.root) if l.data.miss_side == L]
                if leaves:
                    leaves_tags = map(lambda x: x.tag, leaves)
                    if ANY in leaves_tags:
                        root_leaf_id = len(leaves_tags) - leaves_tags[::-1].index(ANY) - 1
                    elif root.tag in leaves_tags:
                        root_leaf_id = leaves_tags.index(root.tag)
                    leaf_id = leaves[root_leaf_id].identifier
                    t_r_cp = copy.deepcopy(t_r)
                    t_r_cp.paste(leaf_id, t_l)
                    t_r_cp.link_past_node(leaf_id)
                    trees_cp[ptr+1] = t_r_cp
                    del trees_cp[ptr]
                    if ptr > 0: ptr -= 1
                    combine = True

            if not combine and t_r[t_r.root].data.comb_side == CL and \
                all([n.data.miss_side == '' for n in t_r.all_nodes()]):
                root = t_r[t_r.root]
                leaves = [l for l in t_l.leaves(t_l.root) if l.data.miss_side == R]
                if leaves:
                    leaves_tags = map(lambda x: x.tag, leaves)
                    if ANY in leaves_tags:
                        root_leaf_id = len(leaves_tags) - leaves_tags[::-1].index(ANY) - 1
                    elif root.tag in leaves_tags:
                        root_leaf_id = leaves_tags.index(root.tag)
                    leaf_id = leaves[root_leaf_id].identifier
                    t_l_cp = copy.deepcopy(t_l)
                    t_l_cp.paste(leaf_id, t_r)
                    t_l_cp.link_past_node(leaf_id)
                    trees_cp[ptr] = t_l_cp
                    del trees_cp[ptr+1]
                    if ptr > 0: ptr -= 1
                    combine = True

            if not combine:
                ptr += 1
        return trees_cp

    def _combine_trees(self, trees):
        ptr = 0
        trees_cp = copy.deepcopy(trees)
        while ptr < len(trees_cp)-1:
            root_leaf_id = []
            sub_tag_trees = trees_cp[ptr : ptr+2]
            roots = []
            for t, s in zip(sub_tag_trees, (CR, CL)):
                #check if 1st tree combines to the right and
                # 2nd tree combine combines to the left
                #and also take trees that dont have missing nodes
                if t[t.root].data.comb_side == s and \
                    all([n.data.miss_side == '' for n in t.all_nodes()]):
                    roots.append(t[t.root])
                else:
                    roots.append(None)
            #TODO skip rest of code if both roots are None
            # roots = [t[t.root] if t[t.root].data.comb_side == s else None
            #             for t,s in zip(sub_tag_trees, (CR, CL))]
            #TODO roots that don't have missing nodes
            leaves = [[l for l in t.leaves(t.root) if l.data.miss_side == s]
                        for t, s in zip(sub_tag_trees, (R, L)) ]
            for r, ls in zip(roots, leaves[::-1]):
                leaves_tags = map(lambda x: x.tag, ls)
                if r is not None and ANY in leaves_tags:
                    root_leaf_id.append(len(leaves_tags) - leaves_tags[::-1].index(ANY) - 1)
                elif r is not None and r.tag in leaves_tags:
                    root_leaf_id.append(leaves_tags.index(r.tag))
                else:
                    root_leaf_id.append(None)
            if root_leaf_id[1] is not None:
                leaf_id = leaves[0][root_leaf_id[1]].identifier
                trees_cp[ptr].paste(leaf_id, trees_cp[ptr+1])
                trees_cp[ptr].link_past_node(leaf_id)
                del trees_cp[ptr+1]
                if ptr > 0: ptr -= 1
            elif root_leaf_id[0] is not None:
                leaf_id = leaves[1][root_leaf_id[0]].identifier
                trees_cp[ptr+1].paste(leaf_id, trees_cp[ptr])
                trees_cp[ptr+1].link_past_node(leaf_id)
                del trees_cp[ptr]
                if ptr > 0: ptr -= 1
            else:
                ptr += 1
        return trees_cp

    def is_valid(self, tags):
        if self.tree == [] and len(self.rank) == 1:
            self.tree = [tags[self.rid][self.rank[0]].tree]
            return True
        if len(self.tree) == 1:
            return True
        ct = self.combine_trees(self.tree)
        if len(ct) == 1 :
            self.tree = ct
            return True
        return False


class Solver(AStar):

    class ClosedList(object):

        def __init__(self):
            self.rindex = {}
            self.lindex = {}

        def put(self, node):
            if node.rid in self.rindex:
                if node not in self.rindex[node.rid]:
                    self.rindex[node.rid].append(node)
            else:
                self.rindex[node.rid] = [node]

            if node.lid in self.lindex:
                if node not in self.lindex[node.lid]:
                    self.lindex[node.lid].append(node)
            else:
                self.lindex[node.lid] = [node]

        def getr(self, rid):
            return self.rindex.get(rid, [])

        def getl(self, lid):
            return self.lindex.get(lid, [])

    def __init__(self, tags):
        self.tags = tags
        self.cl = Solver.ClosedList()
        self.seen = []

    def heuristic_cost(self, current, goal):
        idx_range = range(current.rid) + range(current.lid, goal.lid)
        rank = [0] * len(idx_range)
        pos = zip(idx_range ,rank)
        return sum([self.tags[rng][rnk].score for rng, rnk in pos])

    def real_cost(self, current):
        if current.is_valid(self.tags):
            idx_range = range(current.rid, current.lid)
            pos = zip(idx_range, current.rank)
            return sum([self.tags[rng][rnk].score for rng, rnk in pos])
        return .0

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getr(node.lid):
            nb_node = TagNode(node.rid, nb.lid, node.rank+nb.rank, node.tree + nb.tree)
            if nb_node.is_valid(self.tags) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getl(node.rid):
            nb_node = TagNode(nb.rid, node.lid, nb.rank+node.rank, nb.tree + node.tree)
            if nb_node.is_valid(self.tags) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] < len(self.tags[node.rid]) - 1:
            nb_node = TagNode(node.rid, node.lid, [node.rank[0] + 1])
            if nb_node.is_valid(self.tags) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal):
        if current.idx == goal.idx and len(current.tree) == 1:
            ct = current.tree[0]
            return all([l.data.miss_side == '' for l in ct.leaves(ct.root)])
        return False

def convert_to_TagTree(tag_matrix):
    tags = []
    pos_id = 0
    for tm in tag_matrix:
        tag_row = []
        for el in tm:
            tag_row.extend([TagTree(el[0], el[1], pos_id)])
            pos_id = tag_row[-1].max_pos_id
        tags.append(tag_row)
    return tags

def find_tag(tree):
    sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
    new_tag = []
    while (sorted_nodes):
        _new_tag =[]
        if tree[sorted_nodes[0]].is_leaf():
            _new_tag.append(tree[sorted_nodes[0]].tag)
            del sorted_nodes[0]
        else:
            while (not tree[sorted_nodes[0]].is_leaf()):
                ch = [R+c.tag if sorted_nodes[1] < c.identifier else L+c.tag
                            for c in tree.siblings(sorted_nodes[1])]
                _new_tag.append(''.join([tree[sorted_nodes[0]].tag] + ch))
                del sorted_nodes[0]
            _new_tag.append(tree[sorted_nodes[0]].tag)
            del sorted_nodes[0]
        new_tag.append(_new_tag)
    return new_tag


def _find_tag(tree):
    sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
    new_tag = []
    while (sorted_nodes):
        sub_tag = []
        if tree[sorted_nodes[0]].data.comb_side != '':
            sub_tag = [tree[sorted_nodes[0]].data.comb_side]
        while (not tree[sorted_nodes[0]].is_leaf()):
            sub_tag += [sorted_nodes[0]]
            del sorted_nodes[0]
        sub_tag += [sorted_nodes[0]]
        del sorted_nodes[0]
        new_tag.append(sub_tag)
    return extend_path(tree, new_tag)

def solve_tree_search(tag_matrix, verbose, num_goals):
    tags = convert_to_TagTree(tag_matrix)
    max_lid = len(tag_matrix)
    start = [TagNode(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = TagNode(0, max_lid, [])
    # let's solve it
    paths = Solver(tags).astar(start, goal, num_goals, verbose = verbose)
    trees_res = []
    tags_res = []
    for path in paths:
        path = list(path)[-1]
        trees_res.append(path.tree[0])
        tags_res.append(_find_tag(path.tree[0]))
    return trees_res, tags_res
