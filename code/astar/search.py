import copy
import numpy as np
import re

from treelib import Node, Tree

from astar import AStar


class TagTree(object):

    class Miss(object):
        def __init__(self, miss, side=''):
            self.miss = miss
            self.side = side

    def __init__(self, tag, score, pos_id=0):
        self.tag = tag
        self.score = score
        self.tree = Tree()
        parent_id = None
        t_list = [t.replace('\\', '|\\').replace('/', '|/').split('|')
                                                        for t in self.tag]
        for tag_nodes in t_list:
            self.tree.create_node(tag_nodes[0], pos_id,
                                    parent=parent_id,
                                    data=TagTree.Miss(False))
            parent_id = pos_id
            pos_id += 1
            for tag_node in tag_nodes[1:]:
                self.tree.create_node(tag_node[1:], pos_id,
                                        parent=parent_id,
                                        data=TagTree.Miss(True, tag_node[0]))
                pos_id += 1
        self.max_pos_id = pos_id


class TagNode(object):

    def __init__(self, rid, lid, rank):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank

    def combine_trees(self, trees):
        ptr = 0
        trees_cp = copy.deepcopy(trees)
        while ptr < len(trees_cp)-1:
            root_leave_id = []
            sub_tag_trees = trees_cp[ptr : ptr+2]
            roots = [t[t.root] for t in sub_tag_trees]
            leaves = [[l for l in t.leaves(t.root) if l.data.miss]
                        for t in sub_tag_trees]
            for r, ls in zip(roots, leaves[::-1]):
                try:
                    root_leave_id.append(map(lambda x: x.tag, ls).index(r.tag))
                except ValueError:
                    root_leave_id.append(None)
            if root_leave_id[0] is not None and len(leaves[0]) == 0:
                leave_id = leaves[1][root_leave_id[0]].identifier
                trees_cp[ptr+1].paste(leave_id, trees_cp[ptr])
                trees_cp[ptr+1].link_past_node(leave_id)
                del trees_cp[ptr]
                if ptr > 0: ptr -= 1
            elif root_leave_id[1] is not None and len(leaves[1]) == 0:
                leave_id = leaves[0][root_leave_id[1]].identifier
                trees_cp[ptr].paste(leave_id, trees_cp[ptr+1])
                trees_cp[ptr].link_past_node(leave_id)
                del trees_cp[ptr+1]
                if ptr > 0: ptr -= 1
            else:
                ptr += 1
        return trees_cp

    def is_valid(self, tags):
        trees = []
        for i in xrange(self.rid, self.lid):
            trees.append(tags[self.rank[i-self.rid]][i].tree)
        ct = self.combine_trees(trees)
        if len(ct) == 1 :
            self.tree = ct[0]
            return True
        return False


class Solver(AStar):

    class ClosedList(object):

        def __init__(self):
            self.rindex = {}
            self.lindex = {}

        def put(self, node):
            if node.rid in self.rindex:
                self.rindex[node.rid].append(node)
            else:
                self.rindex[node.rid] = [node]

            if node.lid in self.lindex:
                self.lindex[node.lid].append(node)
            else:
                self.lindex[node.lid] = [node]

        def getr(self, rid):
            return self.rindex.get(rid, [])

        def getl(self, lid):
            return self.lindex.get(lid, [])

    def __init__(self, tags):
        self.tags = np.array(tags)
        self.cl = Solver.ClosedList()

    def heuristic_cost(self, current, goal):
        idx_range = range(current.rid) + range(current.lid, goal.lid)
        rank = [0] * len(idx_range)
        pos = zip(rank, idx_range)
        return sum([self.tags[el].score for el in pos])

    def real_cost(self, current):
        idx_range = range(current.rid, current.lid)
        pos = zip(current.rank, idx_range)
        if current.is_valid(self.tags):
            return sum([self.tags[el].score for el in pos])
        return .0

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getr(node.lid):
            nb_node = TagNode(node.rid, nb.lid, node.rank+nb.rank)
            if nb_node.is_valid(self.tags):
                neighbors.append(nb_node)
        for nb in self.cl.getl(node.rid):
            nb_node = TagNode(nb.rid, node.lid, nb.rank+node.rank)
            if nb_node.is_valid(self.tags):
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] < self.tags.shape[0] - 1:
            nb_node = TagNode(node.rid, node.lid, [node.rank[0] + 1])
            neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal): #TODO
        if current.idx == goal.idx:
            ct = current.tree
            return all([not l.data.miss for l in ct.leaves(ct.root)])
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
    return np.array(tags).T


def solve_tree_search(tag_matrix, verbose):

    tags = convert_to_TagTree(tag_matrix)
    max_lid = len(tag_matrix)
    start = [TagNode(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = TagNode(0, max_lid, [])
    # let's solve it
    path = Solver(tags).astar(start, goal, verbose = verbose)
    if path is not None:
        path = list(path)[-1]
        return [range(path.idx[1]), path.rank, [0]*len(path.rank)]
    return []
