from Astar import AStar
import numpy as np
from treelib import Node, Tree
import copy


class Tag(object):
    def __init__(self, tag, score, pos_id=0):
        self.tag = tag
        self.score = score
        self.tree, self.pos_id = self.convert_to_tree(pos_id)

    class nProp(object):
        def __init__(self, miss):
            self.miss = miss

    def convert_to_tree(self, pos_id):
        tree = Tree()
        parent_id = None
        for t in [subTag.split('\\') for subTag in self.tag.split('+')]:
        # for t in [subTag.split('\\') for subTag in self.tag]:
            tree.create_node(t[0], pos_id, parent = parent_id, data = Tag.nProp(False))
            parent_id = pos_id
            pos_id += 1
            for n in t[1:]:
                tree.create_node(n, pos_id, parent = parent_id, data = Tag.nProp(True))
                pos_id += 1
        return tree, pos_id

class TagNode(object):

    def __init__(self, rid, lid, rank):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank

    def combine_trees(self, trees):
        ptr = 0
        tagtrees = copy.deepcopy(trees)
        while ptr<len(tagtrees)-1:
            subtagtrees = tagtrees[ptr:ptr+2]
            roots = [t[t.root] for t in subtagtrees]
            leaves = [ [l for l in t.leaves(t.root) if l.data.miss] for t in subtagtrees]
            if roots[0].tag in map(lambda l: l.tag, leaves[1]) and \
            len(leaves[0]) == 0:
                idm = map(lambda l: l.tag, leaves[1]).index(roots[0].tag)
                tagtrees[ptr+1].paste(leaves[1][idm].identifier, tagtrees[ptr])
                tagtrees[ptr+1].link_past_node(leaves[1][idm].identifier)
                del tagtrees[ptr]
                if ptr > 0 : ptr -= 1
            elif roots[1].tag in map(lambda l: l.tag, leaves[0]) and \
            len(leaves[1]) == 0:
                idm = map(lambda l: l.tag, leaves[0]).index(roots[1].tag)
                tagtrees[ptr].paste(leaves[0][idm].identifier, tagtrees[ptr+1])
                tagtrees[ptr].link_past_node(leaves[0][idm].identifier)
                del tagtrees[ptr+1]
                if ptr > 0 : ptr -= 1
            else:
                ptr += 1
        return tagtrees

    def is_valid(self, tags):
        trees = [tags[self.rank[i-self.rid]][i].tree for i in xrange(self.rid,self.lid)]
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
            return [] if not rid in self.rindex else self.rindex[rid]
            #return self.rindex.get(rid, []) #TODO shortcut

        def getl(self, lid):
            return [] if not lid in self.lindex else self.lindex[lid]

    def __init__(self, tags):
        self.tags = np.array(tags)
        self.cl = Solver.ClosedList()

    def heuristic_cost(self, current, goal):
        idx_range = range(0, current.rid)+range(current.lid, goal.lid)
        rank = [0]*len(idx_range)
        pos = zip(rank, idx_range)
        return sum([self.tags[el].score for el in pos])

    def real_cost(self, current):
        idx_range = range(current.rid, current.lid)
        pos = zip(current.rank, idx_range)
        return .0 if not current.is_valid(self.tags) else sum([self.tags[el].score for el in pos])

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = [TagNode(node.rid, nb.lid, node.rank+nb.rank)
                    for nb in self.cl.getr(node.lid) \
                    if TagNode(node.rid, nb.lid, node.rank+nb.rank).is_valid(self.tags)]

        neighbors += [TagNode(nb.rid, node.lid, nb.rank+node.rank)
                    for nb in self.cl.getl(node.rid) \
                    if TagNode(nb.rid, node.lid, nb.rank+node.rank).is_valid(self.tags)]

        if len(node.rank) == 1 and node.rank[0] < self.tags.shape[0] - 1:
            node_rp1 = TagNode(node.rid, node.lid, [node.rank[0] + 1])
            neighbors.append(node_rp1)
        return neighbors

    def is_goal_reached(self, current, goal): #TODO
        if current.idx == goal.idx:
            ct = current.tree
            return len([l for l in ct.leaves(ct.root) if l.data.miss]) == 0
        return False


def convert_to_TagC(TagMat):
    length = len(TagMat)
    tags = []
    pos_id = 0
    for tm in TagMat:
        tag_row = []
        for el in tm:
            tag_row.extend([Tag(el[0], el[1], pos_id)])
            pos_id = tag_row[-1].pos_id
        tags.append(tag_row)
    return np.array(tags).T


def solve_treeSearch(TagMat):

    tags = convert_to_TagC(TagMat)
    max_lid = len(TagMat)
    start = [TagNode(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = TagNode(0, max_lid, [])
    # let's solve it
    foundPath = Solver(tags).astar(start, goal)
    if not foundPath is None:
        foundPath = list(foundPath)[-1]
        return zip(xrange(foundPath.idx[1]), foundPath.rank)
    return foundPath


    # def toy_example(rank, tagsVal):
    #     length = len(tagsVal)
    #     tags = np.empty((rank, length), object)
    #
    #     tvals = np.random.rand(rank, length)
    #     tvals /= -tvals.sum(axis=0)
    #     tvals.sort(axis=0)
    #     tvals = -tvals
    #
    #     pos_id = 0
    #     for i in range(rank):
    #         for j in range(length):
    #             if i == 1:
    #                 tags[i][j] = Tag(tagsVal[j], tvals[i][j], pos_id)
    #             else:
    #                 tags[i][j] = Tag('NP\\NNP+NNP', tvals[i][j], pos_id)
    #             pos_id = tags[i][j].pos_id
    #     return tvals, tags
