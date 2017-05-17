from Astar import AStar
import numpy as np

from itertools import permutations, product


class Tag(object):
    def __init__(self, tag, score):
        self.tag = tag
        self.score = score
    
        
class Tree:
    
    def __init__(self, rid, lid, rank):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank


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

        def getl(self, lid):
            return [] if not lid in self.lindex else self.lindex[lid]

        def exist(self, rid, lid, rank):
            set_rid = set(self.getr(rid))
            set_lid = set(self.getl(lid))
            candidates = list(set_rid.intersection(set_lid))
            for candidate in candidates:
                if candidate.rank == rank:
                    return True
            return False

    def __init__(self, tags):
        self.tags = tags
        self.cl = Solver.ClosedList()

    def heuristic_cost(self, current, goal):
        idx_range = range(0, current.rid)+range(current.lid, goal.lid) 
        rank = [0]*len(idx_range)
        pos = zip(rank, idx_range)
        return sum([self.tags[el].score for el in pos])

    def is_valid(self, node): # TODO
        
        nvalid = range(1,5)
        nrank = [0,0,0,0]
        node_range = range(node.rid, node.lid)

        if set(nvalid).issubset(set(node_range)): 
            idx = node_range.index(nvalid[0]),len(nvalid)
            return not node.rank[idx[0]:idx[0]+idx[1]] == nrank
        else:
            return True
        
    
    def real_cost(self, current):
        
        idx_range = range(current.rid, current.lid)
        pos = zip(current.rank, idx_range)
        return .0 if not self.is_valid(current) else sum([self.tags[el].score for el in pos])

    def move_to_closed(self, current):
        self.cl.put(current)
    
    def neighbors(self, node):
        
        neighbors = [Tree(node.rid, nb.lid, node.rank+nb.rank) 
                    for nb in self.cl.getr(node.lid)] 
            
        if len(node.rank) == 1 and node.rank[0] < self.tags.shape[0] - 1:
            node_rp1 = Tree(node.rid, node.lid, [node.rank[0] + 1])      
            neighbors.append(node_rp1)
        return neighbors 
        
    def is_goal_reached(self, current, goal):
        return current.idx == goal.idx #and current.rank == goal.rank


def toy_example(rank, length):
    
    tags = np.empty((rank, length), object)
    
    tvals = np.random.rand(rank, length)
    tvals /= -tvals.sum(axis=0)
    tvals.sort(axis=0)
    tvals = -tvals

    for i in range(rank):
        for j in range(length):
            tags[i][j] = Tag(str(i+j), tvals[i][j])

    return tvals, tags


def solve_treeSearch():
    
    max_rank, max_lid = (3, 5)
    tvals, tags = toy_example(max_rank, max_lid)

    start = [Tree(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = Tree(0, max_lid, [])

    # let's solve it
    foundPath = Solver(tags).astar(start, goal)
    if not foundPath is None:
        foundPath = list(foundPath)
        print tvals    
        for fp in foundPath:
            print fp.idx, fp.rank

solve_treeSearch()
