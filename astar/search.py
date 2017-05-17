from Astar import AStar
import numpy as np

from itertools import permutations, product

 

# s1  = {'gscore': .0, 'fscore': .0, 'data' : {'rid' : 0, 'lid' : 1, 'rank' : [1]}}
# s2  = {'gscore': .0, 'fscore': .0, 'data' : {'rid' : 2, 'lid' : 4, 'rank' : [1,1,1]}}
# s3  = {'gscore': .0, 'fscore': .0, 'data' : {'rid' : 0, 'lid' : 4, 'rank' : [1]*5}}
#
# cl = ClosedList()
#    
# cl.put(AStar.State(**s1))
# cl.put(AStar.State(**s2))
# cl.put(AStar.State(**s3))

class Tag(object):
    def __init__(self, tag, score):
        self.tag = tag
        self.score = score
    
        
tags = np.empty((3, 5), object)
    
tvals = np.random.rand(3,5)
tvals /= tvals.sum(axis=0)

for i in range(3):
    for j in range(5):
        tags[i][j] = Tag(str(i+j), tvals[i][j])


class Tree:
    
    def __init__(self, rid, lid, rank):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
  #      self.successors = []


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

        def get(self, rid):
            return [] if not rid in self.rindex else self.rindex[rid]
            

    def __init__(self, tags):
        self.tags = tags
        self.cl = Solver.ClosedList()

    def heuristic_cost(self, current, goal):
        idx_range = range(0, current.rid)+range(current.lid, goal.lid) 
        rank = [0]*len(idx_range)
        pos = zip(rank, idx_range)
        return sum([tags[el].score for el in pos])

    def is_valid(self):
            return True  # TODO
    
    def real_cost(self, current):
        
        idx_range = range(current.rid, current.lid)
        pos = zip(current.rank, idx_range)
        return 0 if not self.is_valid() else sum([tags[el].score for el in pos])

    def move_to_closed(self, current):
        self.cl.put(current)
    
    def neighbors(self, node):
        
        neighbors = [Tree(node.rid, nb.lid, node.rank+nb.rank) for nb in self.cl.get(node.lid)]
        if len(node.rank) == 1 and node.rank[0] < self.tags.shape[0] - 1:
            node_rp1 = Tree(node.rid, node.lid, [node.rank[0] + 1])      
            neighbors.append(node_rp1)
        return neighbors 
        
    def is_goal_reached(self, current, goal):
        return current.idx == goal.idx #and current.rank == goal.rank


def solve_treeSearch(tags):
    
    max_rank, max_lid = tags.shape

    trees = []
    trees.extend([Tree(idx[0], idx[1], list(rank)) for idx in permutations(xrange(max_lid+1), 2) if idx[0] < idx[1]
                  for rank in product(xrange(max_rank), repeat=idx[1]-idx[0])])
    
    start = [trees[i] for i, x in enumerate(trees) if len(x.rank) == 1 and x.rank[0] == 0]
    
    goal = [trees[i] for i, x in enumerate(trees) if x.rank == [max_rank-1]*max_lid]  # TODO

    # let's solve it
    foundPath = list(Solver(tags).astar(start, goal[0]))
    print tvals    
    print foundPath[-1].idx, foundPath[-1].rank

solve_treeSearch(tags)
