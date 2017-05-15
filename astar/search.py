from Astar import AStar
import numpy as np

from itertools import permutations, product

 
# class ClosedList(object):
#
#    def __init__(self):
#        self.rindex = {}
#        self.lindex = {}
#        return
#
#    def put(self, State):
#        if State.data['rid'] in  self.rindex:
#            self.rindex[State.data['rid']].append(State)
#        else:
#            self.rindex[State.data['rid']] = [State]
#
#        if State.data['lid'] in self.lindex:
#            self.lindex[State.data['lid']].append(State)
#        else:
#            self.lindex[State.data['lid']] = [State]
#        return
#
#    #assumes unique States
#    def get(self, rid, lid):
#        if not rid in self.rindex or not lid in self.lindex:
#            return None
#        else:
#            res = list(set(self.rindex[rid]).intersection(self.lindex[lid]))
#            return  None if len(res)==0 else res[0]
#
#
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
    
for i in range(3):
    for j in range(5):
        tags[i][j] = Tag(str(i+j), i+j)


class Solver(AStar):

    def __init__(self, tags):
        self.tags = tags

    def heuristic_cost_estimate(self, current, goal):
        """computes the 'direct' distance between two (x,y) tuples"""
        idx_range = range(0, current.rid)+range(current.lid, goal.lid) #TODO fix this eq
        rank = [0]*len(idx_range)
        pos = zip(rank, idx_range)
        return sum([tags[el].score for el in pos])

    def is_valid(self):
        """

        :rtype: object
        """
        return True  # TODO
    
    def distance_between(self, tree, tree1):
        idx_range = range(tree.rid, tree.lid)
        pos = zip(tree.rank, idx_range)
        return 0 if not self.is_valid() else sum([tags[el].score for el in pos])

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        return node.successors
        
    def is_goal_reached(self, current, goal):
        return current.idx == goal.idx and current.rank == goal.rank


class Tree:
    
    def __init__(self, rid, lid, rank):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
        self.successors = []


def solve_treeSearch(tags):
    
    max_rank, max_lid = tags.shape

    trees = []
    trees.extend([Tree(idx[0], idx[1], list(rank)) for idx in permutations(xrange(max_lid+1), 2) if idx[0] < idx[1]
                  for rank in product(xrange(max_rank), repeat=idx[1]-idx[0])])
       
    for j, t in enumerate(trees):
        t.successors.extend([trees[i] for i, x in enumerate(trees) if x.rid == t.rid and x.lid > t.lid and x.rank[:len(t.rank)] == t.rank])
        if len(t.rank) == 1 and t.rank[0] < max_rank - 1:
            t.successors.extend([trees[j + 1]])
    
    start = [trees[i] for i, x in enumerate(trees) if len(x.rank) == 1 and x.rank[0] == 0]
    
    goal = [trees[i] for i, x in enumerate(trees) if x.rank == [max_rank-1]*max_lid]  # TODO

 #   
    # let's solve it
    foundPath = list(Solver(tags).astar(start, goal[0]))
    import pdb; pdb.set_trace()
    print foundPath

solve_treeSearch(tags)
