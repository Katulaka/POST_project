from astar import AStar
from utils.tags.tag_node import TagNode

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

def solve_tree_search(tags, num_goals, time_out, verbose=1):
    max_lid = len(tags)
    start = [TagNode(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = TagNode(0, max_lid, [])
    # let's solve it
    paths = Solver(tags).astar(start, goal, num_goals, time_out, verbose)
    trees_res = []
    for path in paths:
        path = list(path)[-1]
        trees_res.append(path.tree[0])
    return trees_res
