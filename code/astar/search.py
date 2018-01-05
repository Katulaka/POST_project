from astar import AStar
old = False
if old:
    from node_t import NodeT
    from tag_tree import convert_to_TagTree
else:
    from refactor.node_t import NodeT #TODO new format
    from refactor.tree_t_s import convert_to_TreeTS #TODO new format

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

    def __init__(self, ts_mat):
        self.ts_mat = ts_mat
        self.cl = Solver.ClosedList()
        self.seen = []

    def is_max_len(self, current, max_len, max_node):
        if len(current.rank) > max_len:
            return len(current.rank), current
        else:
            return max_len, max_node

    def print_fn(self, current, name):
        print ('%s: range %s, rank %s, score %f'
                %(name, current.data.idx, current.data.rank, current.fscore))

    def heuristic_cost(self, current, goal):
        idx_range = range(current.rid) + range(current.lid, goal.lid)
        rank = [0] * len(idx_range)
        pos = zip(idx_range ,rank)
        return sum([self.ts_mat[rng][rnk].score for rng, rnk in pos])

    def real_cost(self, current):
        if current.is_valid(self.ts_mat):
            idx_range = range(current.rid, current.lid)
            pos = zip(idx_range, current.rank)
            return sum([self.ts_mat[rng][rnk].score for rng, rnk in pos])
        return .0

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getr(node.lid):
            nb_node = NodeT(node.rid, nb.lid, node.rank+nb.rank, node.tree + nb.tree)
            if nb_node.is_valid(self.ts_mat) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getl(node.rid):
            nb_node = NodeT(nb.rid, node.lid, nb.rank+node.rank, nb.tree + node.tree)
            if nb_node.is_valid(self.ts_mat) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] < len(self.ts_mat[node.rid]) - 1:
            nb_node = NodeT(node.rid, node.lid, [node.rank[0] + 1])
            if nb_node.is_valid(self.ts_mat) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal):
        if current.idx == goal.idx and len(current.tree) == 1:
            ct = current.tree[0]
            if old:
                return all([l.data.miss_side == '' for l in ct.leaves(ct.root)])
            return ct.is_no_missing_leaves() #TODO new format
        return False

def solve_tree_search(tag_score_mat, words, num_goals, time_out, verbose=1):
    if old:
        ts_mat = convert_to_TagTree(tag_score_mat, words)
    else:
        ts_mat = convert_to_TreeTS(tag_score_mat, words) #TODO new format
    max_lid = len(ts_mat)
    start = [NodeT(idx, idx+1, [0]) for idx in xrange(max_lid)]
    goal = NodeT(0, max_lid, [])
    # let's solve it
    paths, max_path = Solver(ts_mat).astar(start, goal, num_goals, time_out, verbose)
    trees_res = []
    for path in paths:
        path = list(path)[-1]
        trees_res.append(path.tree[0])
    return trees_res
