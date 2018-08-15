import numpy as np
from .astar import AStar
from .node_t import NodeT
from tree_t import TreeT

import itertools

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

    def __init__(self, ts_mat, no_val_gap):
        self.ts_mat = ts_mat
        self.miss_tag_any = no_val_gap
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

    def heuristic_cost(self, current, goal, cost_coeff):
        r_rng = [0] if current.rid == 0 else list(range(current.rid))
        l_rng = list(range(current.lid, goal.lid))
        idx_range = r_rng + l_rng
        return cost_coeff * sum([self.ts_mat[rng][0]['score'] for rng in idx_range])

    def real_cost(self, current):

        if current.is_valid(self.ts_mat[current.rid][current.rank[0]]['tree'], self.miss_tag_any):
            idx_range = list(range(current.rid, current.lid))
            pos = zip(idx_range, current.rank)
            return sum([self.ts_mat[rng][rnk]['score'] for rng, rnk in pos])
        return .0

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getr(node.lid):
            nb_node = NodeT(node.rid, nb.lid, node.rank+nb.rank, node.tree + nb.tree)
            if nb_node.is_valid(self.ts_mat[nb_node.rid][nb_node.rank[0]]['tree'], self.miss_tag_any) and nb_node not in self.seen:

                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getl(node.rid):
            nb_node = NodeT(nb.rid, node.lid, nb.rank+node.rank, nb.tree + node.tree)
            # if nb_node.is_valid(self.ts_mat[nb_node.rid][nb_node.rank[0]].tree, self.miss_tag_any) and nb_node not in self.seen:
            if nb_node.is_valid(self.ts_mat[nb_node.rid][nb_node.rank[0]]['tree'], self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] < len(self.ts_mat[node.rid]) - 1:
            nb_node = NodeT(node.rid, node.lid, [node.rank[0] + 1])
            # if nb_node.is_valid(self.ts_mat[nb_node.rid][nb_node.rank[0]].tree, self.miss_tag_any) and nb_node not in self.seen:
            if nb_node.is_valid(self.ts_mat[nb_node.rid][nb_node.rank[0]]['tree'], self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal):
        if current.idx == goal.idx and len(current.tree) == 1:
            return current.tree[0].is_no_missing_leaves()
        return False

def solve_tree_search(tag_score_mat, words, no_val_gap, num_goals, time_out,
                        time_th=2. , verbose=1):

    tree_score_mat = []
    pos_id = 0
    # import pdb; pdb.set_trace()
    for tag_score_row, word in zip(tag_score_mat, words):
        tree_score_row = []
        for tag, score in tag_score_row:
            tree = TreeT()
            pos_id = tree.from_tag_to_tree(tag, word, pos_id)
            tree_score_row.extend([{'tree': tree, 'score': score}])
        if tree_score_row == []:
            return []
        tree_score_mat.append(tree_score_row)

    # import pdb; pdb.set_trace()
    # ts_mat = convert_to_TreeTS(tag_score_mat, words)
    # if any([t == [] for t in ts_mat]):
    #     return []
    max_lid = len(tree_score_mat)
    max_rank = max([len(t) for t in tree_score_mat])
    start = [NodeT(idx, idx+1, [0]) for idx in range(max_lid)]
    goal = NodeT(0, max_lid, [])
    # let's solve it
    solve = Solver(tree_score_mat, no_val_gap)
    paths, max_path = solve.astar(start, goal, num_goals, time_out)
    trees_res = []
    patterns = []
    for path in paths:
        path = list(path)[-1]
        trees_res.append(path.tree[0])
        # pattern = np.concatenate(([np.ones(max_lid, dtype=int)],np.zeros((max_rank-1,max_lid), dtype=int)))
        # for s in solve.seen :
        #     pattern[s.rank, range(*s.idx)] += 1
        # patterns.append(pattern)
    # return trees_res, patterns
    return trees_res, paths
