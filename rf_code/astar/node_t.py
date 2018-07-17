from tree_t import combine_trees, can_combine_trees

class NodeT(object):

    def __init__(self, rid, lid, rank, tree=[]):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
        self.tree = tree #list of TreeT's

    def __eq__(self, other):
        return self.rank == other.rank and self.idx == other.idx

    def __hash__(self):
        return id(self)

    def is_valid(self, curr_tree, miss_tag_any):
        if self.tree == [] and len(self.rank) == 1:
            self.tree = [curr_tree]
        if len(self.tree) == 1:
            return True
        # try:
        combine, r_dst_l_src = can_combine_trees(self.tree, miss_tag_any)
        # except:
        #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        if combine:
            self.tree = combine_trees(self.tree, miss_tag_any, r_dst_l_src)
            return True
        # ct = combine_trees(self.tree, miss_tag_any)
        # if len(ct) == 1 :
        #     self.tree = ct
        #     return True
        return False
