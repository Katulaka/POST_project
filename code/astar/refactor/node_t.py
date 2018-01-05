import cPickle
import copy

from utils.tags.tag_ops import R, L, CR, CL, ANY
from utils.conf import Config

def fast_copy(src):
    return cPickle.loads(cPickle.dumps(src))

class NodeT(object):

    def __init__(self, rid, lid, rank, tree=[]):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
        self.tree = tree #list of TreeT's

    def __eq__(self, other):
        return self.rank == other.rank and self.idx == other.idx

    def combine_pair(self, t_dst, t_src, comb_side, miss_side):
        if t_src.is_combine_to(comb_side) and t_src.is_complete_tree():
            miss_tag = ANY if Config.no_val_gap else t_src.root_tag
            leaves = t_dst.get_missing_leaves_to(miss_tag, miss_side)
            if leaves:
                t_dst_cp = fast_copy(t_dst)
                t_dst_cp.combine_tree(t_src, leaves[-1])
                return t_dst_cp
        return None

    def combine_trees(self):
        ptr = 0
        if len(self.tree) > 2:
            import pdb; pdb.set_trace()
        trees_cp = copy.copy(self.tree)
        while ptr < len(trees_cp)-1:
            t_l = trees_cp[ptr]
            t_r = trees_cp[ptr+1]
            #try combining left tree into right tree
            t_comb = self.combine_pair(t_r, t_l, CR, L)
            if t_comb:
                trees_cp[ptr+1] = t_comb
                del trees_cp[ptr]
                if ptr > 0: ptr -= 1
            else:
                #try combining right tree into left tree
                t_comb = self.combine_pair(t_l, t_r, CL, R)
                if t_comb:
                    trees_cp[ptr] = t_comb
                    del trees_cp[ptr+1]
                    if ptr > 0: ptr -= 1
                else:
                    ptr += 1

            # combine = False
            # t_l = trees_cp[ptr]
            # t_r = trees_cp[ptr+1]
            #
            # if t_l.is_combine_right() and t_l.is_complete_tree():
            #     miss_tag = ANY if Config.no_val_gap else t_l.root_tag
            #     leaves = t_r.get_missing_leaves_left(miss_val)
            #     if leaves:
            #         t_r_cp = fast_copy(t_r)
            #         t_r_cp.combine_tree(t_l, leaves[-1])
            #         trees_cp[ptr+1] = t_r_cp
            #         del trees_cp[ptr]
            #         if ptr > 0: ptr -= 1
            #         combine = True

            # if not combine and t_r.is_combine_left() and t_r.is_complete_tree():
            #     miss_tag = ANY if Config.no_val_gap else t_r.root_tag
            #     leaves = t_l.get_missing_leaves_right(miss_tag)
            #     if leaves:
            #         t_l_cp = fast_copy(t_l)
            #         t_l_cp.combine_tree(t_r, leaves[-1])
            #         trees_cp[ptr] = t_l_cp
            #         del trees_cp[ptr+1]
            #         if ptr > 0: ptr -= 1
            #         combine = True

            # if not combine:
            #     ptr += 1
        return trees_cp

    def is_valid(self, tags):
        if self.tree == [] and len(self.rank) == 1:
            self.tree = [tags[self.rid][self.rank[0]].tree]
            return True
        if len(self.tree) == 1:
            return True
        ct = self.combine_trees()
        if len(ct) == 1 :
            self.tree = ct
            return True
        return False
