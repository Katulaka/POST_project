import cPickle
import copy

from tag_symbols import R, L, CR, CL, ANY

def fast_copy(src):
    return cPickle.loads(cPickle.dumps(src))

class TagNode(object):

    def __init__(self, rid, lid, rank, tree=[]):
        self.rid = rid
        self.lid = lid
        self.idx = (rid, lid)
        self.rank = rank
        self.tree = tree

    def __eq__(self, other):
        return self.rank == other.rank and self.idx == other.idx

    def combine_trees(self):
        ptr = 0
        if len(self.tree) > 2:
            import pdb; pdb.set_trace()
        trees_cp = copy.copy(self.tree)
        while ptr < len(trees_cp)-1:
            combine = False
            t_l = trees_cp[ptr]
            t_r = trees_cp[ptr+1]
            if t_l[t_l.root].data.comb_side == CR and \
                all([n.data.miss_side == '' for n in t_l.all_nodes()]):
                root = t_l[t_l.root]
                leaves = [l for l in t_r.leaves(t_r.root) if l.data.miss_side == L]
                if leaves:
                    leaves_tags = map(lambda x: x.tag, leaves)
                    if ANY in leaves_tags:
                        root_leaf_id = len(leaves_tags) - leaves_tags[::-1].index(ANY) - 1
                    elif root.tag in leaves_tags:
                        root_leaf_id = leaves_tags.index(root.tag)
                    leaf_id = leaves[root_leaf_id].identifier
                    t_r_cp = fast_copy(t_r)
                    t_r_cp.paste(leaf_id, t_l)
                    t_r_cp.link_past_node(leaf_id)
                    trees_cp[ptr+1] = t_r_cp
                    del trees_cp[ptr]
                    if ptr > 0: ptr -= 1
                    combine = True

            if not combine and t_r[t_r.root].data.comb_side == CL and \
                all([n.data.miss_side == '' for n in t_r.all_nodes()]):
                root = t_r[t_r.root]
                leaves = [l for l in t_l.leaves(t_l.root) if l.data.miss_side == R]
                if leaves:
                    leaves_tags = map(lambda x: x.tag, leaves)
                    if ANY in leaves_tags:
                        root_leaf_id = len(leaves_tags) - leaves_tags[::-1].index(ANY) - 1
                    elif root.tag in leaves_tags:
                        root_leaf_id = leaves_tags.index(root.tag)
                    try:
                        leaf_id = leaves[root_leaf_id].identifier
                        t_l_cp = fast_copy(t_l)
                        t_l_cp.paste(leaf_id, t_r)
                        t_l_cp.link_past_node(leaf_id)
                        trees_cp[ptr] = t_l_cp
                        del trees_cp[ptr+1]
                        if ptr > 0: ptr -= 1
                        combine = True
                    except:
                        import pdb; pdb.set_trace()

            if not combine:
                ptr += 1
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
