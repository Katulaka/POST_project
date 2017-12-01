import copy as copy

from tag_symbols import R, L, CL, CR
from tags_to_trees import get_trees
from utils.utils import operate_on_Narray, _operate_on_Narray

def extend_path(tree, path):
    tags = []
    for p in path:
        _res = []
        _p = copy.copy(p)
        if _p[0] in [CL, CR]:
            _res.append(_p[0])
            _p = _p[1:]
        while _p[:-1]:
            el_p = _p.pop(0)
            _res.append(tree[el_p].tag)
            for c in tree.children(el_p):
                if c.identifier != _p[0]:
                    _res.append(R+c.tag if c.identifier > _p[0] else L+c.tag)
        _res.append(tree[_p[0]].tag)
        tags.append(_res)
    return tags

def gen_tag(tree, nid, path):

    # Stop condition
    if tree[nid].is_leaf():
        path[nid] = []
        return nid, tree[nid].data.height

    # Recursion
    flag = CR
    for child in tree.children(nid):
        cid = child.identifier
        leaf_id, height = gen_tag(tree, cid, path)

        if (height == 0):
            # Reached end of path can add flag
            path[leaf_id].insert(0, flag)
            # path[leaf_id].append(flag)

        if height > 0:
            path[leaf_id].insert(0, nid)
            # only single child will have height>0
            # and its value will be the one that is returned
            # to the parent
            ret_leaf_id, ret_height = leaf_id, height-1

            # once we reached a height>0, it means that
            # this path includes the parent, and thus flag
            # direction should flip
            flag = CL

    return ret_leaf_id, ret_height

def gen_tags(fin):
    print ("gen_tags")

    for i, tree in enumerate(get_trees(fin)):
        path = {}
        try:
            gen_tag(tree, tree.root, path)
            yield (extend_path(tree, path.values()),
                            [tree[key].tag for key in path.keys()])
        except:
            print ("Wrong tree %d in %s" % (i, fin))


def _to_mrg(tree):
    nid = tree.root
    if tree[nid].is_leaf():
        return  ' (' + tree[nid].tag + ' ' + tree[nid].data.word + ')'

    res = ' (' + tree[nid].tag

    for c_nid in sorted(tree.children(nid), key=lambda x: x.identifier):
        res += to_mrg(tree.subtree(c_nid.identifier))

    return res + ')'

def to_mrg(trees):
    return _operate_on_Narray(trees, _to_mrg)


def find_tag(tree):
    sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
    new_tag = []
    while (sorted_nodes):
        _new_tag =[]
        if tree[sorted_nodes[0]].is_leaf():
            _new_tag.append(tree[sorted_nodes[0]].tag)
            del sorted_nodes[0]
        else:
            while (not tree[sorted_nodes[0]].is_leaf()):
                ch = [R+c.tag if sorted_nodes[1] < c.identifier else L+c.tag
                            for c in tree.siblings(sorted_nodes[1])]
                _new_tag.append(''.join([tree[sorted_nodes[0]].tag] + ch))
                del sorted_nodes[0]
            _new_tag.append(tree[sorted_nodes[0]].tag)
            del sorted_nodes[0]
        new_tag.append(_new_tag)
    return new_tag


def _find_tag(tree):
    sorted_nodes = sorted([t.identifier for t in tree.all_nodes()])
    new_tag = []
    while (sorted_nodes):
        sub_tag = []
        if tree[sorted_nodes[0]].data.comb_side != '':
            sub_tag = [tree[sorted_nodes[0]].data.comb_side]
        while (not tree[sorted_nodes[0]].is_leaf()):
            sub_tag += [sorted_nodes[0]]
            del sorted_nodes[0]
        sub_tag += [sorted_nodes[0]]
        del sorted_nodes[0]
        new_tag.append(sub_tag)
    return extend_path(tree, new_tag)
