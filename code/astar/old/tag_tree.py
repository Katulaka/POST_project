from treelib import Node, Tree
from utils.tags.tag_symbols import CR, CL


class Prop(object):
    def __init__(self, miss_side='', comb_side='', word =''):
        self.miss_side = miss_side
        self.comb_side = comb_side
        self.word = word

class TagTree(object):

    def __init__(self, tag, score, word, pos_id=0):
        self.tag = tag
        self.score = score
        self.tree = Tree()
        self.max_pos_id = pos_id

        self.buildTree(word)

    def buildTree(self, word):
        parent_id = None
        for tag_nodes in self.tag:
            if tag_nodes[0] in [CL, CR]:
                c_side = tag_nodes[0]
                _tag_nodes = tag_nodes[1:] if len(tag_nodes) > 1 else ['']
            else:
                c_side = ''
                _tag_nodes = tag_nodes
            self.tree.create_node(_tag_nodes[0], self.max_pos_id,
                                    parent=parent_id,
                                    data=Prop(comb_side=c_side))

            parent_id = self.max_pos_id
            self.max_pos_id += 1
            for tag_node in _tag_nodes[1:]:
                self.tree.create_node(tag_node[1:], self.max_pos_id,
                                        parent=parent_id,
                                        data=Prop(miss_side=tag_node[0]))
                self.max_pos_id += 1
        for l in self.tree.leaves():
            if l.data.miss_side == '':
                l.data.word = word
                break

def convert_to_TagTree(tag_matrix, sent):
    #TODO what to do about trees with only missing leaves
    #TODO what to do about trees with only more than 1 non-missing leave
    tags = []
    pos_id = 0
    for tm, word in zip(tag_matrix, sent):
        tag_row = []
        for el in tm:
            tag_row.extend([TagTree(el[0], el[1], word, pos_id)])
            pos_id = tag_row[-1].max_pos_id
        tags.append(tag_row)
    return tags
