from utils.tags.tree_t import TreeT

class TreeTS(object):

    def __init__(self, tag, score, word, pos_id=0):
        self.tag = tag
        self.score = score
        self.tree = TreeT()
        self.max_pos_id = self.tree.from_tag_to_tree(tag, word, pos_id)

def convert_to_TreeTS(tag_score_mat, words):
    tree_t_s_mat = []
    pos_id = 0
    for tag_score_row, word in zip(tag_score_mat, words):
        tree_t_s_row = []
        for tag, score in tag_score_row:
            tree_t_s_row.extend([TreeTS(tag, score, word, pos_id)])
            pos_id = tree_t_s_row[-1].max_pos_id
        tree_t_s_mat.append(tree_t_s_row)
    return tree_t_s_mat
