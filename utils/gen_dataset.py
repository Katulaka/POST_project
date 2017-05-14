from __future__ import print_function
from treelib import Node, Tree
from get_rawdata import make_dir
import re
import os


def is_balanced(string):
    iparens = iter('(){}[]<>')
    parens  = dict(zip(iparens, iparens))
    closing = parens.values()
    stack   = []
    for ch in string:
        delim = parens.get(ch, None)
        if delim:
            stack.append(delim)
        elif ch in closing:
            if not stack or ch != stack.pop():
                return False
    return not stack

def read_balanced_line(fin):
    s = ""
    lines = iter(open(fin, 'r'))
    for line in lines:
        if line.strip():
            s = s + line.split('\n')[0]
            if (is_balanced(s) and s != ""):
                yield s
                s = ""

def get_tree(tree, line, max_id = 0, leaf_id = 1,  parent_id = None):
    
    # starts by ['(', 'pos']
    pos_tag     = line[1].split('-')[0].split('=')[0].split('|')[0]
    if  parent_id == None:
        pos_id = 0
    else:
        pos_id = max_id
        max_id      += 1

    tree.create_node(pos_tag, pos_id, parent = parent_id, data = nRange(0,0))  
    
    parent_id   = pos_id
    total_offset = 2
    

    if (line[2] != '('): 
        # sub-tree is leaf
        # line[0:3] = ['(', 'pos', 'word', ')']        
        word_tag    = line[2]
        tree.create_node(word_tag, leaf_id, parent = parent_id, data = nRange(0,0))

        return (4, max_id,  leaf_id+1)
    
    line     = line[2:]
    
    while (line[0] != ')'):
        offset, max_id, leaf_id      = get_tree(tree, line, max_id, leaf_id, parent_id)
        total_offset       += offset
        line                = line[offset:]   
    
    return (total_offset + 1, max_id, leaf_id)


def get_trees(fin):

    for line in read_balanced_line(fin):
    
        tree = Tree()
        line = re.sub(r'\([\s]*-NONE-[^\)]+\)', '', line)
        p = re.compile('\([\S]+[\s]+\)')
        while p.findall(line):
            line = re.sub(p, '', line)

        max_id =  len([w for w in re.compile('[\S]+\)').findall(line) if not w.endswith('))')]) + 1
        line = line.replace('(' , ' ( ').replace(')', ' ) ').split()[1:]
        get_tree(tree, line, max_id)
        #return tree
        yield tree


class nRange(object):

    def __init__(self, minRange, maxRange):
        self.mRange = (minRange, maxRange)



def genRange(tree, nid, minR, maxR):

    if tree[nid].is_leaf():
        tree[nid].data.mRange = (minR, maxR)
        return (maxR, maxR + 1)


    for child in tree.children(nid):
        cid = child.identifier
        (minR, maxR) = genRange(tree, cid, minR, maxR)

    tree[nid].data.mRange = (min(map(lambda child: child.data.mRange[0], tree.children(nid))), \
                                max(map(lambda child: child.data.mRange[1], tree.children(nid))))

    return (minR, maxR)


def extend_path(tree, pid, leaf_id, path_dict):
 
    path_tag    = tree.parent(pid).tag
    if tree.siblings(pid): 
        siblings    = map(lambda sibling: sibling.tag ,  tree.siblings(pid))
        siblings.insert(0, path_tag)
        path_tag    = "\\".join(siblings)
    
    path_dict[leaf_id]  = "+".join([path_tag, path_dict[leaf_id]])


def gen_suptag(tree, nid, hieght_dict, path_dict):

    #Stop condition
    if tree[nid].is_leaf():
       
        pid             = tree.parent(nid).identifier
        path_dict[nid]  = tree[pid].tag

        if hieght_dict[nid] > 1 :
            extend_path(tree, pid, nid, path_dict)
                                
        return (nid, hieght_dict[nid]-1)
    
    #Recursion
    for child in tree.children(nid):
        cid = child.identifier
        (leaf_id, hieght) = gen_suptag(tree, cid, hieght_dict, path_dict)

    if (hieght == 1):
        return (None, 1) 

    elif (hieght > 1):
        pid = tree.parent(nid).identifier        
        extend_path(tree, pid, leaf_id, path_dict)
     
    return (leaf_id, hieght - 1)
      

def gen_height_list(tree, tree_dep_dict):

    hieght_dict = {}

    for leaf in  tree.leaves(tree.root):
        lid = leaf.identifier
        depid = tree_dep_dict[lid]
        if (depid == tree.root):
             hieght_dict[lid] =  tree.depth(leaf)

        else:
            minR = min(tree[depid].data.mRange[0], tree[lid].data.mRange[0])
            maxR = max(tree[depid].data.mRange[1], tree[lid].data.mRange[1])
            height = 0
            pid = tree.parent(lid).identifier
        
            while tree[pid].data.mRange[0] > minR or tree[pid].data.mRange[1] < maxR:
                height += 1             
                pid = tree.parent(pid).identifier

            hieght_dict[lid] = height

    return hieght_dict


def gen_tree_dep_dict(data_in):
    
    fin = "dep_treebank"
    os.system("java -jar code/pennconverter.jar < "+data_in+ "> "+fin)
    
    dep_dict_list = []
    dep_dict = {}
    lines = iter(open(fin, 'r'))
    for line in lines:
        words =  line.split()
        if words :
            dep_dict[int(words[0])] = int(words[6])
        else:
            dep_dict_list.append(dep_dict)
            dep_dict = {}

    os.remove(fin)

    return dep_dict_list

def gen_tags(fin):
    
    tree_dep_dict = gen_tree_dep_dict(fin)

    for i, tree in enumerate(get_trees(fin)):
        genRange(tree, tree.root, 0, 1)
        hieght_dict =  gen_height_list(tree, tree_dep_dict[i])
        path_dict    = {}
        gen_suptag(tree, tree.root, hieght_dict, path_dict)
        yield (' '.join(path_dict.values()), \
                        ' '.join(map(lambda key: tree[key].tag , path_dict.keys())))
        
        
def generate_data(src_dir, dest_dir):

    for sub_dir in os.listdir(src_dir):
        out_path = os.path.join(dest_dir, sub_dir)
        make_dir(os.path.join(out_path, "tags"))
        make_dir(os.path.join(out_path, "words"))
                    
        for f_in in os.listdir(os.path.join(src_dir, sub_dir)):
            data_in     =  os.path.join(src_dir, sub_dir, f_in)
            tags_out    =  os.path.join(out_path, "tags", f_in+'.tg')
            words_out    =  os.path.join(out_path, "words", f_in+'.wrd')

            with open(tags_out, 'w') as t_file:
                with open(words_out, 'w') as w_file:
                    for s_tags, s_words in gen_tags(data_in):
                        print(s_tags, file=t_file)
                        print(s_words, file=w_file)


if __name__ == '__main__':
    
    src_dir = 'raw_data/wsj/'
    dest_dir = 'proc_data'
    generate_data(src_dir, dest_dir)

