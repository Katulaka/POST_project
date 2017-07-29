from __future__ import print_function
import os

def make_dir(path_dir):
    if not os.path.exists(path_dir):
        try:
            os.makedirs(os.path.abspath(path_dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def read_file(fname):
    with open(fname) as f:
        text = f.read().splitlines()
    return list(map(lambda x: x.split(), text))

def flatten(_list):
     for it in _list:
        for element in it:
            yield element

def is_balanced(string):
    iparens = iter('(){}[]<>')
    parens = dict(zip(iparens, iparens))
    closing = parens.values()
    stack = []
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
            if is_balanced(s) and s != "":
                yield s
                s = ""
