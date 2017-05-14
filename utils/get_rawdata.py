from __future__ import print_function
import os
import shutil


def make_dir(path_dir):
    
    if not os.path.exists(path_dir):
        try:
            os.makedirs(os.path.abspath(path_dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

#TODO: hidden files
def del_extra_files(root_dir):

    print("[del_extra_files] Deleting mismatching files between dirs")
    
    for f in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, f)) and f not in ["{:02d}".format(x) for x in xrange(25)]:
            shutil.rmtree(os.path.join(root_dir, f))
        if os.path.isfile(os.path.join(root_dir, f)):
            os.remove(os.path.join(root_dir, f))

def get_data(data_path):

    print("[get_data] Getting RAW data from corpora")

    make_dir(data_path)
    
    if not os.path.exists(os.path.join(data_path, 'wsj')):
        os.system("scp -r login.eecs.berkeley.edu:/project/eecs/nlp/corpora/'EnglishTreebank/wsj/ " + data_path)
        del_extra_files(os.path.join(data_path, 'wsj'))

    
if __name__ == '__main__':
    





    
