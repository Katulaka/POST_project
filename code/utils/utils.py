from multiprocessing import Process, Queue
from itertools import chain
import numpy as np
# import cPickle
import _pickle as cPickle

def _operate_on_Narray(A, function, *kwarg): #recursive up to the element level
    if isinstance(A, list):
        return [_operate_on_Narray(a, function, *kwarg) for a in A]
    return function(A, *kwarg)

def _getattr_operate_on_Narray(A, function, *kwarg): #recursive up to the element level
    if isinstance(A, list):
        return [_getattr_operate_on_Narray(a, function, *kwarg) for a in A]
    return getattr(A, function)(*kwarg)

def operate_on_Narray(A, function, *kwarg): #recursive up to the list level
    if isinstance(A, list) and all(not isinstance(el, list) for el in A):
        return function(A, *kwarg)
    return [operate_on_Narray(a, function, *kwarg) for a in A]

def fast_copy(src):
    return cPickle.loads(cPickle.dumps(src))

def repeated(f, n):
    def repeat(arg):
        return reduce(lambda r, g: g(r), [f] * n, arg)
    return repeat

def flattenNd(A, n):
    return repeated(lambda x: list(chain.from_iterable(x)), n)(A)

def arr_dim(arr):
    return 1 + arr_dim(arr[0]) if (type(arr) == list) else 0

def flatten_to_1D(A):
    n =  arr_dim(A) - 1
    return flattenNd(A, n)

def _select(A,i):
    return list(np.array(A)[i])

# def profile(fn, *args): #TODO
#     import cProfile, pstats, StringIO
#     pr = cProfile.Profile()
#     pr.enable()
#     fn(*args)
#     pr.disable()
#     s = StringIO.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print s.getvalue()

class ProcessWithReturnValue(Process):

    from multiprocessing import Process, Queue

    def __init__(self, group=None, target=None, name=None, res_q = None, args=()):
        Process.__init__(self, group, target, name, args)
        self._name   = name
        self._res_q = res_q
        self._args   = args
        self._target = target
        self._return = None

    def run(self):
        if self._target is not None:
            self._res_q.put(self._target(*self._args))


    def join(self):
        Process.join(self)
        self._return = self._res_q.get()
        return self._name, self._return
