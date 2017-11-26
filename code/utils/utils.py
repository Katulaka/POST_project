from itertools import chain
from multiprocessing import Process, Queue

def _operate_on_Narray(A, function):
    if isinstance(A, list):
        return [_operate_on_Narray(a,function) for a in A]
    return function(A)

def operate_on_Narray(A, function, *kwarg):
    if isinstance(A, list) and all(not isinstance(el, list) for el in A):
        return function(A, *kwarg)
    return [operate_on_Narray(a,function,*kwarg) for a in A]


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

class ProcessWithReturnValue(Process):

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
