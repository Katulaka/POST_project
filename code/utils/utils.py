from itertools import chain

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
