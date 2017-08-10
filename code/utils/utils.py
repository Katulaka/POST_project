
#TODO move to general functions file
def operate_on_Narray(A, function):
    if isinstance(A, list):
        return [operate_on_Narray(a,function) for a in A]
    return function(A)
