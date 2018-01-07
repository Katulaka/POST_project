from utils.utils import operate_on_Narray, _operate_on_Narray

R = '}'
L = '{'
CR = '>'
CL = '<'
UP = '+'
NA = '|'
ANY = '*'

class TagOp(object):

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def _mod_tag(self, tag, l_sym, r_sym):
        return tag.replace(L, l_sym+L+r_sym).replace(R, l_sym+R+r_sym)

    def _tag_split(self, tag):
        return self._mod_tag(tag, UP, '').split(UP)

    def _slash_split(self, tag):
        return self._mod_tag(tag, UP, UP).split(UP)

    def _revese(self, tag):
        return UP.join(tag.split(UP)[::-1])

    def _remove_val_gap(self, tag):
        return self._mod_tag(tag, '', ANY+NA).split(NA)[0]

    def modify_tag(self, tag):
        if not self.direction: #if don't want to keep left/right indication.
            tag = tag.replace(R, L)   # default: all left

        if self.no_val_gap:
            tag = tag.replace(L, L+ANY+NA).replace(R, R+ANY+NA).split(NA)[0]

        return tag

    def _modify_fn(self, tag_list):
        return [self.modify_tag(tag) for tag in tag_list]

    def modify_fn(self, tags):
        return _operate_on_Narray(tags, self.modify_tag)

    def combine_tag(self, tag):
        res = []
        _tag = tag
        try:
            if tag[0] in [CL, CR]:
                res.append(tag[:2])
                _tag = tag[2:]
        except:
            return res
        for t in _tag:
            if res and (res[-1][-1].endswith(L) or res[-1][-1].endswith(R)):
                res[-1] += [t]
            elif res and (t.startswith(L) or t.startswith(R)):
                res[-1] += [t]
            else:
                res.append([t])
        return res

    def combine_fn(self, tags):
        return operate_on_Narray(tags, self.combine_tag)

    def add_right(self, tag):
        return R + tag

    def add_left(self, tag):
        return L + tag
