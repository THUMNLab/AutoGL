# tackle different backend
from autogl.backend import DependentBackend
_isdgl=DependentBackend.is_dgl()

def is_dgl():
    return _isdgl

if is_dgl():
    def bk_mask(data,mask):
        return data.ndata[f'{mask}_mask']
    def bk_label(data):
        return data.ndata['label']
    def bk_feat(data):
        return data.ndata['feat']
    def bk_gconv(op,data,feat):
        return op(data,feat)
else:
    def bk_mask(data,mask):
        return data[f'{mask}_mask']
    def bk_label(data):
        return data.y
    def bk_feat(data):
        return data.x
    def bk_gconv(op,data,feat):
        return op(feat,data.edge_index)
