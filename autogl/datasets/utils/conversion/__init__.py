try:
    import dgl
except ModuleNotFoundError:
    dgl = None
else:
    from ._to_dgl_dataset import to_dgl_dataset
try:
    import torch_geometric
except ModuleNotFoundError:
    torch_geometric = None
else:
    from ._to_pyg_dataset import to_pyg_dataset
