try:
    import dgl
except ModuleNotFoundError:
    dgl = None
else:
    from ._to_dgl_dataset import general_static_graphs_to_dgl_dataset
try:
    import torch_geometric
except ModuleNotFoundError:
    torch_geometric = None
else:
    from ._to_pyg_dataset import general_static_graphs_to_pyg_dataset
