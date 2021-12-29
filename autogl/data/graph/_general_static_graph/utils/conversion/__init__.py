from ._general import StaticGraphToGeneralData, static_graph_to_general_data
from ._nx import (
    HomogeneousStaticGraphToNetworkX
)

try:
    import dgl
except ModuleNotFoundError:
    dgl = None
else:
    from ._dgl import (
        DGLGraphToGeneralStaticGraph, dgl_graph_to_general_static_graph,
        GeneralStaticGraphToDGLGraph, general_static_graph_to_dgl_graph,
    )
try:
    import torch_geometric
except ModuleNotFoundError:
    torch_geometric = None
else:
    from ._pyg import StaticGraphToPyGData, static_graph_to_pyg_data
