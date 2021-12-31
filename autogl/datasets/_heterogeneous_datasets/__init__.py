from autogl import backend as _backend

if _backend.DependentBackend.is_dgl():
    from ._dgl_heterogeneous_datasets import (
        ACMHANDataset, ACMHGTDataset
    )
