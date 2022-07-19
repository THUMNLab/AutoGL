from .. import _data_preprocessor


class StructureEngineer(_data_preprocessor.DataPreprocessor):
    ...


from .._data_preprocessor_registry import DataPreprocessorUniversalRegistry
from deeprobust.graph.defense.gcn_preprocess import GCNJaccard as Jaccard
@DataPreprocessorUniversalRegistry.register_data_preprocessor("gcnjaccard")
class GCNJaccard(StructureEngineer):
    def __init__(self, threshold=0.01, *args, **kwargs):
        super(GCNJaccard, self).__init__(*args, **kwargs)
        self.engine=Jaccard(2,2,2)
        self.engine.threshold=threshold
    def _transform(self,data):
        features=data.x
        adj=data.edge_index
        modified_adj = self.engine.drop_dissimilar_edges(features, adj)
        data.edge_index=modified_adj
        return data