import autogl

if autogl.backend.DependentBackend.is_dgl():
    from ._data_preprocessor_dgl import DataPreprocessor
else:
    from ._data_preprocessor import DataPreprocessor
