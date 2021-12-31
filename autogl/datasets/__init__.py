from autogl import backend as _backend
from ._dataset_registry import (
    DatasetUniversalRegistry,
    build_dataset_from_name
)

from ._gtn_data import (
    GTNACMDataset,
    GTNDBLPDataset,
    GTNIMDBDataset,
)

from ._matlab_matrix import BlogCatalogDataset, WIKIPEDIADataset
from ._ogb import (
    OGBNProductsDataset, OGBNProteinsDataset, OGBNArxivDataset, OGBNPapers100MDataset,
    OGBLPPADataset, OGBLCOLLABDataset, OGBLDDIDataset, OGBLCitation2Dataset,
    OGBGMOLHIVDataset, OGBGMOLPCBADataset, OGBGPPADataset, OGBGCode2Dataset
)

if _backend.DependentBackend.is_dgl():
    from ._dgl import (
        CoraDataset,
        CiteSeerDataset,
        PubMedDataset,
        RedditDataset,
        AmazonComputersDataset,
        AmazonPhotoDataset,
        CoauthorPhysicsDataset,
        CoauthorCSDataset,
        MUTAGDataset,
        ENZYMESDataset,
        IMDBBinaryDataset,
        IMDBMultiDataset,
        RedditBinaryDataset,
        REDDITMulti5KDataset,
        COLLABDataset,
        ProteinsDataset,
        PTCMRDataset,
        NCI1Dataset
    )
    from ._heterogeneous_datasets import ACMHANDataset, ACMHGTDataset

elif _backend.DependentBackend.is_pyg():
    from ._pyg import (
        CoraDataset,
        CiteSeerDataset,
        PubMedDataset,
        FlickrDataset,
        RedditDataset,
        AmazonComputersDataset,
        AmazonPhotoDataset,
        CoauthorPhysicsDataset,
        CoauthorCSDataset,
        PPIDataset,
        QM9Dataset,
        MUTAGDataset,
        ENZYMESDataset,
        IMDBBinaryDataset,
        IMDBMultiDataset,
        RedditBinaryDataset,
        REDDITMulti5KDataset,
        REDDITMulti12KDataset,
        COLLABDataset,
        ProteinsDataset,
        PTCMRDataset,
        NCI1Dataset,
        NCI109Dataset,
        ModelNet10TrainingDataset,
        ModelNet10TestDataset,
        ModelNet40TrainingDataset,
        ModelNet40TestDataset
    )

if _backend.DependentBackend.is_pyg():
    __all__ = [
        "CoraDataset",
        "CiteSeerDataset",
        "PubMedDataset",
        "FlickrDataset",
        "RedditDataset",
        "AmazonComputersDataset",
        "AmazonPhotoDataset",
        "CoauthorPhysicsDataset",
        "CoauthorCSDataset",
        "PPIDataset",
        "QM9Dataset",
        "MUTAGDataset",
        "ENZYMESDataset",
        "IMDBBinaryDataset",
        "IMDBMultiDataset",
        "RedditBinaryDataset",
        "REDDITMulti5KDataset",
        "REDDITMulti12KDataset",
        "COLLABDataset",
        "ProteinsDataset",
        "PTCMRDataset",
        "NCI1Dataset",
        "NCI109Dataset",
        "ModelNet10TrainingDataset",
        "ModelNet10TestDataset",
        "ModelNet40TrainingDataset",
        "ModelNet40TestDataset",
        "OGBNProductsDataset",
        "OGBNProteinsDataset",
        "OGBNArxivDataset",
        "OGBNPapers100MDataset",
        "OGBLPPADataset",
        "OGBLCOLLABDataset",
        "OGBLDDIDataset",
        "OGBLCitation2Dataset",
        "OGBGMOLHIVDataset",
        "OGBGMOLPCBADataset",
        "OGBGPPADataset",
        "OGBGCode2Dataset",
        "GTNACMDataset",
        "GTNDBLPDataset",
        "GTNIMDBDataset",
        "BlogCatalogDataset",
        "WIKIPEDIADataset"
    ]
else:
    __all__ = [
        "CoraDataset",
        "CiteSeerDataset",
        "PubMedDataset",
        "RedditDataset",
        "AmazonComputersDataset",
        "AmazonPhotoDataset",
        "CoauthorPhysicsDataset",
        "CoauthorCSDataset",
        "MUTAGDataset",
        "ENZYMESDataset",
        "IMDBBinaryDataset",
        "IMDBMultiDataset",
        "RedditBinaryDataset",
        "REDDITMulti5KDataset",
        "COLLABDataset",
        "ProteinsDataset",
        "PTCMRDataset",
        "NCI1Dataset",
        "ACMHANDataset",
        "ACMHGTDataset",
        "OGBNProductsDataset",
        "OGBNProteinsDataset",
        "OGBNArxivDataset",
        "OGBNPapers100MDataset",
        "OGBLPPADataset",
        "OGBLCOLLABDataset",
        "OGBLDDIDataset",
        "OGBLCitation2Dataset",
        "OGBGMOLHIVDataset",
        "OGBGMOLPCBADataset",
        "OGBGPPADataset",
        "OGBGCode2Dataset",
        "GTNACMDataset",
        "GTNDBLPDataset",
        "GTNIMDBDataset",
        "BlogCatalogDataset",
        "WIKIPEDIADataset"
    ]
