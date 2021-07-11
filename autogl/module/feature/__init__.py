from .base import BaseFeature
from .base import BaseFeatureEngineer

FEATURE_DICT = {}


def register_feature(name):
    def register_feature_cls(cls):
        if name in FEATURE_DICT:
            raise ValueError(
                "Cannot register duplicate feature engineer ({})".format(name)
            )
        # if not issubclass(cls, BaseFeatureEngineer):
        if not issubclass(cls, BaseFeature):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseFeatureEngineer".format(
                    name, cls.__name__
                )
            )
        FEATURE_DICT[name] = cls
        return cls

    return register_feature_cls


from .auto_feature import AutoFeatureEngineer

from .generators import (
    BaseGenerator,
    GeGraphlet,
    GeEigen,
    GePageRank,
    register_pyg,
    pygfunc,
    PYGGenerator,
    PYGLocalDegreeProfile,
    PYGNormalizeFeatures,
    PYGOneHotDegree,
)

from .selectors import BaseSelector, SeFilterConstant, SeGBDT

from .graph import (
    BaseGraph,
    SgNetLSD,
    register_nx,
    NxGraph,
    nxfunc,
    NxLargeCliqueSize,
    NxAverageClusteringApproximate,
    NxDegreeAssortativityCoefficient,
    NxDegreePearsonCorrelationCoefficient,
    NxHasBridge,
    NxGraphCliqueNumber,
    NxGraphNumberOfCliques,
    NxTransitivity,
    NxAverageClustering,
    NxIsConnected,
    NxNumberConnectedComponents,
    NxIsDistanceRegular,
    NxLocalEfficiency,
    NxGlobalEfficiency,
    NxIsEulerian,
)

__all__ = [
    "BaseFeatureEngineer",
    "AutoFeatureEngineer",
    "BaseFeature",
    "BaseGenerator",
    "GeGraphlet",
    "GeEigen",
    "GePageRank",
    "register_pyg",
    "pygfunc",
    "PYGGenerator",
    "PYGLocalDegreeProfile",
    "PYGNormalizeFeatures",
    "PYGOneHotDegree",
    "BaseSelector",
    "SeFilterConstant",
    "SeGBDT",
    "BaseGraph",
    "SgNetLSD",
    "register_nx",
    "NxGraph",
    "nxfunc",
    "NxLargeCliqueSize",
    "NxAverageClusteringApproximate",
    "NxDegreeAssortativityCoefficient",
    "NxDegreePearsonCorrelationCoefficient",
    "NxHasBridge",
    "NxGraphCliqueNumber",
    "NxGraphNumberOfCliques",
    "NxTransitivity",
    "NxAverageClustering",
    "NxIsConnected",
    "NxNumberConnectedComponents",
    "NxIsDistanceRegular",
    "NxLocalEfficiency",
    "NxGlobalEfficiency",
    "NxIsEulerian",
]
