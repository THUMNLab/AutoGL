import hyperopt

from torch_geometric.nn import GCNConv, SAGEConv
from . import register_hpo
from .nas import BaseEstimator, GraphSpace
from .darts import DartsTrainer
from .base import BaseHPOptimizer, TimeTooLimitedError


@register_hpo("test")
class TestHPO(BaseHPOptimizer):
    """
    Random search algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        num_features = dataset[0].x.shape[1]
        num_classes = dataset.num_classes

        op1 = lambda x, y: GCNConv(x, y)
        op2 = lambda x, y: SAGEConv(x, y)
        ops = [op1, op2]
        model = GraphSpace(num_features, 64, num_classes, ops)
        tr = BaseEstimator()
        nas = DartsTrainer()
        a = nas.search(model, dataset, tr)
        print(a)
        print(type(a))
        return 1, 2

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        return cls(args)
