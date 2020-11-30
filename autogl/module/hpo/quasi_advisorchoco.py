import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.quasi_random_search import QuasiRandomSearchAlgorithm


@register_hpo("quasi")
class QuasiAdvisorChoco(AdvisorBaseHPOptimizer):
    """
    Quasi random search algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = QuasiRandomSearchAlgorithm()

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        return cls(args)
