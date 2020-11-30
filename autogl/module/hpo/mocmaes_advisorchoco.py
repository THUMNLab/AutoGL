import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.mocmaes import MocmaesAlgorithm


@register_hpo("mocmaes")
class MocmaesAdvisorChoco(AdvisorBaseHPOptimizer):
    """
    MOCMAES algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, args):
        super().__init__(args)
        self.method = MocmaesAlgorithm()

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        return cls(args)
