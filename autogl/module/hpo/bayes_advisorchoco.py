import hyperopt

# from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.chocolate_bayes import ChocolateBayesAlgorithm

# @register_hpo("randadvisor")


class BayesAdvisorChoco(AdvisorBaseHPOptimizer):
    def __init__(self, args):
        super().__init__(args)
        self.method = ChocolateBayesAlgorithm()

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        return cls(args)
