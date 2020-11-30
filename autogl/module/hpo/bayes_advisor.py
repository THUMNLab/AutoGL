import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.bayesian_optimization import BayesianOptimization


@register_hpo("bayes")
class BayesAdvisor(AdvisorBaseHPOptimizer):
    """
    Bayes algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = BayesianOptimization()

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        return cls(args)
