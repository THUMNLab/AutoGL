from ...suggestion.algorithm.base_skopt_algorithm import BaseSkoptAlgorithm


class SkoptBayesianOptimization(BaseSkoptAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with bayesian optimization algorithm.
    """

    def __init__(self):
        super(SkoptBayesianOptimization, self).__init__("bayesian_optimization")
