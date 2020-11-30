from ...suggestion.algorithm.base_hyperopt_algorithm import BaseHyperoptAlgorithm


class HyperoptRandomSearchAlgorithm(BaseHyperoptAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with random search algorithm.
    """

    def __init__(self):
        super(HyperoptRandomSearchAlgorithm, self).__init__("random_search")
