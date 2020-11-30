from ...suggestion.algorithm.base_chocolate_algorithm import BaseChocolateAlgorithm


class MocmaesAlgorithm(BaseChocolateAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with MOCMAES algorithm.
    """

    def __init__(self):
        super(MocmaesAlgorithm, self).__init__("MOCMAES")
