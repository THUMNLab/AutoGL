from ...suggestion.algorithm.base_chocolate_algorithm import BaseChocolateAlgorithm


class ChocolateRandomSearchAlgorithm(BaseChocolateAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with random search algorithm.
    """

    def __init__(self):
        super(ChocolateRandomSearchAlgorithm, self).__init__("Random")
