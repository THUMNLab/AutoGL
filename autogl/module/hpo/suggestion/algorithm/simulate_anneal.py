from ...suggestion.algorithm.base_hyperopt_algorithm import BaseHyperoptAlgorithm


class SimulateAnnealAlgorithm(BaseHyperoptAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with simulate anneal algorithm.
    """

    def __init__(self):
        super(SimulateAnnealAlgorithm, self).__init__("anneal")
