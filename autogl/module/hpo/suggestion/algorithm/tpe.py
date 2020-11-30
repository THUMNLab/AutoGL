from ...suggestion.algorithm.base_hyperopt_algorithm import BaseHyperoptAlgorithm


class TpeAlgorithm(BaseHyperoptAlgorithm):
    """
    The implementation is based on https://github.com/tobegit3hub/advisor
    Get the new suggested trials with TPE algorithm.
    """

    def __init__(self):
        super(TpeAlgorithm, self).__init__("tpe")
