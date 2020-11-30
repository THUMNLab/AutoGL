"""
ensemble module
"""


class BaseEnsembler:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, predictions, label, identifiers, feval, *args, **kwargs):
        pass

    def ensemble(self, predictions, identifiers, *args, **kwargs):
        pass

    @classmethod
    def build_ensembler_from_args(cls, args):
        """Build a new ensembler instance."""
        raise NotImplementedError(
            "Ensembler must implement the build_ensembler_from_args method"
        )
