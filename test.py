# For example, create a random HPO by yourself
import random
from autogl.module.hpo.base import BaseHPOptimizer
class RandomOptimizer(BaseHPOptimizer):
    # Get essential parameters at initialization
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get("max_evals", 2)

    # The most important thing you should do is completing optimization function
    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        # 1. Get the search space from trainer.
        space = trainer.hyper_parameter_space + trainer.model.hyper_parameter_space
        # optional: use self._encode_para (in BaseOptimizer) to pretreat the space
        # If you use _encode_para, the NUMERICAL_LIST will be spread to DOUBLE or INTEGER, LOG scaling type will be changed to LINEAR, feasible points in CATEGORICAL will be changed to discrete numbers.
        # You should also use _decode_para to transform the types of parameters back.
        current_space = self._encode_para(space)

        # 2. Define your function to get the performance.
        def fn(dset, para):
            current_trainer = trainer.duplicate_from_hyper_parameter(para)
            current_trainer.train(dset)
            loss, self.is_higher_better = current_trainer.get_valid_score(dset)
            # For convenience, we change the score which is higher better to negative, then we should only minimize the score.
            if self.is_higher_better:
                loss = -loss
            return current_trainer, loss

        # 3. Define the how to get HP suggestions, it should return a parameter dict. You can use history trials to give new suggestions
        def get_random(history_trials):
            hps = {}
            for para in current_space:
                # Because we use _encode_para function before, we should only deal with DOUBLE, INTEGER and DISCRETE
                if para["type"] == "DOUBLE" or para["type"] == "INTEGER":
                    hp = random.random() * (para["maxValue"] - para["minValue"]) + para["minValue"]
                    if para["type"] == "INTEGER":
                        hp = round(hp)
                    hps[para["parameterName"]] = hp
                elif para["type"] == "DISCRETE":
                    feasible_points = para["feasiblePoints"].split(",")
                    hps[para["parameterName"]] = random.choice(feasible_points)
            return hps

        # 4. Run your algorithm. For each turn, get a set of parameters according to history information and evaluate it.
        best_trainer, best_para, best_perf = None, None, None
        self.trials = []
        for i in range(self.max_evals):
            # in this example, we don't need history trails. Since we pass None to history_trails
            new_hp = get_random(None)
            # optional: if you use _encode_para, use _decode_para as well. para_for_trainer undos all transformation in _encode_para, and turns double parameter to interger if needed. para_for_hpo only turns double parameter to interger.
            para_for_trainer, para_for_hpo = self._decode_para(new_hp)
            current_trainer, perf = fn(dataset, para_for_trainer)
            self.trials.append((para_for_hpo, perf))
            if not best_perf or perf < best_perf:
                best_perf = perf
                best_trainer = current_trainer
                best_para = para_for_trainer

        # 5. Return the best trainer and parameter.
        return best_trainer, best_para
