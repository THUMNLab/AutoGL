.. _hpo:

Hyper Parameter Optimization
============================

We support black box hyper parameter optimization in variant search space.

Search Space
------------

Three types of search space are supported, use ``dict`` in python to define your search space.
For numerical list search space. You can either assign a fixed length for the list, if so, you need not provide ``cutPara`` and ``cutFunc``.
Or you can let HPO cut the list to a certain length which is dependent on other parameters. You should provide those parameters' names in ``curPara`` and the function to calculate the cut length in "cutFunc". 

.. code-block:: python

    # numerical search space:
    {
        "parameterName": "xxx",
        "type": "DOUBLE" / "INTEGER",
        "minValue": xx,
        "maxValue": xx,
        "scalingType": "LINEAR" / "LOG"
    }

    # numerical list search space:
    {
        "parameterName": "xxx",
        "type": "NUMERICAL_LIST",
        "numericalType": "DOUBLE" / "INTEGER",
        "length": 3,
        "cutPara": ("para_a", "para_b"),
        "cutFunc": lambda x: x[0] - 1,
        "minValue": [xx,xx,xx],
        "maxValue": [xx,xx,xx],
        "scalingType": "LINEAR" / "LOG"
    }

    # categorical search space:
    {
        "parameterName": xxx,
        "type": "CATEGORICAL"
        "feasiblePoints": [a,b,c]
    }

    # fixed parameter as search space:
    {
        "parameterName": xxx,
        "type": "FIXED",
        "value": xxx
    }
        
How given HPO algorithms support search space is listed as follows:

+------------------+------------+--------------+-----------+------------+
| Algorithm        | numerical  |numerical list|categorical| fixed      |
+==================+============+==============+===========+============+
| Grid             |            |              |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| Random           | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| Anneal           | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| Bayes            | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| TPE [1]_         | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| CMAES [2]_       | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| MOCMAES [3]_     | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
|Quasi random [4]_ | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+
| AutoNE  [5]_     | ✓          |  ✓           |  ✓        | ✓          |
+------------------+------------+--------------+-----------+------------+

Add Your HPOptimizer
--------------------

If you want to add your own HPOptimizer, the only thing you should do is finishing ``optimize`` function in you HPOptimizer:

.. code-block:: python

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


.. [1] Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in neural information processing systems. 2011.
.. [2] Arnold, Dirk V., and Nikolaus Hansen. "Active covariance matrix adaptation for the (1+ 1)-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [3] Voß, Thomas, Nikolaus Hansen, and Christian Igel. "Improved step size adaptation for the MO-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [4] Bratley, Paul, Bennett L. Fox, and Harald Niederreiter. "Programs to generate Niederreiter's low-discrepancy sequences." ACM Transactions on Mathematical Software (TOMS) 20.4 (1994): 494-495.
.. [5] Tu, Ke, et al. "Autone: Hyperparameter optimization for massive network embedding." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.
