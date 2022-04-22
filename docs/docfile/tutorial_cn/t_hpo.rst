.. _hpo_cn:

超参数优化
============================

我们支持不同的搜索空间下的黑盒超参数优化。

搜索空间
------------

我们支持三种类型的搜索空间，请使用python字典``dict``来定义你的搜索空间。
对于数字列表形式的搜索空间，你可以为列表指定一个固定的长度，如果是这样，你不需要提供``cutPara``和``cutFunc``。
或者你可以把列表切成一个特定的长度，这个长度取决于参数``cutPara``和``cutFunc``。你应该在``cutPara``中提供参数名并在``cutFunc``中提供计算剪切长度的函数。

.. code-block:: python

    # 数值搜索空间：
    {
        "parameterName": "xxx",
        "type": "DOUBLE" / "INTEGER",
        "minValue": xx,
        "maxValue": xx,
        "scalingType": "LINEAR" / "LOG"
    }

    # 数值列表搜索空间：
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

    # 类别搜索空间：
    {
        "parameterName": xxx,
        "type": "CATEGORICAL"
        "feasiblePoints": [a,b,c]
    }

    # 固定参数作为搜索空间：
    {
        "parameterName": xxx,
        "type": "FIXED",
        "value": xxx
    }

下表列出了超参数优化算法所支持的搜索空间形式： 

+------------------+------------+--------------+-----------+------------+
| 算法              | 数值        | 数值列表      | 类别       | 固定的      |
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

添加你自己的超参数优化器（HPOptimizer）
--------------------

如果你想添加你自己的 HPOptimizer, 你只需要完成 HPOptimizer 中的``optimize`` 函数:

.. code-block:: python

    # 例如，创建一个随机超参数优化算法
    import random
    from autogl.module.hpo.base import BaseHPOptimizer
    class RandomOptimizer(BaseHPOptimizer):
        # 在初始化时获取基本参数
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_evals = kwargs.get("max_evals", 2)

        # 你应该做的最重要的事情是完成优化函数
        def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
            # 1. 从训练器得到搜索空间。
            space = trainer.hyper_parameter_space + trainer.model.hyper_parameter_space
            # 可选的：使用 self._encode_para (在 BaseOptimizer) 来对搜索空间进行预处理
            # 如果使用 _encode_para, NUMERICAL_LIST 将扩展为 双精度浮点数 或 整数，LOG尺度类型将改为线性，CATEGORICAL中的可行点将改为离散数。
            # 您还应该使用_decode_para将参数类型转换回来。
            current_space = self._encode_para(space)

            # 2. 定义你自己的性能函数。
            def fn(dset, para):
                current_trainer = trainer.duplicate_from_hyper_parameter(para)
                current_trainer.train(dset)
                loss, self.is_higher_better = current_trainer.get_valid_score(dset)
                # 为了方便起见，损失分数越高越好；如果是负数，那么我们就应该把损失分数降到最低。
                if self.is_higher_better:
                    loss = -loss
                return current_trainer, loss

            # 3. 定义如何获得超参数建议，它应该返回一个参数字典。你可以使用历史实验来得到新的建议。
            def get_random(history_trials):
                hps = {}
                for para in current_space:
                    # 因为我们之前使用了_encode_para函数，所以我们应该只处理DOUBLE、INTEGER和DISCRETE函数
                    if para["type"] == "DOUBLE" or para["type"] == "INTEGER":
                        hp = random.random() * (para["maxValue"] - para["minValue"]) + para["minValue"]
                        if para["type"] == "INTEGER":
                            hp = round(hp)
                        hps[para["parameterName"]] = hp
                    elif para["type"] == "DISCRETE":
                        feasible_points = para["feasiblePoints"].split(",")
                        hps[para["parameterName"]] = random.choice(feasible_points)
                return hps

            # 4. 运行算法。对于每个回合，根据历史信息获得一组参数并进行评估。
            best_trainer, best_para, best_perf = None, None, None
            self.trials = []
            for i in range(self.max_evals):
                # 在这个例子中，我们不需要历史追踪。因此我们为history_trails传入None
                new_hp = get_random(None)
                # 可选的：如果你使用参数 _encode_para，也要提供参数 _decode_para 。 para_for_trainer 撤销 _encode_para 中的所有转换，并在需要时将双精度浮点数转换为整数。para_for_hpo 只将双精度浮点数转换为整数。
                para_for_trainer, para_for_hpo = self._decode_para(new_hp)
                current_trainer, perf = fn(dataset, para_for_trainer)
                self.trials.append((para_for_hpo, perf))
                if not best_perf or perf < best_perf:
                    best_perf = perf
                    best_trainer = current_trainer
                    best_para = para_for_trainer

            # 5. 返回最优训练器和参数。
            return best_trainer, best_para


.. [1] Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in neural information processing systems. 2011.
.. [2] Arnold, Dirk V., and Nikolaus Hansen. "Active covariance matrix adaptation for the (1+ 1)-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [3] Voß, Thomas, Nikolaus Hansen, and Christian Igel. "Improved step size adaptation for the MO-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [4] Bratley, Paul, Bennett L. Fox, and Harald Niederreiter. "Programs to generate Niederreiter's low-discrepancy sequences." ACM Transactions on Mathematical Software (TOMS) 20.4 (1994): 494-495.
.. [5] Tu, Ke, et al. "Autone: Hyperparameter optimization for massive network embedding." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.
