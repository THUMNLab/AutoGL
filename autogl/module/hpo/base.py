"""
HPO Module for tuning hyper parameters
"""

from ...utils import get_logger
import hyperopt
import time
import math
import dill
import multiprocessing as mp

mp.set_start_method("spawn", True)


LOGGER = get_logger("HPO")


class BaseHPOptimizer:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.logger = LOGGER

    def _decompose_depend_list_para(self, config):
        self._depend_map = {}

        def get_depended_para(name):
            for p in config:
                if p["parameterName"] == name:
                    return
            raise WrongDependedParameterError("The depended parameter does not exist.")

        for para in config:
            if para["type"] in ("NUMERICAL_LIST", "CATEGORICAL_LIST") and para.get(
                "cutPara", None
            ):
                self._depend_map[para["parameterName"]] = para
                if type(para["cutPara"]) == str:
                    get_depended_para(para["cutPara"])
                else:
                    for dpara in para["cutPara"]:
                        get_depended_para(dpara)

        return config

    def _compose_depend_list_para(self, config):
        for para in self._depend_map:
            cutparas = self._depend_map[para]["cutPara"]
            if type(cutparas) == str:
                dparas = [config[cutparas]]
                # dparas = config[cutparas]
            else:
                dparas = []
                for dpara in cutparas:
                    dparas.append(config[dpara])
            paralen = self._depend_map[para]["cutFunc"](dparas)
            config[para] = config[para][:paralen]
        return config

    def _decompose_list_fixed_para(self, config):
        config = self._decompose_depend_list_para(config)
        fin = []
        self._list_map = {}
        self._fix_map = {}
        for para in config:
            if para["type"] == "NUMERICAL_LIST":
                self._list_map[para["parameterName"]] = para["length"]
                if type(para["minValue"]) != list:
                    para["minValue"] = [para["minValue"] for i in range(para["length"])]
                if type(para["maxValue"]) != list:
                    para["maxValue"] = [para["maxValue"] for i in range(para["length"])]
                for i, x, y in zip(
                    range(para["length"]), para["minValue"], para["maxValue"]
                ):
                    new_para = {}
                    new_para["parameterName"] = para["parameterName"] + "_" + str(i)
                    new_para["type"] = para["numericalType"]
                    new_para["minValue"] = x
                    new_para["maxValue"] = y
                    new_para["scalingType"] = para["scalingType"]
                    fin.append(new_para)
            elif para["type"] == "CATEGORICAL_LIST":
                self._list_map[para["parameterName"]] = para["length"]
                category = para["feasiblePoints"]
                self._category_map[para["parameterName"]] = category

                cur_points = ",".join(map(lambda _x: str(_x), range(len(category))))
                for i in range(para["length"]):
                    new_para = dict()
                    new_para["parameterName"] = para["parameterName"] + "_" + str(i)
                    new_para["type"] = "DISCRETE"
                    new_para["feasiblePoints"] = cur_points
                    fin.append(new_para)
            elif para["type"] == "FIXED":
                self._fix_map[para["parameterName"]] = para["value"]
            else:
                new_para = para.copy()
                new_para["parameterName"] = para["parameterName"] + "_"
                fin.append(new_para)
        return fin

    def _compose_list_fixed_para(self, config):
        fin = {}
        # compose list
        for pname in self._list_map:
            val = []
            for i in range(self._list_map[pname]):
                val.append(config[pname + "_" + str(i)])
                del config[pname + "_" + str(i)]
            if pname in self._category_map:
                val = [self._category_map[pname][i] for i in val]
            fin[pname] = val
        # deal other para
        for pname in config:
            fin[pname[:-1]] = config[pname]
        for pname in self._fix_map:
            fin[pname] = self._fix_map[pname]
        fin = self._compose_depend_list_para(fin)
        return fin

    def _encode_para(self, config):
        """
        Convert all types of para space to DOUBLE(linear), DISCRETE
        config: [{
            "parameterName": "num_layers",
            "type": "DISCRETE",
            "feasiblePoints": "1,2,3,4",
        },{
            "parameterName": "hidden",
            "type": "NUMERICAL_LIST",
            "numericalType": "INTEGER",
            "length": 4,
            "minValue": [4, 4, 4, 4],
            "maxValue": [32, 32, 32, 32],
            "scalingType": "LOG"
        },{
            "parameterName": "dropout",
            "type": "DOUBLE",
            "minValue": 0.1,
            "maxValue": 0.9,
            "scalingType": "LINEAR"
        }]"""
        self._category_map = {}
        self._discrete_map = {}
        self._numerical_map = {}
        config = self._decompose_list_fixed_para(config)

        current_config = []
        for para in config:
            if para["type"] == "DOUBLE" or para["type"] == "INTEGER":
                cur_para = para.copy()
                cur_para["type"] = "DOUBLE"
                if para["scalingType"] == "LOG":
                    cur_para["minValue"] = math.log(para["minValue"])
                    cur_para["maxValue"] = math.log(para["maxValue"])
                current_config.append(cur_para)
                self._numerical_map[para["parameterName"]] = para
            elif para["type"] == "CATEGORICAL" or para["type"] == "DISCRETE":
                if para["type"] == "DISCRETE":
                    cate_list = para["feasiblePoints"].split(",")
                    cate_list = list(map(lambda x: x.strip(), cate_list))
                else:
                    cate_list = para["feasiblePoints"]
                cur_points = ",".join(map(lambda x: str(x), range(len(cate_list))))
                cur_para = para.copy()
                cur_para["feasiblePoints"] = cur_points
                cur_para["type"] = "DISCRETE"
                current_config.append(cur_para)
                if para["type"] == "CATEGORICAL":
                    self._category_map[para["parameterName"]] = cate_list
                else:
                    self._discrete_map[para["parameterName"]] = cate_list
            else:
                current_config.append(para)
        return current_config

    def _decode_para(self, para):
        """
        decode HPO given para to user(externel) para and trial para
        """
        externel_para = para.copy()
        trial_para = para.copy()
        for name in para:
            if name in self._numerical_map:
                old_para = self._numerical_map[name]
                val = para[name]
                if old_para["scalingType"] == "LOG":
                    val = math.exp(val)
                    if val < old_para["minValue"]:
                        val = old_para["minValue"]
                    elif val > old_para["maxValue"]:
                        val = old_para["maxValue"]
                if old_para["type"] == "INTEGER":
                    val = int(round(val))
                externel_para[name] = val
                trial_para[name] = (
                    val if old_para["scalingType"] != "LOG" else math.log(val)
                )
            elif name in self._category_map:
                externel_para[name] = self._category_map[name][int(para[name])]
                trial_para[name] = para[name]
            elif name in self._discrete_map:
                externel_para[name] = eval(self._discrete_map[name][int(para[name])])
                trial_para[name] = para[name]
        externel_para = self._compose_list_fixed_para(externel_para)
        return externel_para, trial_para

    def _print_info(self, para, perf):
        if self.is_higher_better:
            # print("Parameter: {} {}: {} {}".format(para, trainer.get_feval(return_major=True).get_eval_name(), -perf, "higher_better"))
            LOGGER.info(
                "Parameter: {} {}: {} {}".format(
                    para, self.feval_name, -perf, "higher_better"
                )
            )
        else:
            # print("Parameter: {} {}: {} {}".format(para, trainer.get_feval(return_major=True).get_eval_name(), perf, "lower_better"))
            LOGGER.info(
                "Parameter: {} {}: {} {}".format(
                    para, self.feval_name, perf, "lower_better"
                )
            )

    def slave(self, pipe, trainer, dataset):
        while 1:
            x = pipe.recv()
            current_trainer = trainer.duplicate_from_hyper_parameter(x)
            current_trainer.train(dataset)
            loss, is_higher_better = current_trainer.get_valid_score(dataset)
            trainer_pic = dill.dumps(current_trainer)
            pipe.send(trainer_pic)
            pipe.send(loss)

    def _set_up(self, trainer, dataset, time_limit, memory_limit):
        """
        Initialize something used in "optimize"

        Parameters
        ----------
        trainer : ..train.BaseTrainer
            Including model, giving HP space and using for training
        dataset : ...datasets
            Dataset to train and evaluate.
        time_limit : int
            Max time to run HPO
        memory_limit : None
            No implementation yet
        """
        pass

    def _update_trials(self, pid, cur_trainer, perf, is_higher_better):
        """
        After the evaluation phase of each turn, update history trials according to the performance

        Parameters
        ----------
        pid : int
            The process id which runs the evaluation
        cur_trainer : ..train.BaseTrainer
            The trainer used in evaluation, containing the HPs
        perf : float
            The performance of the HP
        is_higher_better : bool
            True if higher perf is better
        """
        pass

    def _get_suggestion(self, pid):
        """
        Give the next HP suggestion

        Parameters
        ----------
        pid : int
            The process id which will run the evaluation

        Returns
        -------
        para_json: dict
            The suggested HP
        """
        pass

    def _best_hp(self):
        """
        Give the best HP and the best trainer as the returns of "optimize"

        Returns
        -------
        trainer: ..train.BaseTrainer
            The trainer including the best trained model
        para_json: dict
            The best HP
        """
        pass

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """
        The main process of optimizing the HP by the method within give model and HP space.
        This process provide an HPO scheme, where giving suggestion and evaluation are taken turns, and can be run on multi-GPU.
        If you want to implement your HPO algorithm, you can either follow this scheme, you should only change "_set_up", "_update_trials", "_get_suggestion", and "_best_hp".
        Or you can rewrite the whole "optimize" function.
        Parameters
        ----------
        trainer : ..train.BaseTrainer
            Including model, giving HP space and using for training
        dataset : ...datasets
            Dataset to train and evaluate.
        time_limit : int
            Max time to run HPO
        memory_limit : None
            No implementation yet

        Returns
        -------
        trainer: ..train.BaseTrainer
            The trainer including the best trained model
        para_json: dict
            The best HP
        """
        start_time = time.time()
        cur_evals = 0

        slaves = 2
        pipes = []
        procs = []
        statuses = []
        for i in range(slaves):
            ps, pr = mp.Pipe()
            pipes.append(ps)
            proc = mp.Process(
                target=self.slave,
                args=(
                    pr,
                    trainer,
                    dataset,
                ),
            )
            proc.start()
            procs.append(proc)
            statuses.append("ready")

        # do something of the certain HPO algo
        self._set_up(trainer, dataset, slaves, time_limit, memory_limit)
        self.feval_name = trainer.get_feval(return_major=True).get_eval_name()
        self.is_higher_better = trainer.get_feval(return_major=True).is_higher_better()

        while cur_evals < self.max_evals:
            # timeout
            if time.time() - start_time > time_limit:
                self.logger.info("Time out of limit, Epoch: {}".format(str(cur_evals)))
                break
            for i in range(slaves):
                sent = pipes[i].poll()
                if sent or statuses[i] == "ready":
                    if sent:
                        cur_trainer = pipes[i].recv()
                        cur_trainer = dill.loads(cur_trainer)
                        perf = pipes[i].recv()
                        self._update_trials(i, cur_trainer, perf, self.is_higher_better)
                        cur_evals += 1
                        if cur_evals >= self.max_evals:
                            break
                    else:
                        statuses[i] = "running"
                    x = self._get_suggestion(i)
                    pipes[i].send(x)

        for i in range(slaves):
            procs[i].terminate()

        best_trainer, best_x = self._best_hp()
        if best_x == None:
            raise TimeTooLimitedError(
                "Given time is too limited to finish one round in HPO."
            )
        return best_trainer, best_x

    @classmethod
    def build_hpo_from_args(cls, args):
        """Build a new hpo instance."""
        raise NotImplementedError(
            "HP Optimizer must implement the build_hpo_from_args method"
        )


class TimeTooLimitedError(Exception):
    pass


class WrongDependedParameterError(Exception):
    pass
