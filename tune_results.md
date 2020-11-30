# Help tune the hp space of models

## New Feature

#### 2020.09.25

Now you can use multiprocessing to quickly get the final result. Seems quicker than `nodeclf_explore_hp_space.py`.
```
python nodeclf_hp_multiprocessing.py --dataset xxx xxx --exclude_device 3 4 5 --max_evals 1 5 10 20 50 --path_to_config ../configs/your_config.yaml --hpo random (or tpe) --cluster32
```
You can use exclude_device to ignore the devices that you can see in `NVIDIA-SMI`.

**attention!**
DO NOT USE CUDA_VISIBLE_DEVICES=xx to force the devices, which will be of no effect when running this code.

**attention!**
If you run your code on cluster32, you must add `--cluster32` since the GPU mapping in cluster32 is a bit strange.

## Environment
- Python >= 3.6.0
- PyTorch >= 1.5.1
- PyTorch-Geometric >= 1.5.1
- scikit-learn >= 0.23.2
- tabulate >= 0.8.7
- lightgbm >= 2.3.0
- psutil >= 5.7.2
- hyperopt >= 0.2.4
- chocolate >= 0.0.2
- bayesian-optimization >= 1.2.0
- NetLSD >= 1.0.2

## How to tune

### 1. change the model and hp space

You need to change certain lines in python file `examples/graphclf_explore_hp_space.py` or `examples/nodeclf_explore_hp_space.py` to set your models and adjust your hp spaces. The hp space example is given at the beginning of the files.

### 2. run the code

```
cd examples
python xxx_explore_hp_space.py --dataset d1 d2 d3 ... --max_evals max_hpo_eval_time
```
Run the cmd above to get randomly sampled `max_hpo_eval_time` models out of your search space on the specifid dataset.

### 3. analyze the results
The step 2 will give you the models with results. You need to adjust the hp space in step 1 according to the results.

To help simplify the analyze process, you can also leverage the analyze tools under `analysis` folder.

First, copy and paste the results (all the lines with the format `{*some hp} score`. Refer to `analysis/record.txt` for an example) to `analysis/record.txt`, then run:
```
python ./analysis/analysis_result.py --path ./analysis/record.txt
```

This will sort the results for you, so that you can observe the common problems bad models have and regularize the corresponding hp space.

### 4. update log

2020.09.03 Now the re-factored models are merged from branch `lhy_graph_clf`.

2020.09.06 Update all modules from master, add graph explore support.