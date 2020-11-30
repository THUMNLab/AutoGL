import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name, utils
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc, BaseModel

dataset = build_dataset_from_name('mutag')
utils.graph_random_splits(dataset, train_ratio=0.4, val_ratio=0.4)

autoClassifier = AutoGraphClassifier.from_config('../configs/graph_classification.yaml')

# train
autoClassifier.fit(
    dataset, 
    time_limit=3600, 
    train_split=0.8, 
    val_split=0.1, 
    cross_validation=True,
    cv_split=10, 
)
autoClassifier.get_leaderboard().show()

print('best single model:\n', autoClassifier.get_leaderboard().get_best_model(0))

# test
predict_result = autoClassifier.predict_proba()
print(Acc.evaluate(predict_result, dataset.data.y[dataset.test_index].cpu().detach().numpy()))