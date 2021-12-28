# test the utils function

from autogl.datasets import utils, build_dataset_from_name

def test_graph_cross_validation():
    dataset = build_dataset_from_name('imdb-b')
    # first level, 10 folds
    utils.graph_cross_validation(dataset, 10)

    # set to fold id
    utils.set_fold(dataset, 1)

    # get train split
    train_dataset = utils.graph_get_split(dataset, "train", False)

    # further split train to train / val
    utils.graph_random_splits(train_dataset, 0.8, 0.2)

test_graph_cross_validation()

