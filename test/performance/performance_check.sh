# graph clf - dgl
for name in "base" "model" "trainer" "trainer_dataset" "solver"
do
    python graph_classification/dgl/$name.py --repeat 10 --dataset MUTAG > graph_classification/dgl/$name.log 2>&1
done

# graph clf - pyg
for name in "base" "model" "trainer" "trainer_dataset" "solver"
do
    for dataset in "MUTAG" "COLLAB"
    do
        for model in "gin" "topkpool"
        do
            python graph_classification/pyg/$name.py --repeat 10 --dataset $dataset --model $model > graph_classification/pyg/$name-$dataset-$model.log 2>&1
        done
    done
done

# node clf
for backend in "pyg" "dgl"
do
    for name in "base" "model" "trainer" "trainer_dataset" "solver"
    do
        for dataset in "Cora" "CiteSeer" "PubMed"
        do
            for model in "gcn" "gat" "sage" "gin"
            do
                python node_classification/$backend/$name.py --repeat 10 --dataset $dataset --model $model > node_classification/$backend/$name-$dataset-$model.log 2>&1
            done
        done
    done
done
