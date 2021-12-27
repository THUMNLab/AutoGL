# https://docs.python.org/3/using/cmdline.html
# 运行 bash /home/jcai/code/AutoGL/test/performance/link_prediction/pyg/link_prediction_model.bash
# MODEL="gcn"
# # python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gcn" --repeat 10 --dataset "Cora"
# python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gcn" --repeat 10 --dataset "PubMed"
# python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gcn" --repeat 10 --dataset "CiteSeer"
MODEL="gat"
# python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gat" --repeat 10 --dataset "Cora"
python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gat" --repeat 10 --dataset "PubMed"
python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "gat" --repeat 10 --dataset "CiteSeer"
# MODEL="sage"
python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "sage" --repeat 10 --dataset "Cora"
python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "sage" --repeat 10 --dataset "PubMed"
python AutoGL/test/performance/link_prediction/pyg/link_prediction_model.py --model "sage" --repeat 10 --dataset "CiteSeer"