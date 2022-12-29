# 智图 (AutoGL)
[English Introduction](https://www.gitlink.org.cn/THUMNLab/AutoGL/tree/main/README_en.md)
用于图数据的自动机器学习框架和工具包。

*由清华大学媒体与网络实验室进行开发与维护*
[Chinese Introduction](README_cn.md)

An autoML framework & toolkit for machine learning on graphs.

若有任何意见或建议，欢迎通过<a href="https://www.gitlink.org.cn/THUMNLab/AutoGL/issues">issues</a> 或邮件<a href="mailto:autogl@tsinghua.edu.cn">autogl@tsinghua.edu.cn</a>与我们联系。

<!--
 [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
% [![Documentation Status](http://mn.cs.tsinghua.edu.cn/autogl/documentation/?badge=latest)](http://mn.cs.tsinghua.edu.cn/autogl/documentation/?badge=latest)-->

## News!

- 2022.12.29 New version! v0.4.0-pre is here!
    - We have proposed __NAS-Bench-Graph__ ([paper](https://openreview.net/pdf?id=bBff294gqLp), [code](https://github.com/THUMNLab/NAS-Bench-Graph), [tutorial](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_nas_bench_graph.html)), the first NAS-benchmark for graphs published in NeurIPS'22. By using AutoGL together with NAS-Bench-Graph, the performance estimation process of GraphNAS algorithms can be greatly speeded up. 
    - We have supported the graph __robustness__ algorithms in AutoGL, including graph structure engineering, robust GNNs and robust GraphNAS. See [robustness tutorial](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_robust.html) for more details.
    - We have supported graph __self-supervised learning__! See [self-supervised learning tutorial](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_ssl_trainer.html) for more details.
- 2021.12.31 Version v0.3.0-pre is released
    - Support [__Deep Graph Library (DGL)__](https://www.dgl.ai/) backend including homogeneous node classification, link prediction, and graph classification tasks. AutoGL is also compatible with PyG 2.0 now.
    - Support __heterogeneous__ node classification! See [hetero tutorial](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_hetero_node_clf.html) .
    - The module `model` now supports __decoupled__ to two additional sub-modules named `encoder` and `decoder`. Under the __decoupled__ design, one `encoder` can be used to solve all kinds of tasks.
    - Enrich [NAS algorithms](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_nas.html) such as [AutoAttend](https://proceedings.mlr.press/v139/guan21a.html), [GASSO](https://proceedings.neurips.cc/paper/2021/hash/8c9f32e03aeb2e3000825c8c875c4edd-Abstract.html), [hardware-aware algorithm](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/documentation/nas.html#autogl.module.nas.estimator.OneShotEstimator_HardwareAware), etc. 
- 2021.07.11 Version 0.2.0-pre is released, which supports [neural architecture search (NAS)](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_nas.html) to customize architectures, [sampling (http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_trainer.html#node-classification-with-sampling) to perform tasks on large datasets, and link prediction. 
- 2021.04.16 Our survey paper about automated machine learning on graphs is accepted by IJCAI! See more [here](http://arxiv.org/abs/2103.00742).
- 2021.04.10 Our paper [__AutoGL: A Library for Automated Graph Learning__](https://arxiv.org/abs/2104.04987) is accepted by _ICLR 2021 Workshop on Geometrical and Topological Representation Learning_! You can cite our paper following methods [here](#Cite).

## 介绍

智图的设计目标是可以简单、快速地对图数据集和任务进行自动机器学习，可供研究者和开发者使用。更多详细信息，可以参阅我们的文档。

下图是智图的整体框架。

<img src="./resources/workflow.svg">

智图通过 `datasets` 类以支持图数据集，其基于 PyTorch Geometric 和 Deep Graph Library 的数据集，并添加了一些函数以支持自动机器学习框架。

智图通过 `AutoGL solvers` 以处理不同的图机器学习任务，利用五个主要模块自动解决给定的任务，即自动特征工程 `auto feature engineer`，神经架构搜索 `neural architecture search`，自动模型 `auto model`，超参数优化 `hyperparameter optimization`，和自动模型集成 `auto ensemble`。

目前，智图支持以下算法：

<table>
    <tbody>
    <tr valign="top">
        <td>特征工程</td>
        <td>图模型</td>
        <td>神经架构搜索</td>
        <td>超参数优化</td>
        <td>模型集成</td>
    </tr>
    <tr valign="top">
        <td><b>生成器</b><br>Graphlets <br> EigenGNN <br> <a href="http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_fe.html">更多 ...</a><br><br><b>选择器</b><br> SeFilterConstant<br> gbdt <br> <br><b>全图特征</b><br> Netlsd<br> NxAverageClustering<br> <a href="http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_fe.html">更多 ...</a></td>
        <td><b>同构图编码器</b><br> GCNEncoder <br> GATEncoder <br> SAGEEncoder <br> GINEncoder <br> <br><b>解码器</b><br>LogSoftmaxDecoder <br> DotProductDecoder <br> SumPoolMLPDecoder <br> JKSumPoolDecoder </td>
        <td>
        <b>搜索算法</b><br>
        Random<br>
        RL<br>
        Evolution<br>
        GASSO<br>
        <a href='http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/documentation/nas.html'>更多 ...</a><br><br>
        <b>搜索空间</b><br>
        SinglePath<br>
        GraphNas<br>
        AutoAttend<br>
        <a href='http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/documentation/nas.html'>更多 ...</a><br><br>
        <b>模型评估</b><br>
        Oneshot<br>
        Scratch<br>
        </td>
        <td> Grid <br> Random <br> Anneal <br> Bayes <br> CAMES <br> MOCAMES <br> Quasi random <br> TPE <br> AutoNE </td>
        <td> Voting <br> Stacking </td>
    </tr>
    </tbody>
</table>

此工具包还可作为一个框架供用户实现和测试自己的自动机器学习或图机器学习模型。

## 安装

### 依赖

在安装智图之前，请首先安装以下依赖项。

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    详细信息请参考<https://pytorch.org/>。

3. 图机器学习工具包

    智图需要 PyTorch Geometric（PyG）或 Deep Graph Library（DGL）作为后端。若两者均安装，可在运行时选择任一后端，参考[这里](http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_backend.html)。

    3.1 PyTorch Geometric (>=1.7.0)

    详细信息请参考<https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html/>。

    3.2 Deep Graph Library (>=0.7.0)

    详细信息请参考<https://dgl.ai/>。


### 安装

#### 通过pip进行安装

运行以下命令以通过`pip`安装智图。

```
pip install autogl
```

#### 从源代码安装

运行以下命令以从源安装智图。

```
git clone https://gitlink.org.cn/THUMNLab/AutoGL.git
cd AutoGL
python setup.py install
```

#### 开发者安装

如果您想以开发者方式安装智图，请运行以下命令以创建软链接，然后即可修改本地程序后而无需重复安装。
```
pip install -e .
```

## 文档

您可参考<a href="http://mn.cs.tsinghua.edu.cn/autogl/documentation/">文档页面</a> 以参阅我们的详细文档。

文档也可以进行本地编译。首先，请安装 sphinx 和 sphinx-rtd-theme:
```
pip install -U Sphinx
pip install sphinx-rtd-theme
```

然后，通过以下方式创建 html 文档：
```
cd docs
make clean && make html
```
文档将在如下路径自动生成：`docs/_build/html`

## 引用

如果您使用了智图代码，请按如下方式引用我们的[论文](https://openreview.net/forum?id=0yHwpLeInDn):
```
@inproceedings{guan2021autogl,
  title={Auto{GL}: A Library for Automated Graph Learning},
  author={Chaoyu Guan and Ziwei Zhang and Haoyang Li and Heng Chang and Zeyang Zhang and Yijian Qin and Jiyan Jiang and Xin Wang and Wenwu Zhu},
  booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
  year={2021},
  url={https://openreview.net/forum?id=0yHwpLeInDn}
}
```

或许您也会发现我们的[综述](http://arxiv.org/abs/2103.00742)有帮助:
```
@article{zhang2021automated,
  title={Automated Machine Learning on Graphs: A Survey},
  author={Zhang, Ziwei and Wang, Xin and Zhu, Wenwu},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  year={2021},
  note={Survey track}
}
```

## 版权相关
从v0.2版本开始，智图的所有代码采用[Apache license](LICENSE)。

