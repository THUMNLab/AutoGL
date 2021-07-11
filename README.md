# Auto Graph Learning

An autoML framework & toolkit for machine learning on graphs.

*Actively under development by @THUMNLab*

Feel free to open <a href="https://github.com/THUMNLab/AutoGL/issues">issues</a> or contact us at <a href="mailto:autogl@tsinghua.edu.cn">autogl@tsinghua.edu.cn</a> if you have any comments or suggestions!

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/autogl/badge/?version=latest)](https://autogl.readthedocs.io/en/latest/?badge=latest)

## News!

- 2021.07.11 New version! v0.2.0-pre is here! In new version, AutoGL support neural architecture search to customize the architectures for the given datasets and tasks. AutoGL also support sampling now to perform tasks on large datasets, including node-wise sampling, layer-wise sampling and sub-graph sampling. Link prediction task is now also supported! Learn more in our [tutorial](https://autogl.readthedocs.io/en/latest/index.html).
- 2021.04.10 Our paper [__AutoGL: A Library for Automated Graph Learning__](https://arxiv.org/abs/2104.04987) are accepted in _ICLR 2021 Workshop on Geometrical and Topological Representation Learning_! You can cite our paper following methods [here](#Cite).

## Introduction

AutoGL is developed for researchers and developers to quickly conduct autoML on the graph datasets & tasks. See our documentation for detailed information!

The workflow below shows the overall framework of AutoGL.

<img src="./resources/workflow.svg">

AutoGL uses `datasets` to maintain dataset for graph-based machine learning, which is based on Dataset in PyTorch Geometric with some support added to corporate with the auto solver framework.

Different graph-based machine learning tasks are solved by different `AutoGL solvers`, which make use of five main modules to automatically solve given tasks, namely `auto feature engineer`, `neural architecture search`, `auto model`, `hyperparameter optimization`, and `auto ensemble`. 

Currently, the following algorithms are supported in AutoGL:


<table>
    <tbody>
    <tr valign="top">
        <td>Feature Engineer</td>
        <td>Model</td>
        <td>NAS</td>
        <td>HPO</td>
        <td>Ensemble</td>
    </tr>
    <tr valign="top">
        <!--<td><b>Generators</b><br>graphlet <br> eigen <br> pagerank <br> PYGLocalDegreeProfile <br> PYGNormalizeFeatures <br> PYGOneHotDegree <br> onehot <br> <br><b>Selectors</b><br> SeFilterConstant<br> gbdt <br> <br><b>Subgraph</b><br> NxLargeCliqueSize<br> NxAverageClusteringApproximate<br> NxDegreeAssortativityCoefficient<br> NxDegreePearsonCorrelationCoefficient<br> NxHasBridge <br>NxGraphCliqueNumber<br> NxGraphNumberOfCliques<br> NxTransitivity<br> NxAverageClustering<br> NxIsConnected<br> NxNumberConnectedComponents<br> NxIsDistanceRegular<br> NxLocalEfficiency<br> NxGlobalEfficiency<br> NxIsEulerian </td>-->
        <td><b>Generators</b><br>graphlet <br> eigen <br> <a href="https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_fe.html">more ...</a><br><br><b>Selectors</b><br> SeFilterConstant<br> gbdt <br> <br><b>Graph</b><br> netlsd<br> NxAverageClustering<br> <a href="https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_fe.html">more ...</a></td>
        <td><b>Node Classification</b><br> GCN <br> GAT <br> GraphSAGE <br><br><b>Graph Classification</b><br> GIN <br> TopKPool </td>
        <td>
        <b>Algorithms</b><br>
        Random<br>
        RL<br>
        <a href='#'>more ...</a><br><br>
        <b>Spaces</b><br>
        SinglePath<br>
        GraphNas<br>
        <a href='#'>more ...</a><br><br>
        <b>Estimators</b><br>
        Oneshot<br>
        Scratch<br>
        </td>
        <td> Grid <br> Random <br> Anneal <br> Bayes <br> CAMES <br> MOCAMES <br> Quasi random <br> TPE <br> AutoNE </td>
        <td> Voting <br> Stacking </td>
    </tr>
    </tbody>
</table>

This toolkit also serves as a framework for users to implement and test their own autoML or graph-based machine learning models.

## Installation

### Requirements

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    see <https://pytorch.org/> for installation.

3. PyTorch Geometric (>=1.7.0)

    see <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html> for installation.

### Installation

#### Install from pip

Run the following command to install this package through `pip`.

```
pip install autogl
```

#### Install from source

Run the following command to install this package from the source.

```
git clone https://github.com/THUMNLab/AutoGL.git
cd AutoGL
python setup.py install
```

#### Install for development

If you are a developer of the AutoGL project, please use the following command to create a soft link, then you can modify the local package without install them again.

```
pip install -e .
```

## Documentation

Please refer to our <a href="https://autogl.readthedocs.io/en/latest/index.html">documentation</a> to see the detailed documentation.

You can also make the documentation locally. First, please install sphinx and sphinx-rtd-theme:
```
pip install -U Sphinx
pip install sphinx-rtd-theme
```
Then, make an html documentation by:
```
cd docs
make clean && make html
```

The documentation will be automatically generated under `docs/_build/html`

## Cite

You can cite [our paper](https://arxiv.org/abs/2104.04987) as follows if you use this code in your own work:
```
@inproceedings{
guan2021autogl,
title={Auto{GL}: A Library for Automated Graph Learning},
author={Chaoyu Guan and Ziwei Zhang and Haoyang Li and Heng Chang and Zeyang Zhang and Yijian Qin and Jiyan Jiang and Xin Wang and Wenwu Zhu},
booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
year={2021},
url={https://openreview.net/forum?id=0yHwpLeInDn}
}
```

## License

We follow [MIT license](LICENSE) across the entire codebase.
