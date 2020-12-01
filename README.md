# Auto Graph Learning

An autoML framework & toolkit for machine learning on graphs.

*Actively under development by @THUMNLab*

Feel free to open <a href="https://github.com/THUMNLab/AutoGL/issues">issues</a> or contact us at <a href="mailto:autogl@126.com">autogl@126.com</a> if you have any comments or suggestions!

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/autogl/badge/?version=latest)](https://autogl.readthedocs.io/en/latest/?badge=latest)
## Introduction

AutoGL is developed for researchers and developers to quickly conduct autoML on the graph datasets & tasks. See our documentation for detailed information!

The workflow below shows the overall framework of AutoGL.

<img src="./resources/workflow.svg">

AutoGL uses `datasets` to maintain dataset for graph-based machine learning, which is based on Dataset in PyTorch Geometric with some support added to corporate with the auto solver framework.

Different graph-based machine learning tasks are solved by different `AutoGL solvers`, which make use of four main modules to automatically solve given tasks, namely `auto feature engineer`, `auto model`, `hyperparameter optimization`, and `auto ensemble`. 

Currently, the following algorithms are supported in AutoGL:


<table>
    <tbody>
    <tr valign="top">
        <td>Feature Engineer</td>
        <td>Model</td>
        <td>HPO</td>
        <td>Ensemble</td>
    </tr>
    <tr valign="top">
        <!--<td><b>Generators</b><br>graphlet <br> eigen <br> pagerank <br> PYGLocalDegreeProfile <br> PYGNormalizeFeatures <br> PYGOneHotDegree <br> onehot <br> <br><b>Selectors</b><br> SeFilterConstant<br> gbdt <br> <br><b>Subgraph</b><br> NxLargeCliqueSize<br> NxAverageClusteringApproximate<br> NxDegreeAssortativityCoefficient<br> NxDegreePearsonCorrelationCoefficient<br> NxHasBridge <br>NxGraphCliqueNumber<br> NxGraphNumberOfCliques<br> NxTransitivity<br> NxAverageClustering<br> NxIsConnected<br> NxNumberConnectedComponents<br> NxIsDistanceRegular<br> NxLocalEfficiency<br> NxGlobalEfficiency<br> NxIsEulerian </td>-->
        <td><b>Generators</b><br>graphlet <br> eigen <br> <a href="https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_fe.html">more ...</a><br><br><b>Selectors</b><br> SeFilterConstant<br> gbdt <br> <br><b>Subgraph</b><br> netlsd<br> NxAverageClustering<br> <a href="https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_fe.html">more ...</a></td>
        <td><b>Node Classification</b><br> GCN <br> GAT <br> GraphSAGE <br><br><b>Graph Classification</b><br> GIN <br> TopKPool </td>
        <td> Grid <br> Random <br> Anneal <br> Bayes <br> CAMES <br> MOCAMES <br> Quasi random <br> TPE <br> AutoNE </td>
        <td> Voting </td>
    </tr>
    </tbody>
</table>

This toolkit also serves as a platform for users to implement and test their own autoML or graph-based machine learning models.

## Installation

### Requirements

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.5.1)

    see <https://pytorch.org/> for installation.

3. PyTorch Geometric

    see <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html> for installation.

### Installation

#### Install from pip

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

## License

We follow [MIT license](LICENSE) across the entire codebase.
