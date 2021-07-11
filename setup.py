try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch not installed. "
        "Please appropriately install PyTorch, "
        "see https://pytorch.org/ for installation."
    )
try:
    import torch_scatter
    import torch_sparse
    import torch_geometric
    PYG_VER = torch_geometric.__version__.split('.')
    PYG_VER = [int(PYG_VER[0]), int(PYG_VER[1])]
    assert PYG_VER >= [1, 7], "PyTorch-Geometric version should be at least 1.7.0"
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch-Geometric not fully installed. "
        "Please appropriately install PyTorch-Geometric, "
        "see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for installation."
    )

if torch.__version__.startswith('1.8.'):
    try:
        import torch_cluster
        import torch_spline_conv
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyTorch-Geometric not fully installed. "
            "For PyTorch version 1.8.x, you should also install torch_cluster and torch_spline_conv "
            "see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for installation."
        )

from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

''' https://packaging.python.org/guides/distributing-packages-using-setuptools/ '''
''' https://setuptools.readthedocs.io/en/latest/ '''
setup(
    name='autogl',
    version='0.2.0-pre',
    author='THUMNLab/aglteam',
    maintainer='THUMNLab/aglteam',
    author_email='autogl@tsinghua.edu.cn',
    description='AutoML tools for graph-structure dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(),
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='~=3.6',
    # https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6"
    ],
    # https://setuptools.readthedocs.io/en/latest/userguide/dependency_management.html
    # note that setup_requires and tests_require are deprecated
    install_requires=[
        'bayesian-optimization',
        'chocolate',
        'dill',
        'hyperopt',
        'lightgbm',
        'networkx',
        'numpy',
        'netlsd',
        'ogb',
        'psutil',
        'pyyaml',
        'requests',
        'scikit-learn',
        'scipy',
        'tabulate',
        'torch',
        'torch-geometric',
        'torch-scatter',
        'torch-sparse',
        'tqdm',
        'nni'
    ]
)