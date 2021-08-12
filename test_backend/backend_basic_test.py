import os
from autogl.backend import DependentBackend


def test01():
    print(DependentBackend.is_dgl())
    print(DependentBackend.is_pyg())


def test02():
    os.environ["AUTOGL_BACKEND"] = "pyg"
    print(DependentBackend.is_dgl())
    print(DependentBackend.is_pyg())


if __name__ == '__main__':
    test02()
