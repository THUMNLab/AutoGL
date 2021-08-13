import os

os.environ["AUTOGL_BACKEND"] = "pyg"
from autogl.backend import DependentBackend

if __name__ == '__main__':
    print(DependentBackend.is_dgl())
    print(DependentBackend.is_pyg())
