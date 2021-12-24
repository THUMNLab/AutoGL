import os
import autogl

def test_backend():
    environ = os.environ.get("AUTOGL_BACKEND", None)
    backend_name = autogl.backend.DependentBackend.get_backend_name()
    if environ in ['pyg', 'dgl']:
        assert backend_name == environ
    else:
        try:
            import dgl
            assert backend_name == 'dgl'
            return
        except ImportError:
            pass

        try:
            import torch_geometric
            assert backend_name == 'pyg'
            return
        except ImportError:
            pass

if __name__ == '__main__':
    test_backend()
