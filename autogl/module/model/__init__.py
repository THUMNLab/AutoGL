import importlib
import sys
from ...backend import DependentBackend

# load corresponding backend of subclass
def _load_subclass_backend(backend):
    sub_module = importlib.import_module(f'.{backend.get_backend_name()}', __name__)
    this = sys.modules[__name__]
    for api, obj in sub_module.__dict__.items():
        setattr(this, api, obj)

_load_subclass_backend(DependentBackend)
