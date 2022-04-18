.. _backend:

Backend Support
===============

Currently, AutoGL support both pytorch geometric backend and deep graph library backend to
enable users from both end benifiting the automation of graph learning.

To specify one specific backend, you can declare the backend using environment variables
``AUTOGL_BACKEND``. For example:

.. code-block:: python

    AUTOGL_BACKEND=pyg python xxx.py

or

.. code-block:: python

    import os
    os.environ["AUTOGL_BACKEND"] = "pyg"
    import autogl
    
    ...


If no backend is specified, AutoGL will use the backend in your environment. If you have both
Deep Graph Library and PyTorch Geometric installed, the default backend will be Deep Graph Library.

You can also get current backend in the code by:

.. code-block :: python

    from autogl.backend import DependentBackend
    print(DependentBackend.get_backend_name())
