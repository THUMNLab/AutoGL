.. _backend:

Backend Support
===============

Currently, AutoGL support both pytorch geometric backend and deep graph library backend to
enable users from both end benifiting the automation of graph learning.

To specify one specific backend, you can declare the backend using environment variables
``AUTOGL_BACKEND``. For example:

.. code-block :: cmd

    AUTOGL_BACKEND=pyg python xxx.py

or

.. code-block :: python

    import os
    os.environ["AUTOGL_BACKEND"] = "pyg"
    import autogl
    
    ...

Users can use the backend as they wish to quickly conduct their automation experiments.
