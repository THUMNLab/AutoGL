.. _backend_cn:

Backend Support
===============

目前，AutoGL支持使用PyTorch-Geometric或Deep Graph Library作为后端，以便熟悉两者之一的用户均可受益于自动图学习。

为指定特定的后端，用户可以使用环境变量``AUTOGL_BACKEND``进行声明，例如：

.. code-block:: python

    AUTOGL_BACKEND=pyg python xxx.py

或

.. code-block:: python

    import os
    os.environ["AUTOGL_BACKEND"] = "pyg"
    import autogl

    ...


如果环境变量``AUTOGL_BACKEND``未声明，AutoGL会根据用户的Python运行环境中所安装的图学习库自动选择。
如果PyTorch-Geometric和Deep Graph Library均已安装，则Deep Graph Library将被作为默认的后端。

可以以编程方式获得当前使用的后端：

.. code-block:: python

    from autogl.backend import DependentBackend
    print(DependentBackend.get_backend_name())
