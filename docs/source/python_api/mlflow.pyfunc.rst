mlflow.pyfunc
=============

.. automodule:: mlflow.pyfunc
    :members:
    :undoc-members:
    :show-inheritance:

.. Include ``PythonModelContext`` as a renamed class to avoid documenting constructor parameters.
   This class is meant to be constructed implicitly, and users should only be aware of its
   documented member properties.
.. autoclass:: mlflow.pyfunc.PythonModelContext()
    :members:
    :undoc-members:

.. Include ``PythonModel`` as a class to include documentation for the ``__init__`` method
   without documenting ``__init__`` for other classes in the same module.
.. autoclass:: mlflow.pyfunc.PythonModel
    :members:
    :undoc-members:

    .. automethod:: __init__
