mlflow.pyfunc
=============

.. automodule:: mlflow.pyfunc
    :members:
    :undoc-members:
    :show-inheritance:

.. Include ``get_default_pip_requirements`` and ``get_default_conda_env``,
   which are imported from `mlflow.pyfunc.model`, in the `mlflow.pyfunc` namespace
.. autofunction:: mlflow.pyfunc.get_default_pip_requirements
.. autofunction:: mlflow.pyfunc.get_default_conda_env

.. Include ``PythonModelContext`` as a renamed class to avoid documenting constructor parameters.
   This class is meant to be constructed implicitly, and users should only be aware of its
   documented member properties.
.. autoclass:: mlflow.pyfunc.PythonModelContext()
    :members:
    :undoc-members:

.. Include ``PythonModel``, which is imported from `mlflow.pyfunc.model`, in the
   `mlflow.pyfunc` namespace
.. autoclass:: mlflow.pyfunc.PythonModel
    :members:
    :undoc-members:

.. Include ``ChatModel``, which is imported from `mlflow.pyfunc.model`, in the
   `mlflow.pyfunc` namespace
.. autoclass:: mlflow.pyfunc.ChatModel
    :members:
    :undoc-members:
