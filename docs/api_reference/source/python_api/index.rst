.. _python-api:

Python API
==========

The MLflow Python API is organized into the following modules. The most common functions are
exposed in the :py:mod:`mlflow` module, so we recommend starting there.

.. toctree::
  :glob:
  :maxdepth: 1

  *
  openai/index.rst


See also the :ref:`index of all functions and classes<genindex>`.

Log Levels
----------

MLflow Python APIs log information during execution using the Python Logging API. You can 
configure the log level for MLflow logs using the following code snippet. Learn more about Python
log levels at the
`Python language logging guide <https://docs.python.org/3/howto/logging.html>`_.

.. code-block:: python

    import logging

    logger = logging.getLogger("mlflow")

    # Set log level to debugging
    logger.setLevel(logging.DEBUG)
