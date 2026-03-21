"""Export and import of generic R models.

This module defines generic filesystem format for R models and provides utilities
for saving and loading to and from this format. The format is self contained in the sense
that it includes all necessary information for anyone to load it and use it. Dependencies
are either stored directly with the model or referenced via a Conda environment.

The convention for rfunc models is to have a ``predict`` method or function with the following
signature::

    predict(data: DataFrame) -> DataFrame

This convention is relied on by other MLflow components.

Rfunc model format is defined as a directory structure containing all required data, code, and
configuration:

.. code::

    ./dst-path/
            ./MLmodel: configuration

It must contain MLmodel file in its root with "r_function" format.

Example:

.. code:: shell

  >tree R/mlflow/R/inst/examples/R/lm/model
  ├── MLmodel
  └── r_model.bin

  >cat R/mlflow/R/inst/examples/R/lm/model/MLmodel
  time_created: 1.5337659e+09
  flavors:
    r_function:
      version: 0.1.0
      model: r_model.bin

"""

FLAVOR_NAME = "crate"
