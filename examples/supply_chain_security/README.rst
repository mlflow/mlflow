Python Package Anti-Tampering with MLflow
-----------------------------------------
This directory contains an MLflow project showing how to harden the ML supply chain, and in particular
how to protect against Python package tampering by enforcing
`hash checks <https://pip.pypa.io/en/latest/cli/pip_install/#hash-checking-mode>`_ on packages.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``).

The model is trained locally by running:

.. code-block:: bash

  mlflow run .

At the end of the training, note the run ID (say ``e651fcd4dab140a2bd4d3745a32370ac``).

The model is served locally by running:

.. code-block:: bash

  mlflow models serve -m runs:/e651fcd4dab140a2bd4d3745a32370ac/model

Inference is performed by sending JSON POST requests to http://localhost:5000/invocations:

.. code-block:: bash

  curl -X POST -d "{\"dataframe_split\": {\"data\":[[0.0199132142,0.0506801187,0.1048086895,0.0700725447,-0.0359677813,-0.0266789028,-0.0249926566,-0.002592262,0.0037117382,0.0403433716]]}}" -H "Content-Type: application/json" http://localhost:5000/invocations

Which returns ``[235.11371081266924]``.

Structure of this MLflow Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  name: mlflow-supply-chain-security
  channels:
  - nodefaults
  dependencies:
  - python=3.9
  - pip
  - pip:
    - --require-hashes
    - -r requirements.txt

This ensures that all the package requirements referenced in ``requirements.txt`` have been pinned through both version and hash:

.. code-block:: text

  mlflow==1.20.2 \
      --hash=sha256:963c22532e82a93450674ab97d62f9e528ed0906b580fadb7c003e696197557c \
      --hash=sha256:b15ff0c7e5e64f864a0b40c99b9a582227315eca2065d9f831db9aeb8f24637b
  numpy==1.21.4 \
      --hash=sha256:0b78ecfa070460104934e2caf51694ccd00f37d5e5dbe76f021b1b0b0d221823 \
  ...

That same conda environment is referenced when logging the model in ``train.py`` so the environment matches during inference:

.. code-block:: python

  mlflow.sklearn.log_model(
      model,
      name="model",
      signature=mlflow.models.infer_signature(X_train[:10], y_train[:10]),
      input_example=X_train[:10],
      conda_env="conda.yaml",
  )

The package requirements are managed in ``requirements.in``:

.. code-block:: text

  pandas==1.3.2
  scikit-learn==0.24.2
  mlflow==1.20.2

They are compiled using ``pip-tools`` to resolve all the package dependencies, their versions, and their hashes:

.. code-block:: bash

  pip install pip-tools
  pip-compile --generate-hashes --output-file=requirements.txt requirements.in
