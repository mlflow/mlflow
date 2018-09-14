Contributing to MLflow
======================
We welcome community contributions to MLflow. This page describes how to develop/test your changes
to MLflow locally.

Python API
----------

Prerequisites
~~~~~~~~~~~~~

We recommend installing MLflow in its own conda environment for development, as follows:

.. code:: bash

    conda create --name mlflow-dev-env
    source activate mlflow-dev-env
    pip install -r dev-requirements.txt
    pip install -r test-requirements.txt
    pip install -e .  # installs mlflow from current checkout


``npm`` is required to run the Javascript dev server.
You can verify that ``npm`` is on the PATH by running ``npm -v``, and
`install npm <https://www.npmjs.com/get-npm>`_ if needed.

Install Node Modules
~~~~~~~~~~~~~~~~~~~~
Before running the Javascript dev server or building a distributable wheel, install Javascript
dependencies via:

.. code:: bash

   cd mlflow/server/js
   npm install
   cd - # return to root repository directory

If modifying dependencies in ``mlflow/server/js/package.json``, run ``npm update`` within
``mlflow/server/js`` to install the updated dependencies.


Java API
--------

The MLflow Java client depends on the Python client, so first install the Python client in a conda
environment as described above. `Install <https://www.oracle.com/technetwork/java/javase/downloads/index.html>`_
the Java 8 JDK (or above), and `download <https://maven.apache.org/download.cgi>`_
and `install <https://maven.apache.org/install.html>`_ Maven. You can then build and run tests via:

.. code:: bash

  cd mlflow/java
  mvn clean package

R API
-----

The MLflow R client depends on the Python client, so first install the Python client in a conda
environment as described above. Install R, then run the following to install dependencies for
building MLflow locally

.. code:: bash

  cd mlflow/R/mlflow
  NOT_CRAN=true Rscript -e 'install.packages("devtools", "https://cloud.r-project.org")'
  NOT_CRAN=true Rscript -e 'devtools::install_deps(dependencies = TRUE)'

Build the R client via:

.. code:: bash

  R CMD build .

Run tests:

.. code:: bash

  R CMD check --no-build-vignettes --no-manual --no-tests mlflow*tar.gz
  cd tests
  LINTR_COMMENT_BOT=false Rscript ../.travis.R


Launching the Development UI
----------------------------
We recommend `Running the Javascript Dev Server`_ - otherwise, the tracking frontend will request
files in the ``mlflow/server/js/build`` directory, which is not checked into Git.
Alternatively, you can generate the necessary files in ``mlflow/server/js/build`` as described in
`Building a Distributable Artifact`_.


Tests and Lint
--------------
Verify that the unit tests & linter pass before submitting a pull request by running:

.. code:: bash

    pytest
    ./lint.sh

When running ``pytest --requires-ssh`` it is necessary that passwordless SSH access to localhost
is available. This can be achieved by adding the SSH public key to authorized keys:
``cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys``.


Running the Javascript Dev Server
---------------------------------
`Install Node Modules`_, then run the following:

In one shell:

.. code:: bash

   mlflow ui

In another shell:

.. code:: bash

   cd mlflow/server/js
   npm start

The MLflow Tracking UI will show runs logged in ``./mlruns`` at `<http://localhost:3000>`_.

Building a Distributable Artifact
---------------------------------
`Install Node Modules`_, then run the following:

Generate JS files in ``mlflow/server/js/build``:

.. code:: bash

   cd mlflow/server/js
   npm run build

Build a pip-installable wheel in ``dist/``:

.. code:: bash

   cd -
   python setup.py bdist_wheel

Building Protobuf Files
------------------------
To build protobuf files, simply run ``generate-protos.sh``. The required ``protoc`` version is ``3.6.0``.


Writing Docs
------------
Install the necessary Python dependencies via ``pip install -r dev-requirements.txt``. Then run

.. code:: bash

   cd docs
   make livehtml
