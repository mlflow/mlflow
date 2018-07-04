Contributing to MLflow
======================
We welcome community contributions to MLflow. This page describes how to develop/test your changes
to MLflow locally.

Prerequisites
-------------

We recommend installing MLflow in its own virtualenv for development, as follows:

.. code:: bash

    virtualenv env
    source env/bin/activate
    pip install -r dev-requirements.txt
    pip install -r tox-requirements.txt
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

If modifying dependencies in ``mlflow/server/js/package.json``, run `npm update` within
``mlflow/server/js`` to install the updated dependencies.


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
