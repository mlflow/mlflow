Contributing to MLflow
======================
We welcome community contributions to MLflow. This page provides useful information about
contributing to MLflow.

.. contents:: **Table of Contents**
  :local:
  :depth: 3

Governance
##########

Governance of MLflow is conducted by the Technical Steering Committee (TSC), which is currently includes the following members:

 - Patrick Wendell (pwendell@gmail.com)

 - Reynold Xin (reynoldx@gmail.com)

 - Matei Zaharia (matei@cs.stanford.edu)

The founding technical charter can be found `here <https://github.com/mlflow/mlflow/blob/master/mlflow-charter.pdf>`_.

Contribution process
####################
The MLflow contribution process starts with filing a GitHub issue. MLflow defines four
categories of issues: feature requests, bug reports, documentation fixes, and installation issues.
Details about each issue type and the issue lifecycle are discussed in the `MLflow Issue Policy
<https://github.com/mlflow/mlflow/blob/master/ISSUE_POLICY.md>`_.

MLflow committers actively `triage <ISSUE_TRIAGE.rst>`_ and respond to GitHub issues. In general, we
recommend waiting for feebdack from an MLflow committer or community member before proceeding to 
implement a feature or patch. This is particularly important for
`significant changes <https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst#write-designs-for-significant-changes>`_,
and will typically be labeled during triage with ``needs design``.

After you have agreed upon an implementation strategy for your feature or patch with an MLflow
committer, the next step is to introduce your changes (see `developing changes
<https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst#developing-and-testing-changes-to-mlflow>`_)
as a pull request against the MLflow Repository or as a standalone MLflow Plugin. MLflow committers
actively review pull requests and are also happy to provide implementation guidance for Plugins.

Once your pull request against the MLflow Repository has been merged, your corresponding changes
will be automatically included in the next MLflow release. Every change is listed in the MLflow
release notes and `Changelog <https://github.com/mlflow/mlflow/blob/master/CHANGELOG.rst>`_.
Congratulations, you have just contributed to MLflow! We appreciate your contribution!

Contribution guidelines
#######################
In this section, we provide guidelines to consider as you develop new features and patches for
MLflow.

Write designs for significant changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For significant changes to MLflow, we recommend outlining a design for the feature or patch and discussing it with
an MLflow committer before investing heavily in implementation. During issue triage, we try to proactively
identify issues require design by labeling them with ``needs design``. This is particularly important if your 
proposed implementation:

- Introduces changes or additions to the `MLflow REST API <https://mlflow.org/docs/latest/rest-api.html>`_

  - The MLflow REST API is implemented by a variety of open source and proprietary platforms. Changes to the REST
    API impact all of these platforms. Accordingly, we encourage developers to thoroughly explore alternatives
    before attempting to introduce REST API changes.

- Introduces new user-facing MLflow APIs

  - MLflow's API surface is carefully designed to generalize across a variety of common ML operations.
    It is important to ensure that new APIs are broadly useful to ML developers, easy to work with,
    and simple yet powerful.

- Adds new library dependencies to MLflow

- Makes changes to critical internal abstractions. Examples include: the Tracking Artifact Repository,
  the Tracking Abstract Store, and the Model Registry Abstract Store.

Make changes backwards compatibile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MLflow's users rely on specific platform and API behaviors in their daily workflows. As new versions
of MLflow are developed and released, it is important to ensure that users' workflows continue to
operate as expected. Accordingly, please take care to consider backwards compatibility when introducing
changes to the MLflow code base. If you are unsure of the backwards compatibility implications of
a particular change, feel free to ask an MLflow committer or community member for input.

Consider introducing new features as MLflow Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`MLflow Plugins <https://mlflow.org/docs/latest/plugins.html>`_ enable integration of third-party modules with many of
MLflow’s components, allowing you to maintain and iterate on certain features independently of the MLflow Repository.
Before implementing changes to the MLflow code base, consider whether your feature might be better structured as an
MLflow Plugin. MLflow Plugins are a great choice for the following types of changes:

1. Supporting a new storage platform for MLflow artifacts
2. Introducing a new implementation of the MLflow Tracking backend (`Abstract Store
   <https://github.com/mlflow/mlflow/blob/cdc6a651d5af0f29bd448d2c87a198cf5d32792b/mlflow/store/tracking/abstract_store.py>`_)
   for a particular platform
3. Introducing a new implementation of the Model Registry backend (`Abstract Store
   <https://github.com/mlflow/mlflow/blob/cdc6a651d5af0f29bd448d2c87a198cf5d32792b/mlflow/store/model_registry/abstract_store.py>`_)
   for a particular platform
4. Automatically capturing and recording information about MLflow Runs created in specific environments

MLflow committers and community members are happy to provide assistance with the development and review of
new MLflow Plugins.

Finally, MLflow maintains a list of Plugins developed by community members, which is located at
https://mlflow.org/docs/latest/plugins.html#community-plugins. This is an excellent way to
inform MLflow users about your exciting new Plugins. To list your plugin, simply introduce
a new pull request against the `corresponding docs section of the MLflow code base
<https://github.com/mlflow/mlflow/blob/cdc6a651d5af0f29bd448d2c87a198cf5d32792b/docs/source/plugins.rst#community-plugins>`_.

For more information about Plugins, see https://mlflow.org/docs/latest/plugins.html.

Developing and testing changes to MLflow
########################################
The majority of the MLflow codebase is developed in Python. This includes the CLI, Tracking Server,
Artifact Repositories (e.g., S3 or Azure Blob Storage backends), and of course the Python fluent,
tracking, and model APIs.

Prerequisites
~~~~~~~~~~~~~

First, install the Python MLflow package from source - this is required for developing & testing
changes across all languages and APIs. We recommend installing MLflow in its own conda environment
by running the following from your checkout of MLflow:

.. code-block:: bash

    conda create --name mlflow-dev-env python=3.6
    source activate mlflow-dev-env
    pip install -r dev-requirements.txt
    pip install -r test-requirements.txt
    pip install -e .  # installs mlflow from current checkout

You may need to run ``conda install cmake`` for the test requirements to properly install, as ``onnx`` needs ``cmake``.

Ensure `Docker <https://www.docker.com/>`_ is installed.

``npm`` is required to run the Javascript dev server and the tracking UI.
You can verify that ``npm`` is on the PATH by running ``npm -v``, and
`install npm <https://www.npmjs.com/get-npm>`_ if needed.

If contributing to MLflow's R APIs, install `R <https://cloud.r-project.org/>`_. For changes to R
documentation, also install `pandoc <https://pandoc.org/installing.html>`_ 2.2.1 or above,
verifying the version of your installation via ``pandoc --version``. If using Mac OSX, note that
the homebrew installation of pandoc may be out of date - you can find newer pandoc versions at
https://github.com/jgm/pandoc/releases.

If contributing to MLflow's Java APIs or modifying Java documentation,
install `Java <https://www.java.com/>`_ and `Apache Maven <https://maven.apache.org/download.cgi>`_.


Install Node Modules
~~~~~~~~~~~~~~~~~~~~
Before running the Javascript dev server or building a distributable wheel, install Javascript
dependencies via:

.. code-block:: bash

   cd mlflow/server/js
   npm install
   cd - # return to root repository directory

If modifying dependencies in ``mlflow/server/js/package.json``, run ``npm update`` within
``mlflow/server/js`` to install the updated dependencies.


Java
~~~~
Certain MLflow modules are implemented in Java, under the ``mlflow/java/`` directory.
These are the Java Tracking API client (``mlflow/java/client``) and the Model Scoring Server
for Java-based models like MLeap (``mlflow/java/scoring``).

Other Java functionality (like artifact storage) depends on the Python package, so first install
the Python package in a conda environment as described above.
`Install <https://www.oracle.com/technetwork/java/javase/downloads/index.html>`_
the Java 8 JDK (or above), and `download <https://maven.apache.org/download.cgi>`_
and `install <https://maven.apache.org/install.html>`_ Maven. You can then build and run tests via:

.. code-block:: bash

  cd mlflow/java
  mvn compile test

If opening a PR that makes API changes, please regenerate API documentation as described in
`Writing Docs`_ and commit the updated docs to your PR branch.


R
~

The ``mlflow/R/mlflow`` directory contains R wrappers for the Projects, Tracking and Models
components. These wrappers depend on the Python package, so first install
the Python package in a conda environment:

.. code-block:: bash

  # Note that we don't pass the -e flag to pip, as the R tests attempt to run the MLflow UI
  # via the CLI, which will not work if we run against the development tracking server
  pip install .


`Install R <https://cloud.r-project.org/>`_, then run the following to install dependencies for
building MLflow locally:

.. code-block:: bash

  cd mlflow/R/mlflow
  NOT_CRAN=true Rscript -e 'install.packages("devtools", repos = "https://cloud.r-project.org")'
  NOT_CRAN=true Rscript -e 'devtools::install_deps(dependencies = TRUE)'

Build the R client via:

.. code-block:: bash

  R CMD build .

Run tests:

.. code-block:: bash

  R CMD check --no-build-vignettes --no-manual --no-tests mlflow*tar.gz
  cd tests
  NOT_CRAN=true LINTR_COMMENT_BOT=false Rscript ../.travis.R
  cd -

Run linter:

.. code-block:: bash

  Rscript -e 'lintr::lint_package()'


If opening a PR that makes API changes, please regenerate API documentation as described in
`Writing Docs`_ and commit the updated docs to your PR branch.

When developing, you can make Python changes available in R by running (from mlflow/R/mlflow):

.. code-block:: bash

  Rscript -e 'reticulate::conda_install("r-mlflow", "../../../.", pip = TRUE)'

Please also follow the recommendations from the
`Advanced R - Style Guide <http://adv-r.had.co.nz/Style.html>`_ regarding naming and styling.

Python
~~~~~~
Verify that the unit tests & linter pass before submitting a pull request by running:

.. code-block:: bash

    ./lint.sh
    ./travis/run-small-python-tests.sh
    # Optionally, run large tests as well. Travis will run large tests on your pull request once
    # small tests pass. Note: models and model deployment tests are considered "large" tests. If
    # making changes to these components, we recommend running the relevant tests (e.g. tests under
    # tests/keras for changes to Keras model support) locally before submitting a pull request.
    ./travis/run-large-python-tests.sh

Python tests are split into "small" & "large" categories, with new tests falling into the "small"
category by default. Tests that take 10 or more seconds to run should be marked as large tests
via the @pytest.mark.large annotation. Dependencies for small and large tests can be added to
travis/small-requirements.txt and travis/large-requirements.txt, respectively.

We use `pytest <https://docs.pytest.org/en/latest/contents.html>`_ to run Python tests.
You can run tests for one or more test directories or files via
``pytest [--large] [file_or_dir] ... [file_or_dir]``, where specifying ``--large`` tells pytest to
run tests annotated with @pytest.mark.large. For example, to run all pyfunc tests
(including large tests), you can run:

.. code-block:: bash

    pytest tests/pyfunc --large

Note: Certain model tests are not well-isolated (can result in OOMs when run in the same Python
process), so simply invoking ``pytest`` or ``pytest tests`` may not work. If you'd like to
run multiple model tests, we recommend doing so via separate ``pytest`` invocations, e.g.
``pytest --verbose tests/sklearn --large && pytest --verbose tests/tensorflow --large``

Note also that some tests do not run as part of PR builds on Travis. In particular, PR builds
exclude:

  - Tests marked with @pytest.mark.requires_ssh. These tests require that passwordless SSH access to
    localhost be enabled, and can be run via ``pytest --requires-ssh``.
  - Tests marked with @pytest.mark.release. These tests can be run via ``pytest --release``.

In addition, the tests in ``tests/examples`` are run as part of a nightly build on Travis and will
not run on Travis jobs triggered by push requests. If your PR changes anything tested by the tests
or the tests themselves, Travis will detect this and run the nightly tests automatically with the
regular build.

If you need to retrigger Travis tests on a PR, you can push an empty commit to your branch. To create
an empty commit, you can use the ``--allow-empty` option, e.g.
``git commit --allow-empty -m "Trigger rebuild"``. Note that this will retrigger an entire rebuild -
it is currently not possible to retrigger individual tests.

If opening a PR that changes or adds new APIs, please update or add Python documentation as
described in `Writing Docs`_ and commit the docs to your PR branch.

Writing Python Tests
~~~~~~~~~~~~~~~~~~~~
If your PR includes code that isn't currently covered by our tests (e.g. adding a new flavor, adding
autolog support to a flavor, etc.), you should write tests that cover your new code. MLflow currently
uses ``pytest==3.2.1`` for testing. Your tests should be added to the relevant file under ``tests``, or
if there is no appropriate file, in a new file prefixed with ``test_`` so that ``pytest`` includes that
file for testing.

If your tests require usage of a tracking URI, the
`pytest fixture <https://docs.pytest.org/en/3.2.1/fixture.html>`_
`tracking_uri_mock <https://github.com/mlflow/mlflow/blob/master/tests/conftest.py#L74>`_ is automatically set up
for every tests. It sets up a mock tracking URI that will set itself up before your test runs and tear itself down after.
If you want to deactivate the mock for your test, mark the test with `@pytest.mark.notrackingurimock` operator.

Adding New Model Flavor Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are adding new framework flavor support, you'll need to modify ``pytest`` and Travis configurations so tests for your code can run properly. Generally, the files you'll have to edit are:

1. ``.travis.yml``: exclude your tests in the Windows bash script
2. ``travis/run-small-python-tests.sh``: add your tests to the list of ignored framework tests
3. ``travis/run-large-python-tests.sh``:

  a. Add your tests to the ignore list, where the other frameworks are ignored
  b. Add a pytest command for your tests along with the other framework tests (as a separate command to avoid OOM issues)

4. ``travis/large-requirements.txt``: add your framework and version to the list of requirements

You can see an example flavor PR `here <https://github.com/mlflow/mlflow/pull/2136/files>`_.


Building Protobuf Files
~~~~~~~~~~~~~~~~~~~~~~~
To build protobuf files, simply run ``generate-protos.sh``. The required ``protoc`` version is ``3.6.0``.
You can find the URL of a system-appropriate installation of ``protoc`` at
https://github.com/protocolbuffers/protobuf/releases/tag/v3.6.0, e.g.
https://github.com/protocolbuffers/protobuf/releases/download/v3.6.0/protoc-3.6.0-osx-x86_64.zip if
you're on 64-bit Mac OSX.

Then, run the following to install ``protoc``:

.. code-block:: bash

    # Update PROTOC_ZIP if on a platform other than 64-bit Mac OSX
    PROTOC_ZIP=protoc-3.6.0-osx-x86_64.zip
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.6.0/$PROTOC_ZIP
    sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
    sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
    rm -f $PROTOC_ZIP

Verify that .proto files and autogenerated code are in sync by running ``./test-generate-protos.sh.``


Database Schema Changes
~~~~~~~~~~~~~~~~~~~~~~~
MLflow's Tracking component supports storing experiment and run data in a SQL backend. To
make changes to the tracking database schema, run the following from your
checkout of MLflow:

.. code-block:: bash

    # starting at the root of the project
    $ pwd
    ~/mlflow
    $ cd mlflow
    # MLflow relies on Alembic (https://alembic.sqlalchemy.org) for schema migrations.
    $ alembic -c mlflow/store/db_migrations/alembic.ini revision -m "add new field to db"
      Generating ~/mlflow/mlflow/store/db_migrations/versions/b446d3984cfa_add_new_field_to_db.py


These commands generate a new migration script (e.g. at
``~/mlflow/mlflow/alembic/versions/12341123_add_new_field_to_db.py``) that you should then edit to add
migration logic.


Launching the Development UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We recommend `Running the Javascript Dev Server`_ - otherwise, the tracking frontend will request
files in the ``mlflow/server/js/build`` directory, which is not checked into Git.
Alternatively, you can generate the necessary files in ``mlflow/server/js/build`` as described in
`Building a Distributable Artifact`_.


Running the Javascript Dev Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`Install Node Modules`_, then run the following:

In one shell:

.. code-block:: bash

   mlflow ui

In another shell:

.. code-block:: bash

   cd mlflow/server/js
   npm start

The MLflow Tracking UI will show runs logged in ``./mlruns`` at `<http://localhost:3000>`_.

Building a Distributable Artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`Install Node Modules`_, then run the following:

Generate JS files in ``mlflow/server/js/build``:

.. code-block:: bash

   cd mlflow/server/js
   npm run build

Build a pip-installable wheel in ``dist/``:

.. code-block:: bash

   cd -
   python setup.py bdist_wheel


Writing Docs
~~~~~~~~~~~~
First, install dependencies for building docs as described in `Prerequisites`_.

To generate a live preview of Python & other rst documentation, run the following snippet. Note
that R & Java API docs must be regenerated separately after each change and are not live-updated;
see subsequent sections for instructions on generating R and Java docs.

.. code-block:: bash

   cd docs
   make livehtml


Generate R API rst doc files via:

.. code-block:: bash

  cd docs
  make rdocs

Generate Java API rst doc files via:

.. code-block:: bash

  cd docs
  make javadocs


Generate API docs for all languages via:

.. code-block:: bash

  cd docs
  make html


If changing existing Python APIs or adding new APIs under existing modules, ensure that references
to the modified APIs are updated in existing docs under ``docs/source``. Note that the Python doc
generation process will automatically produce updated API docs, but you should still audit for
usages of the modified APIs in guides and examples.

If adding a new public Python module, create a corresponding doc file for the module under
``docs/source/python_api`` - `see here <https://github.com/mlflow/mlflow/blob/v0.9.1/docs/source/python_api/mlflow.tracking.rst#mlflowtracking>`_
for an example.
