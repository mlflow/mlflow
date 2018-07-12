.. _projects:

MLflow Projects
===============

An MLflow Project is a format for packaging data science code in a reusable and reproducible way,
based primarily on conventions. In addition, the Projects component includes an API and command-line
tools for running projects, making it possible to chain together projects into workflows.

.. contents:: Table of Contents
  :local:
  :depth: 1

Overview
--------

At the core, MLflow Projects are just a convention for organizing and describing your code to let
other data scientists (or automated tools) run it. Each project is simply a directory of files, or
a Git repository, containing your code. MLflow can run some projects based on a convention for
placing files in this directory (for example, a ``conda.yaml`` file will be treated as a
`Conda <https://conda.io/docs>`_ environment), but you can describe your project in more detail by
adding a ``MLproject`` file, which is a `YAML <https://learnxinyminutes.com/docs/yaml/>`_ formatted
text file. Each project can specify several properties:

Name
    A human-readable name for the project.

Dependencies
    Libraries needed to run the project. MLflow currently uses the
    `Conda <https://conda.io/docs>`_ package manager, which supports both Python packages and native
    libraries (for example, CuDNN or Intel MKL), to specify dependencies.

Entry Points
    Commands that can be executed within the project, and information about their
    parameters. Most projects will contain at least one entry point that you want other users to
    call. Some projects can also contain more than one entry point: for example, you might have a
    single Git repository containing multiple featurization algorithms. You can also call
    any ``.py`` or ``.sh`` file in the project as an entry point. If you list your entry points in
    a ``MLproject`` file, however, you can also specify *parameters* for them, including data
    types and default values.

You can run any project from a Git URI or from a local directory using the ``mlflow run``
command-line tool, or the :py:func:`mlflow.run` Python API. These APIs also allow submitting the
project for remote execution on `Databricks <https://databricks.com>`_.

.. caution::

    By default, MLflow will use a new, temporary working directory for Git projects.
    This means that you should generally pass any file arguments to MLflow
    project using absolute, not relative, paths. If your project declares its parameters, MLflow
    will automatically make paths absolute for parameters of type ``path``.

Specifying Projects
-------------------

By default, any Git repository or local directory is treated as a project, and MLflow uses the
following conventions to determine its parameters:

* The project's name is the name of the directory.
* The `Conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-file-manually>`_
  is specified in ``conda.yaml``, if present.
* Any ``.py`` and ``.sh`` file in the project can be an entry point, with no parameters explicitly
  declared. When you execute such a command with a set of parameters, MLflow will pass each
  parameter on the command line using ``--key value`` syntax.

You can get more control over a project by adding a ``MLproject``, which is simply a text file in
YAML syntax. The MLproject file looks like this:

.. code:: yaml

    name: My Project

    conda_env: my_env.yaml

    entry_points:
      main:
        parameters:
          data_file: path
          regularization: {type: float, default: 0.1}
        command: "python train.py -r {regularization} {data_file}"
      validate:
        parameters:
          data_file: path
        command: "python validate.py {data_file}"

As you can see, the file can specify a name and a different environment file, as well as more
detailed information about each entry point. Specifically, each entry point has a *command* to
run and *parameters* (including data types). We describe these two pieces next.

Command Syntax
^^^^^^^^^^^^^^

When specifying an entry point in an ``MLproject`` file, the command can be any string in Python
`format string syntax <https://docs.python.org/2/library/string.html#formatstrings>`_.
All of the parameters declared in the entry point's ``parameters`` field will be passed into this
string for substitution. If you call the project with additional parameters *not* listed in the
``parameters`` field, MLflow will still pass them using ``--key value`` syntax, so you can use the
``MLproject`` file to declare types and defaults for just a subset of your parameters if you like.

Before substituting parameters in the command, MLflow will escape them using Python's
`shlex.quote <https://docs.python.org/3/library/shlex.html#shlex.quote>`_ function, so you need
not worry about adding quotes inside your command field.

.. _project_parameters:

Specifying Parameters
^^^^^^^^^^^^^^^^^^^^^

MLflow allows specifying a data type and default value for each parameter. You can specify just the
data type by writing:

.. code:: yaml

    parameter_name: data_type

in your YAML file, or add a default value as well using one of the following syntaxes (which are
equivalent in YAML):

.. code:: yaml

    parameter_name: {type: data_type, default: value}  # Short syntax

    parameter_name:     # Long syntax
      type: data_type
      default: value

MLflow supports four parameter types, some of which it treats specially (for example, downloading
data to local files). Any undeclared parameters are treated as ``string``. The parameter types are:

string
    Any text string.

float
    A real number. MLflow validates that the parameter is a number.

path
    A path on the local file system. MLflow will convert any relative paths passed for
    parameters of this type to absolute paths, and will also download any paths passed
    as distributed storage URIs (``s3://`` and ``dbfs://``) to local files. Use this type
    for programs that can only read local files.

uri
    A URI for data either in a local or distributed storage system. MLflow will convert
    any relative paths to absolute paths, as in the ``path`` type. Use this type for programs
    that know how to read from distributed storage (for example using Spark).

Running Projects
----------------

MLflow provides two simple ways to run projects: the ``mlflow run`` :ref:`command-line tool <cli>`, or
the :py:func:`mlflow.run` Python API. Both tools take the following parameters:

Project URI
    Can be either a directory on the local file system or a Git repository path,
    specified as a URI of the form ``https://<repo>`` (to use HTTPS) or ``user@host:path``
    (to use Git over SSH).

Project Version
    Which commit in the Git repository to run, for Git-based projects.

Entry Point
    The name of the entry point to use, which defaults to ``main``. You can use any
    entry point named in the ``MLproject`` file, or any ``.py`` or ``.sh`` file in the project,
    given as a path from the project root (for example, ``src/test.py``).

Parameters
    Key-value parameters. Any parameters with
    :ref:`declared types <project_parameters>` will be validated and transformed if needed.

Deployment Mode
    Both the command-line and API let you :ref:`launch projects remotely <databricks_execution>` on
    a `Databricks <https://databricks.com>`_ environment if you have a Databricks account. This
    includes setting cluster parameters such as a VM type. Of course, you can also run projects on
    any other computing infrastructure of your choice using the local version of the ``mlflow run``
    command (for example, submit a script that does ``mlflow run`` to a standard job queueing system).

For example, in the tutorial we create and publish a MLproject which trains a linear model. The
project is also published on GitHub at https://github.com/databricks/mlflow-example. To execute
this project run

.. code::

    mlflow run git@github.com:databricks/mlflow-example.git -P alpha=0.5

There are also additional options for disabling the creation of a Conda environment, which can be
useful if you quickly want to test a project in your existing shell environment.

.. _databricks_execution:

Remote Execution on Databricks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Support for running projects on Databricks will be released soon -
`sign up here <http://databricks.com/mlflow>`_ to receive updates.


Launching a Run
~~~~~~~~~~~~~~~
First, create a JSON file containing the cluster spec for your run with the attributes
`described here <https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster>`_.
Then, run your project via

``mlflow run <uri> -m databricks --cluster-spec <path>``

``<uri>`` must be a Git repository URI. You can also pass Git credentials via the
``git-username`` and ``git-password`` arguments (or via the ``MLFLOW_GIT_USERNAME`` and
``MLFLOW_GIT_PASSWORD`` environment variables).


Iterating Quickly
-----------------

If you want to rapidly develop a project, we recommend creating an ``MLproject`` file with your
main program specified as the ``main`` entry point, and running it with ``mlflow run .``.
You can even add default parameters in your ``MLproject`` to avoid repeatedly writing them.

Building Multi-Step Workflows
-----------------------------

The :py:func:`mlflow.run` API, combined with :py:mod:`mlflow.tracking`, makes it possible to build
multi-step workflows with separate projects (or entry points in the same project) as the individual
steps. Each call to :py:func:`mlflow.run` returns a run ID, which you can use with
:py:mod:`mlflow.tracking` to determine when the run has ended and get its output artifacts. These artifacts
can then be passed into another step that takes ``path`` or ``uri`` parameters. You can coordinate
all of the workflow in a single Python program that looks at the results of each step and decides
what to submit next using custom code. Some example uses cases for multi-step workflows include:

Modularizing Your Data Science Code
  Different users can publish reusable steps for data featurization, training, validation, and so on, that other users or team can run in their workflows. Because MLflow supports Git versioning, another team can lock their workflow to a specific version of a project, or upgrade to a new one on their own schedule.

Hyperparameter Tuning
  Using :py:func:`mlflow.run` you can launch multiple runs in parallel either on the local machine or on a cloud platform like Databricks. Your driver program can then inspect the metrics from each run in real time to cancel runs, launch new ones, or select the best performing run on a target metric.

Cross-validation
  Sometimes you want to run the same training code on different random splits of training and validation data. With MLflow Projects, you can package the project in a way that allows this, for example, by taking a random seed for the train/validation split as a parameter, or by calling another project first that can split the input data.
