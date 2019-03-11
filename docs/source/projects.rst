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
placing files in this directory (for example, a ``conda.yaml`` file is treated as a
`Conda <https://conda.io/docs>`_ environment), but you can describe your project in more detail by
adding a ``MLproject`` file, which is a `YAML <https://learnxinyminutes.com/docs/yaml/>`_ formatted
text file. Each project can specify several properties:

Name
    A human-readable name for the project.

Entry Points
    Commands that can be run within the project, and information about their
    parameters. Most projects contain at least one entry point that you want other users to
    call. Some projects can also contain more than one entry point: for example, you might have a
    single Git repository containing multiple featurization algorithms. You can also call
    any ``.py`` or ``.sh`` file in the project as an entry point. If you list your entry points in
    a ``MLproject`` file, however, you can also specify *parameters* for them, including data
    types and default values.

Environment
    The software environment that should be used to execute project entry points. This includes all
    library dependencies required by the project code. See :ref:`project-environments` for more
    information about the software environments supported by MLflow Projects, including
    :ref:`Conda environments <project-conda-environments>` and 
    :ref:`Docker containers <project-docker-container-environments>`.

You can run any project from a Git URI or from a local directory using the ``mlflow run``
command-line tool, or the :py:func:`mlflow.projects.run` Python API. These APIs also allow submitting the
project for remote execution on `Databricks <https://databricks.com>`_.

.. important::

    By default, MLflow uses a new, temporary working directory for Git projects.
    This means that you should generally pass any file arguments to MLflow
    project using absolute, not relative, paths. If your project declares its parameters, MLflow
    automatically makes paths absolute for parameters of type ``path``.

Specifying Projects
-------------------

By default, any Git repository or local directory can be treated as an MLflow project; you can
invoke any bash or Python script contained in the directory as a project entry point. The 
:ref:`project-directories` section describes how MLflow interprets directories as projects.

To provide additional control over a project's attributes, you can also include an :ref:`MLProject
file <mlproject-file>` in your project's repository or directory.

Finally, MLflow projects allow you to specify the software :ref:`environment <project-environments>`
that is used to execute project entry points.

.. _project-environments:

Project Environments
^^^^^^^^^^^^^^^^^^^^
MLflow currently supports the following project environments: Conda environment, Docker container, and system environment.

.. _project-conda-environments:

Conda environment
  `Conda <https://conda.io/docs>`_ environments support 
  both Python packages and native libraries (e.g, CuDNN or Intel MKL). When an MLflow Project 
  specifies a Conda environment, it is activated before project code is run.

  By default, MLflow uses the system path to find and run the ``conda`` binary. You can use a 
  different Conda installation by setting the ``MLFLOW_CONDA_HOME`` environment variable; in this 
  case, MLflow attempts to run the binary at ``$MLFLOW_CONDA_HOME/bin/conda``.

  You can specify a Conda environment for your MLflow project by including a ``conda.yaml``
  file in the root of the project directory, or by including a ``conda_env`` entry in your
  ``MLProject`` file. For more information, see the :ref:`project-directories` and 
  :ref:`MLProject File <mlproject-file>` sections.

.. _project-docker-container-environments:

Docker container
  `Docker containers <https://www.docker.com/resources/what-container>`_ allow you to 
  capture non-Python dependencies such as Java libraries. When you run an MLflow Project that 
  specifies a Docker image, MLflow runs the image, mounts the project directory in the resulting 
  container at ``/mlflow/projects/code``, and invokes the project entry point in the container. 
 
  Environment variables, such as ``MLFLOW_TRACKING_URI``, are propagated inside the Docker container 
  during project execution. Additionally, :ref:`runs <concepts>` and 
  :ref:`experiments <organizing-runs-in-experiments>` created by the project are saved to the 
  tracking server specified by your :ref:`tracking URI <where-runs-are-recorded>`. When running 
  against a local tracking URI, MLflow mounts the host system's tracking directory
  (e.g., a local ``mlruns`` directory) inside the container so that metrics, parameters, and 
  artifacts logged during project execution are accessible afterwards.

  See `Dockerized Model Training with MLflow <https://github.com/mlflow/mlflow/tree/master/examples/docker>`_ 
  for an example of an MLflow project with a Docker environment.

  .. important::

    You can specify Docker container environments only using an 
    :ref:`MLProject file <mlproject-file>`.
    
System environment
  All of the project's dependencies must be installed on your system prior to project execution. 
  The system environment is supplied at runtime. It is not part of the MLflow Project's
  directory contents or ``MLProject`` file. For information about using the current system
  environment when running a project, see the ``Environment`` parameter description in the 
  :ref:`running-projects` section. 

.. _project-directories:

Project Directories
^^^^^^^^^^^^^^^^^^^

When running an MLflow Project directory or repository that does *not* contain an ``MLProject`` 
file, MLflow uses the following conventions to determine the project's attributes:

* The project's name is the name of the directory.

* The `Conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-file-manually>`_
  is specified in ``conda.yaml``, if present. If no ``conda.yaml`` file is present, MLflow
  uses a Conda environment containing only Python (specifically, the latest Python available to
  Conda) when running the project.

* Any ``.py`` and ``.sh`` file in the project can be an entry point. MLflow uses Python
  to execute entry points with the ``.py`` extension, and it uses bash to execute entry points with
  the ``.sh`` extension. For more information about specifying project entrypoints at runtime,
  see :ref:`running-projects`.

* By default, entry points do not have any parameters when an ``MLProject`` file is not included.
  Parameters can be supplied at runtime via the ``mlflow run`` CLI or the 
  :py:func:`mlflow.projects.run` Python API. Runtime parameters are passed to the entry point on the 
  command line using ``--key value`` syntax. For more information about running projects and
  with runtime parameters, see :ref:`running-projects`. 

.. _mlproject-file: 

The MLProject File
^^^^^^^^^^^^^^^^^^

You can get more control over an MLflow Project by adding an ``MLproject`` file, which is a text
file in YAML syntax, to the project's root directory. The following is an example of an 
``MLProject`` file: 

.. code-block:: yaml

    name: My Project

    conda_env: my_env.yaml
    # Can have a docker_env instead of a conda_env, e.g.
    # docker_env:
    #    image:  mlflow-docker-example

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

As you can see, the file can specify a name and a Conda or docker environment, as well as more
detailed information about each entry point. Specifically, each entry point defines a *command* to
run and *parameters* to pass to the command (including data types). We describe these two pieces 
next.

Command Syntax
~~~~~~~~~~~~~~

When specifying an entry point in an ``MLproject`` file, the command can be any string in Python
`format string syntax <https://docs.python.org/2/library/string.html#formatstrings>`_.
All of the parameters declared in the entry point's ``parameters`` field are passed into this
string for substitution. If you call the project with additional parameters *not* listed in the
``parameters`` field, MLflow passes them using ``--key value`` syntax, so you can use the
``MLproject`` file to declare types and defaults for just a subset of your parameters.

Before substituting parameters in the command, MLflow escapes them using the Python
`shlex.quote <https://docs.python.org/3/library/shlex.html#shlex.quote>`_ function, so you don't 
need to worry about adding quotes inside your command field.

.. _project_parameters:

Specifying Parameters
~~~~~~~~~~~~~~~~~~~~~

MLflow allows specifying a data type and default value for each parameter. You can specify just the
data type by writing:

.. code-block:: yaml

    parameter_name: data_type

in your YAML file, or add a default value as well using one of the following syntaxes (which are
equivalent in YAML):

.. code-block:: yaml

    parameter_name: {type: data_type, default: value}  # Short syntax

    parameter_name:     # Long syntax
      type: data_type
      default: value

MLflow supports four parameter types, some of which it treats specially (for example, downloading
data to local files). Any undeclared parameters are treated as ``string``. The parameter types are:

string
    A text string.

float
    A real number. MLflow validates that the parameter is a number.

path
    A path on the local file system. MLflow converts any relative ``path`` parameters to absolute 
    paths. MLflow also downloads any paths passed as distributed storage URIs 
    (``s3://`` and ``dbfs://``) to local files. Use this type for programs that can only read local 
    files.

uri
    A URI for data either in a local or distributed storage system. MLflow converts
    relative paths to absolute paths, as in the ``path`` type. Use this type for programs
    that know how to read from distributed storage (e.g., programs that use Spark).

.. _running-projects:

Running Projects
----------------

MLflow provides two simple ways to run projects: the ``mlflow run`` :ref:`command-line tool <cli>`, or
the :py:func:`mlflow.projects.run` Python API. Both tools take the following parameters:

Project URI
    A directory on the local file system or a Git repository path,
    specified as a URI of the form ``https://<repo>`` (to use HTTPS) or ``user@host:path``
    (to use Git over SSH). To run against an MLproject file located in a subdirectory of the project, 
    add a '#' to the end of the URI argument, followed by the relative path from the project's root directory
    to the subdirectory containing the desired project.

Project Version
    For Git-based projects, the commit hash or branch name in the Git repository.

Entry Point
    The name of the entry point, which defaults to ``main``. You can use any
    entry point named in the ``MLproject`` file, or any ``.py`` or ``.sh`` file in the project,
    given as a path from the project root (for example, ``src/test.py``).

Parameters
    Key-value parameters. Any parameters with
    :ref:`declared types <project_parameters>` are validated and transformed if needed.

Deployment Mode
    Both the command-line and API let you :ref:`launch projects remotely <databricks_execution>` on
    a `Databricks <https://databricks.com>`_ environment if you have a Databricks account. This
    includes setting cluster parameters such as a VM type. Of course, you can also run projects on
    any other computing infrastructure of your choice using the local version of the ``mlflow run``
    command (for example, submit a script that does ``mlflow run`` to a standard job queueing system).

Environment
    By default, MLflow Projects are run in the environment specified by the project directory
    or the ``MLProject`` file (see :ref:`Specifying Project Environments <project-environments>`).
    You can ignore a project's specified environment and run the project in the current
    system environment by supplying the ``--no-conda`` flag.

For example, the tutorial creates and publishes an MLflow project that trains a linear model. The
project is also published on GitHub at https://github.com/mlflow/mlflow-example. To run
this project:

.. code-block:: bash

    mlflow run git@github.com:mlflow/mlflow-example.git -P alpha=0.5

There are also additional options for disabling the creation of a Conda environment, which can be
useful if you quickly want to test a project in your existing shell environment.

.. _databricks_execution:

Remote Execution on Databricks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support for running projects remotely on Databricks is in beta preview and requires a Databricks account. 
To receive future updates about the feature, `sign up here <http://databricks.com/mlflow>`_.

.. important::

  Remote execution for MLflow projects with Docker environments is *not* currently supported.

Launching a Remote Execution on Databricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use this feature, you need to have a Databricks account (Community Edition is not yet supported)
and you must have set up the `Databricks CLI <https://github.com/databricks/databricks-cli>`_. Find more detailed instructions in the Databricks docs (`Azure Databricks <https://docs.databricks.com/applications/mlflow/index.html>`_, `Databricks on AWS <https://docs.databricks.com/applications/mlflow/index.html>`_). A brief overview of how to use the feature is as follows:

First, create a JSON file containing the 
`cluster specification <https://docs.databricks.com/api/latest/jobs.html#jobsclusterspecnewcluster>`_
for your run. Then, run your project using the command

.. code-block:: bash

  mlflow run <uri> -m databricks --cluster-spec <json-cluster-spec>

where ``<uri>`` is a Git repository URI or a folder. You can pass Git credentials with the
``git-username`` and ``git-password`` arguments or using the ``MLFLOW_GIT_USERNAME`` and
``MLFLOW_GIT_PASSWORD`` environment variables.

Iterating Quickly
-----------------

If you want to rapidly develop a project, we recommend creating an ``MLproject`` file with your
main program specified as the ``main`` entry point, and running it with ``mlflow run .``.
To avoid repeatedly writing them you can add default parameters in your ``MLproject`` file.

Building Multistep Workflows
-----------------------------

The :py:func:`mlflow.projects.run` API, combined with :py:mod:`mlflow.tracking`, makes it possible to build
multi-step workflows with separate projects (or entry points in the same project) as the individual
steps. Each call to :py:func:`mlflow.projects.run` returns a run object, that you can use with
:py:mod:`mlflow.tracking` to determine when the run has ended and get its output artifacts. These artifacts
can then be passed into another step that takes ``path`` or ``uri`` parameters. You can coordinate
all of the workflow in a single Python program that looks at the results of each step and decides
what to submit next using custom code. Some example uses cases for multi-step workflows include:

Modularizing Your Data Science Code
  Different users can publish reusable steps for data featurization, training, validation, and so on, that other users or team can run in their workflows. Because MLflow supports Git versioning, another team can lock their workflow to a specific version of a project, or upgrade to a new one on their own schedule.

Hyperparameter Tuning
  Using :py:func:`mlflow.projects.run` you can launch multiple runs in parallel either on the local machine or on a cloud platform like Databricks. Your driver program can then inspect the metrics from each run in real time to cancel runs, launch new ones, or select the best performing run on a target metric.

Cross-validation
  Sometimes you want to run the same training code on different random splits of training and validation data. With MLflow Projects, you can package the project in a way that allows this, for example, by taking a random seed for the train/validation split as a parameter, or by calling another project first that can split the input data.

For an example of how to construct such a multistep workflow, see the MLflow `Multistep Workflow Example project <https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow>`_.
