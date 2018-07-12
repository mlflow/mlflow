.. _concepts:

Concepts
========

MLflow is organized into three components: :ref:`Tracking<tracking>`, :ref:`Projects<projects>`, and
:ref:`Models<models>`. You can use each of these components on their own---for example, maybe you
want to export models in MLflow's model format without using Tracking or Projects---but they are
also designed to work well together.

MLflow's core philosophy is to put as few constraints as possible on your workflow: it is designed
to work with any machine learning library, determine most things about your code by convention, and
require minimal changes to integrate into an existing codebase. At the same time, MLflow aims to
take any codebase written in its format and make it reproducible and reusable by multiple data
scientists. On this page, we describe a typical ML workflow and where MLflow fits in.


The Machine Learning Workflow
-----------------------------

Machine learning requires experimenting with a wide range of datasets, data preparation steps, and
algorithms to build a model that maximizes some target metric. Once you have built a model, you also
need to deploy it to a production system, monitor its performance, and continuously retrain it on
new data and compare with alternative models.

Being productive with machine learning can therefore be challenging for several reasons:

* **It's difficult to keep track of experiments.** When you are just working with files on your
  laptop, or with an interactive notebook, how do you tell which data, code and parameters went into
  getting a particular result?

* **It's difficult to reproduce code.** Even if you have meticulously tracked the code versions and
  parameters, you need to capture the whole environment (for example, library dependencies) to get the
  same result again. This is especially challenging if you want another data scientist to use your
  code, or if you want to run the same code at scale on another platform (for example, in the cloud).

* **There's no standard way to package and deploy models.** Every data science team comes up with
  its own approach for each ML library that it uses, and the link between a model and the code and
  parameters that produced it is often lost.

Moreover, although individual ML libraries provide solutions to some of these problems (for example, model
serving), to get the best result you usually want to try *multiple ML libraries*. MLflow lets you
train, reuse, and deploy models with any library and package them into reproducible steps that other
data scientists can use as a "black box," without even having to know which library you are using.

MLflow Components
-------------------

MLflow provides three components to help manage the ML workflow:

**MLflow Tracking** is an API and UI for logging parameters, code versions, metrics, and output files
when running your machine learning code and for later visualizing the results. You can use MLflow Tracking in
any environment (for example, a standalone script or a notebook) to log results to local files or to a
server, then compare multiple runs. Teams can also use it to compare results from different users.

**MLflow Projects** are a standard format for packaging reusable data science code. Each project
is simply a directory with code or a Git repository, and uses a descriptor file or simply
convention to specify its dependencies and how to run the code. For example, projects can contain
a ``conda.yaml`` file for specifying a Python `Conda <https://conda.io/docs/>`_ environment.
When you use the MLflow Tracking API in a Project, MLflow automatically remembers the project
version executed (for example, Git commit) and any parameters. You can easily run existing MLflow
Projects from GitHub or your own Git repository, and chain them into multi-step workflows.

**MLflow Models** offer a convention for packaging machine learning models in multiple flavors, and
a variety of tools to help you deploy them. Each Model is saved as a directory containing arbitrary
files and a descriptor file that lists several "flavors" the model can be used in. For example, a
TensorFlow model can be loaded as a TensorFlow DAG, or as a Python function to apply to input data.
MLflow provides tools to deploy many common model types to diverse platforms: for example, any model
supporting the "Python function" flavor can be deployed to a Docker-based REST server, to cloud
platforms such as Azure ML and AWS SageMaker, and as a user-defined function in Apache Spark for
batch and streaming inference. If you output MLflow Models using the Tracking API, MLflow will also
automatically remember which Project and run they came from.

..
    TODO: example app and data

Scalability and Big Data
------------------------

Data is the key to obtaining good results in machine learning, so MLflow is designed to scale to
large data sets, large output files (for example, models), and large numbers of experiments. Specifically,
MLflow supports scaling in three dimensions:

* An individual MLflow run can execute on a distributed cluster, for example, using
  `Apache Spark <https://spark.apache.org>`_. You can launch runs on the distributed infrastructure
  of your choice and report results to a Tracking Server to compare them. MLflow includes a
  built-in API to launch runs on `Databricks <https://databricks.com/>`_.

* MLflow supports launching multiple runs in parallel with different parameters, for example, for
  hyperparameter tuning. You can simply use the :ref:`Projects API<projects>` to start multiple
  runs and the :ref:`Tracking API<tracking>` to track them.

* MLflow Projects can take input from, and write output to, distributed storage systems such as
  AWS S3 and `DBFS <https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html>`_.
  MLflow can automatically download such files locally for projects that can only run on local
  files, or give the project a distributed storage URI if it supports that. This means that you
  can write projects that build large datasets, such as featurizing a 100 TB file.

Example Use Cases
-----------------

There are multiple ways you can use MLflow, whether you are a data scientist working alone or part
of a large organization:

**Individual Data Scientists** can use MLflow Tracking to track experiments locally on their
machine, organize code in projects for future reuse, and output models that production engineers can
then deploy using MLflow's deployment tools. MLflow Tracking just reads and writes files to the
local file system by default, so there is no need to deploy a server.

**Data Science Teams** can deploy an MLflow Tracking server to log and compare results across
multiple users working on the same problem. By setting up a convention for naming their parameters
and metrics, they can try different algorithms to tackle the same problem and then run the same
algorithms again on new data to compare models in the future. Moreover, anyone can download and
run another model.

**Large Organizations** can share projects, models, and results using MLflow. Any team can run
another team's code using MLflow Projects, so organizations can package useful training and data
preparation steps that other teams can use, or compare results from many teams on the same task.
Moreover, engineering teams can easily move workflows from R&D to staging to production.

**Production Engineers** can deploy models from diverse ML libraries in the same way, store the
models as files in a management system of their choice, and track which run a model came from.

**Researchers and Open Source Developers** can publish code to GitHub in the MLflow Project format,
making it easy for anyone to run their code using the
``mlflow run github.com/...`` command.

**ML Library Developers** can output models in the MLflow Model format to have them automatically
support deployment using MLflow's built-in tools. In addition, deployment tool developers (for example, a
cloud vendor building a serving platform) can automatically support a large variety of models.
