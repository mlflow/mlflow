.. _local_model_deployment:

Deploy MLflow Model as a Local Inference Server
===============================================

MLflow allows you to deploy your model as a locally using just a single command.
This approach is ideal for lightweight applications or for testing your model locally before moving it to a staging or production environment.

If you are new to MLflow model deployment, please read the guide on `MLflow Deployment <index.html>`_ first to understand the basic concepts of MLflow models and deployments.


Deploying Inference Server
--------------------------

Before deploying, you must have an MLflow Model. If you don't have one, you can create a sample scikit-learn model by following the `MLflow Tracking Quickstart <../getting-started/index.html>`_.
Remember to note down the model URI, such as ``runs:/<run_id>/<artifact_path>`` (or ``models:/<model_name>/<model_version>`` if you registered the model in the `MLflow Model Registry <../model-registry.html>`_).

Once you have the model ready, deploying to a local server is straightforward. Use the `mlflow models serve <../cli.html#mlflow-models-serve>`_ command for a one-step deployment.
This command starts a local server that listens on the specified port and serves your model.

.. tabs::

    .. code-tab:: bash

       mlflow models serve -m runs:/<run_id>/model -p 5000

    .. code-tab:: python

       import mlflow

       model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
       model.serve(port=5000)


You can then send a test request to the server as follows:

.. code-block:: bash

    curl http://127.0.0.1:5000/invocations -H "Content-Type:application/json;"  --data '{"inputs": [[1, 2], [3, 4], [5, 6]]}'


Several command line options are available to customize the server's behavior. For instance, the ``--env-manager`` option allows you to
choose a specific environment manager, like Anaconda, to create the virtual environment. The `mlflow models` module also provides
additional useful commands, such as building a Docker image or generating a Dockerfile. For comprehensive details, please refer 
to the `MLflow CLI Reference <../cli.html#mlflow-models>`_.


.. _local-inference-server-spec:

Inference Server Specification
------------------------------

Endpoints
~~~~~~~~~
The inference server provides 4 endpoints:

* ``/invocations``: An inference endpoint that accepts POST requests with input data and returns predictions.

* ``/ping``: Used for health checks.

* ``/health``: Same as /ping

* ``/version``: Returns the MLflow version.

Accepted Input Formats
~~~~~~~~~~~~~~~~~~~~~~
The ``/invocations`` endpoint accepts CSV or JSON inputs. The input format must be specified in the
``Content-Type`` header as either ``application/json`` or ``application/csv``.

CSV Input
*********

CSV input must be a valid pandas.DataFrame CSV representation. For example:

``curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/csv' --data '1,2,3,4'``

JSON Input
**********

JSON input must be a dictionary with exactly one of the following fields, further specifying the input data's type and encoding:

.. list-table::
    :widths: 20 40 40
    :header-rows: 1
    :class: wrap-table

    * - Field
      - Description
      - Example
    * - ``dataframe_split``
      - Pandas DataFrames in the ``split`` orientation.
      - ``{"dataframe_split": pandas_df.to_dict(orient='split')}``
    * - ``dataframe_records``
      - Pandas DataFrame in the ``records`` orientation. **We do not recommend using this format because it is not guaranteed to preserve column ordering.**
      - ``{"dataframe_records": pandas_df.to_dict(orient='records')}``
    * - ``instances``
      - Tensor input formatted as described in `TF Serving's API docs <https://www.tensorflow.org/tfx/serving/api_rest#request_format_2>`_ where the provided inputs will be cast to Numpy arrays.
      - ``{"instances": [1.0, 2.0, 5.0]}``
    * - ``inputs``
      - Same as ``instances`` but with a different key.
      - ``{"inputs": [["I", "have", "a",  "pen"], ["I" "have", "an", "apple"]]}``

The JSON input can also include an optional ``params`` field for passing additional parameters.
Valid parameter types are ``Union[DataType, List[DataType], None]``, where DataType is
:py:class:`MLflow data types <mlflow.types.DataType>`. To pass parameters,
a valid :ref:`Model Signature <model-signature>` with ``params`` must be defined.

.. code-block:: bash

    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
        "inputs": {"question": ["What color is it?"],
                   "context": ["Some people said it was green but I know that it is pink."]},
        "params": {"max_answer_len": 10}
    }'

.. note:: Since JSON discards type information, MLflow will cast the JSON input to the input type specified
    in the model's schema if available. If your model is sensitive to input types, it is recommended that
    a schema is provided for the model to ensure that type mismatch errors do not occur at inference time.
    In particular, Deep Learning models are typically strict about input types and will need a model schema in order
    for the model to score correctly. For complex data types, see :ref:`encoding-complex-data` below.

.. _encoding-complex-data:

Encoding complex data
*********************

Complex data types, such as dates or binary, do not have a native JSON representation. If you include a model
signature, MLflow can automatically decode supported data types from JSON. The following data type conversions
are supported:

* binary: data is expected to be base64 encoded, MLflow will automatically base64 decode.

* datetime: data is expected to be encoded as a string according to
  `ISO 8601 specification <https://www.iso.org/iso-8601-date-and-time-format.html>`_.
  MLflow will parse this into the appropriate datetime representation on the given platform.

Example requests:

.. code-block:: bash

    # record-oriented DataFrame input with binary column "b"
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '[
        {"a": 0, "b": "dGVzdCBiaW5hcnkgZGF0YSAw"},
        {"a": 1, "b": "dGVzdCBiaW5hcnkgZGF0YSAx"},
        {"a": 2, "b": "dGVzdCBiaW5hcnkgZGF0YSAy"}
    ]'

    # record-oriented DataFrame input with datetime column "b"
    curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '[
        {"a": 0, "b": "2020-01-01T00:00:00Z"},
        {"a": 1, "b": "2020-02-01T12:34:56Z"},
        {"a": 2, "b": "2021-03-01T00:00:00Z"}
    ]'


.. _serving_frameworks:

Serving Frameworks
------------------
By default, MLflow uses `Flask <https://flask.palletsprojects.com/en/1.1.x/>`_, a lightweight WSGI web application framework for Python, to serve the
inference endpoint. Alternatively, you can use `Seldon's MLServer <https://mlserver.readthedocs.io/en/latest/>`_, which is used as the core Python
inference server in Kubernetes-native frameworks like `Seldon Core <https://docs.seldon.io/projects/seldon-core/en/latest/>`_ and
`KServe (formerly known as KFServing) <https://kserve.github.io/website/>`_.

.. |flask-logo| raw:: html

        <div>
            <img src="../_static/images/logos/flask-logo.png" width="60%" style="display: block; margin: auto;">
        </div>

.. |mlserver-logo| raw:: html

            <div>
                <img src="../_static/images/logos/seldon-mlserver-logo.png" width="70%" style="display: block; margin: auto;">
            </div>


.. list-table::
    :widths: 20 40 40
    :header-rows: 1
    :class: wrap-table

    * -
      - |flask-logo|
      - |mlserver-logo|
    * - **Use Case**
      - General lightweight purpose including local testing.
      - Kubernetes cluster deployment.
    * - **Set Up**
      - Flask is installed by default with MLflow.
      - Needs to be installed separately.
    * - **Maturity**
      - Established and stable.
      - Relatively less mature. 
    * - **Performance**
      - Suitable for lightweight applications but not optimized for high performance.
      - Designed for high-performance ML workloads, often delivering better throughput and efficiency.
    * - **Scalability**
      - Not inherently scalable, but can be augmented with other tools.
      - Achieves high scalability with Kubernetes-native frameworks such as Seldon Core and KServe.


MLServer exposes the same scoring API through the ``/invocations`` endpoint.
To deploy with MLServer, first install additional dependencies with ``pip install mlflow[extras]``,
then execute the deployment command with the ``--enable-mlserver`` option. For example,

.. tabs::

    .. code-tab:: bash

       mlflow models serve -m runs:/<run_id>/model -p 5000 --enable-mlserver

    .. code-tab:: python

       import mlflow

       model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
       model.serve(port=5000, enable_mlserver=True)

To read more about the integration between MLflow and MLServer, please check the `end-to-end example <https://mlserver.readthedocs.io/en/latest/examples/mlflow/README.html>`_ in the MLServer documentation.
You can also find guides to deploy MLflow models to a Kubernetes cluster using MLServer in `Deploying a model to Kubernetes <deploy-model-to-kubernetes/index.html>`_.

Running Batch Inference
-----------------------
Instead of running an online inference endpoint, you can execute a single batch inference job on local files using
the `mlflow models predict <../cli.html#mlflow-models-predict>`_ command. The following command runs the model
prediction on ``input.csv`` and outputs the results to ``output.csv``.

.. tabs::

    .. code-tab:: bash

       mlflow models predict -m runs:/<run_id>/model -i input.csv -o output.csv

    .. code-tab:: python

       import mlflow

       model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
       predictions = model.predict(pd.read_csv("input.csv"))
       predictions.to_csv("output.csv")


Troubleshooting
---------------
