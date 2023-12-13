Develop ML model with MLflow and deploy to Kubernetes
=====================================================

.. note::

  This tutorial assumes that you have a Kubernetes cluster, but you can also complete this tutorial with a local machine,
  by using local cluster emulation tool such as `Kind <https://kind.sigs.k8s.io/docs/user/quick-start>`_ or `Minikube <https://minikube.sigs.k8s.io/docs/start/>`_.


This guide showcases how you can use MLflow end-to-end to:

- Train a linear regression model with `MLflow Tracking <../../../tracking.html>`_.
- Run hyper parameter tuning to find the best model.
- Package model weight and dependencies as `MLflow Model <../../../models.html>`_.
- Test the model serving locally with `mlserver <https://mlserver.readthedocs.io/en/latest/>`_ using `mlflow models serve <../../../cli.html#mlflow-models-serve>`_ command.
- Deploy the model to Kubernetes cluster using `KServe <https://kserve.github.io/website/>`_ with MLflow.

We will cover end-to-end model developemnt process including model training and testing, however,
if you already have your model and just want to know how to deploy it to Kubernetes, you can skip to :ref:`Step 6 - Test Model Serving Locally <step-6-test-model-serving-locally>`


Introduction: Scalable Model Serving with KServe and MLServer
-------------------------------------------------------------

MLflow provides an easy-to-use interface to deploy modeles as an Flask-based inference server. You can deploy the same inference
server to Kubernetes cluster by containerizing it via ``mlflow models build-docker`` command. However, this approach is not scalable
and might not be suitable for production use cases. Flask is not designed for high performance and managing multiple instances of
inference servers manually is not easy.

Fortunately, MLflow provides a solution for this. MLflow supports alternative inference engine called `MLServer <https://mlserver.readthedocs.io/en/latest/>`_,
which enables one-step deployment to popular serverless model serving frameworks on Kubernetes, such as `KServe <https://kserve.github.io/website/>`_, and 
`Seldon Core <https://docs.seldon.io/projects/seldon-core/en/latest/>`_.


What is KServe?
~~~~~~~~~~~~~~~

`KServe <https://kserve.github.io/website/>`_, formally known as KFServing, provides performant, scalable, and high abstraction interfaces for common machine learning (ML) frameworks like Tensorflow, XGBoost, scikit-learn, Pytorch.
It provides advanced features that helps operating large-scale machine learning systems, such as **autoscaling**, **canary rollout**, **A/B testing**, **monitoring**,
**explability**, and more, taking advantage of Kubernetes ecosystem including `KNative <https://knative.dev/>`_ and `Istio <https://istio.io/>`_.

Benefit of using MLflow with KServe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While KServe provides highly scalable and production-ready model serving framework, it might require some effort to deploy your model there.
MLflow provides a simple way to deploy models to Kubernetes cluster with KServe and MLServer. Also, it offers seamless **End-to-end model management**,
as a single place to manage the entire ML lifecycle, including `experiment tracking <../../../tracking.html>`_, `model packaging <../../../models.html>`_,
`versioning <../../../model-registry.html>`_, `evaluation <../../model-evaluation.html>`_, and `deployment <../../deployments.html>`_.


Step 1: Installing mlflow and additional dependencies
-----------------------------------------------------
Install mlflow  to your local machine using the following command:

.. code-block:: bash

  pip install mlflow[extras]

``[extra]`` will install additional dependencies required for this tutorial including `mlserver <https://mlserver.readthedocs.io/en/latest/>`_ and
`scikit-learn <https://scikit-learn.org/stable/>`_ (not required for deployment, just for this tutorial)

You can check if mlflow is installed correctly by running:

.. code-block:: bash

  mlflow --version

Step 2: Setting up a Kubernetes cluster
---------------------------------------

.. tabs::

  .. tab:: Kubernetes Cluster

        Assuming you already have a Kubernetes cluster, you can install KServe following `the official instructions <https://github.com/kserve/kserve#hammer_and_wrench-installation>`_.

  .. tab:: Local Machine Emulation

        You can follow `KServe QuickStart <https://kserve.github.io/website/latest/get_started/>`_ to set up a local cluster with `Kind <https://kind.sigs.k8s.io/docs/user/quick-start>`_
        and install KServe on it.

Now that you have a Kubernetes cluster as a deployment target, let's move on to creating the MLflow Model to deploy.

Step 3: Training the Model
--------------------------

In this tutorial, we will train a model to predict the quality of wine based on quantitative features.
We use the `wine quality dataset <http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>`_ from the UCI Machine Learning Repository,
and `ElasticNet <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_, a simple linear regression model.

First, let's train a single model with default hyperparameters. Execute the following code in a notebook or as a Python script.

.. note::

  For the sake of convenience, we use the `mlflow.sklearn.autolog() <../../../api/python/mlflow.sklearn.html#mlflow.sklearn.autolog>`_ function.
  This function let MLflow to automatically log the appropriate set of model parameters and metrics. To learn more about the autologging feature
  or how to log manually instead, see the `MLflow Tracking documentation <../../../tracking.html>`_.

.. code-block:: python

  import mlflow
  import mlflow.sklearn

  from sklearn import datasets
  from sklearn.linear_model import ElasticNet
  from sklearn.model_selection import train_test_split


  def eval_metrics(pred, actual):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2


  # Set th experiment name
  mlflow.set_experiment("wine-quality")

  # Enable auto-logging to MLflow
  mlflow.sklearn.autolog()

  # Load wine quality dataset
  X, y = datasets.load_wine(return_X_y=True)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  # Start a run and train a model
  with mlflow.start_run(run_name="default-params"):
      lr = ElasticNet()
      lr.fit(X_train, y_train)

      y_pred = lr.predict(X_test)
      metrics = eval_metrics(y_pred, y_test)


Now you have trained a model, let's check if the parameter and metrics are logged correctly via MLflow UI.
you can start MLflow UI by running the following command in your terminal:

.. code-block:: bash

  mlflow ui --port 5000

Then you can access the UI at http://localhost:5000

.. figure:: ../../_static/images/deployment/tracking-ui-default.png
    :align: center
    :figwidth: 80%

Open the experient named "wine-quality" on the left navigation. Then click the run "default-params" shown in the table, to find the logged parameters and metrics.
For this case, you should see parameters including ``alpha`` and ``l1_ratio`` and metrics like ``training_score`` and ``mean_absolute_error_X_test``.

Step 4: Running Hyperparameter Tuning
-------------------------------------

Now we have a baseline model, let's try to improve it by tuning the hyperparameters.
Here we run a simple grid search to find the best combination of ``alpha`` and ``l1_ratio``.

.. code-block:: python

  from itertools import product
  import warnings

  warnings.filterwarnings("ignore")

  alphas = [0.2, 0.5, 1.0]
  l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]

  # Create a parent run bundles attempts
  with mlflow.start_run(run_name="hyper-parameter-turning"):
      # Create a child run for each hyperparameter combination
      for alpha, l1 in product(alphas, l1s):
          with mlflow.start_run(nested=True):
              lr = ElasticNet(alpha=alpha, l1_ratio=l1)
              lr.fit(X_train, y_train)

              # In real scenario, you should use a validation set to evaluate
              # the model, but here we use the test set for simplicity
              y_pred = lr.predict(X_test)
              metrics = eval_metrics(y_pred, y_test)

Here we tried 15 combinations of ``alpha`` and ``l1_ratio``. In order to manage a lot of runs nicely, we used parent-child run feature.
This technique is useful when you want to group a set of runs together, like hyper parameter tuning. Please refer to :ref:`Create Child Runs <child_runs>`
for more details.

When you open the MLflow UI again, you should see the runs are grouped under the parent run "hyper-parameter-turning".

In order to compare the results and find the best model, you can visualize the metrics in the MLflow UI.
1. Select the parent job ("hyper-parameter-turning") to select all the child runs together.
2. Click "Chart" tab to see the metrics in a chart.
3. By default it shows a bar chart for a single metric. You can add different chart such as scatter plot to compare multiple metrics.

.. figure:: ../../_static/images/deployment/hyper-parameter-tuning-ui.png
    :align: center
    :figwidth: 80%

In this case, we can see the left top corner is the best model, with ``alpha=0.2`` and ``l1_ratio=0`` (you may see different results).

Step 5: Package Model and Dependencies
--------------------------------------
Since we are using autologging, Mlflow automatically logs `Model <../../../models.html>`_ for each run, which already packages the model weight
and dependencies in the ready to deploy format.

.. note::

  In practice, it is also recommended to use `MLflow Model Registry <../../../model-registry.html>`_ to register and manage the models.


Let's briefly check how the format looks like. You can check the logged model via ``Artifacts`` tab in the Run detail page.

.. code-block::

  model
  ├── MLmodel
  ├── model.pkl
  ├── conda.yaml
  ├── python_env.yaml
  └── requirements.txt

``model.pkl`` is the serialized model weight file, and ``MLmodel`` containe general metadata that tells MLflow how to load the model.
Other files define the dependencies that are needed to run the model.

.. note::

  When you choose manual logging, you need to log the model explicitly using :py:func:`mlflow.sklearn.log_model <mlflow.sklearn.log_model>`
  like this:

  .. code-block:: python

    mlflow.sklearn.log_model(lr, "model")

.. _step-6-test-model-serving-locally:

Step 6: Test the model serving locally
--------------------------------------

Now you are ready to deploy the model, but before that, let's test the model serving locally. As described in the
`Deploy MLflow Model Locally <../deploy-model-locally.html>`_, you can run a local inferecen server by just a single command.
Don't forget to add ``enable-mlserver`` flag to let MLflow use MLServer as the inference server, so it runs the model in the
same way as it will be run in Kubernetes.

.. code-block:: bash

  mlflow models serve -m runs:/<run_id_for_your_best_run>/model -p 1234 --enable-mlserver

This starts a local server that listens on port 1234. You can send a request to the server using ``curl`` command:

.. code-block:: bash

    $ curl -X POST -H "Content-Type:application/json" --data '{"inputs": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]' http://127.0.0.1:1234/invocations

    {"predictions": [-0.03416275504140387]}

You can find more details about the request format and response format in :ref:`Inference Server Specification <local-inference-server-spec>`.


Step 7: Deploy the Model to KServe
----------------------------------

Finally we are all set to deploy the model to Kubernetes cluster.

Create Namespace
~~~~~~~~~~~~~~~~

First, create a test namespace for deploying KServe resources and your model:

.. code-block:: bash

  $ kubectl create namespace mlflow-kserve-test


Create Deployment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a YAML file that describes the model deployment to KServe.

In KServe configuration file, we can specify the model for deployment in two ways:

1. Build a Docker image with the model and specify the image URI.
2. Specify the model URI directly (this only works if your model is stored in remote storage).

Please open the tab below for the details of each approach.



.. tabs::

  .. tab:: Using Docker Image

    .. raw:: html

      <h4>Register Docker Account</h4>

    KServe cannot resolve locally built Docker image, so you need to push the image to a Docker registry.
    In this tutorial, we simply push the image to `Docker Hub <https://hub.docker.com/>`_, however, you can use any other Docker registry such as
    `Amazon ECR <https://aws.amazon.com/ecr/>`_ or private registry.

    If you don't have DockerHub account yet, you can create one at https://hub.docker.com/signup.

    .. raw:: html

      <h4>Build a Docker Image</h4>

    You can build a ready-to-deploy Docker image with ``mlflow models build-docker`` command:

    .. code-block:: bash

      $ mlflow models build-docker -m runs:/<run_id_for_your_best_run>/model -n <your_dockerhub_user_name>/mlflow-wine-classifier --enable-mlserver

    This command will build a Docker image with the model and dependencies, and tag it as ``mlflow-wine-classifier:latest``.

    .. raw:: html

      <h4>Push the Docker Image</h4>

    Once the image is built, you can push it to Docker Hub (or push to other registry you want using the appropriate command):

    .. code-block:: bash

      $ docker push <your_dockerhub_user_name>/mlflow-wine-classifier

    .. raw:: html

      <h4>Write Deployment Configuration</h4>

    Then create a YAML file like this:

    .. code-block:: yaml

      apiVersion: "serving.kserve.io/v1beta1"
      kind: "InferenceService"
      metadata:
        name: "mlflow-wine-classifier"
        namespace: "mlflow-kserve-test"
      spec:
        predictor:
          containers:
            - name: "mlflow-wine-classifier"
              image: "<your_docker_user_name>/mlflow-wine-classifier"
              ports:
                - containerPort: 8080
                  protocol: TCP
              env:
                - name: PROTOCOL
                  value: "v2"


  .. tab:: Using Model URI

    .. raw:: html

      <h4>Get Remote Model URI</h4>

    KServe configuration allows you to specify model URI directly, however, it doesn't support MLflow specific URI schema like ``runs:/`` and ``model:/``,
    and local file URI like ``file:///``. We need to specify the model URI in a remote storage URI format e.g. ``s3://xxx`` or ``gs://xxx``.
    By default, MLflow simply stores the model in local file system, so you have to configure MLflow to store the model in remote storage.
    Please refer to `Artifact Store <../../../tracking.html#artifact-stores>`_ for how to set this up.

    Once you have configured the artifact store, you can repeat the above steps for the model training.

    .. raw:: html

      <h4>Create Deployment Configuration</h4>

    Using the remote model URI, you can create a YAML file like this:

    .. code-block:: yaml

      apiVersion: "serving.kserve.io/v1beta1"
      kind: "InferenceService"
      metadata:
        name: "mlflow-wine-classifier"
        namespace: "mlflow-kserve-test"
      spec:
        predictor:
          model:
            modelFormat:
              name: mlflow
            protocolVersion: v2
            storageUri: "<your_model_uri>"

Deploy Inference Service
~~~~~~~~~~~~~~~~~~~~~~~~

Run the following ``kubectl`` command to deploy a new ``InferenceService`` to your Kubernetes cluster:

.. code-block:: bash

  $ kubectl apply -f YOUR_CONFIG_FILE.yaml

  inferenceservice.serving.kserve.io/mlflow-wine-classifier created

You can check the status of the deployment by running:

.. code-block:: bash

 $ kubectl get inferenceservice mlflow-wine-classifier

  NAME                     URL                                                     READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION
  mlflow-wine-classifier   http://mlflow-wine-classifier.mlflow-kserve-test.local   True             100                    mlflow-wine-classifier-100

.. note::

  It may take a few minutes for the deployment status to be ready. You can also run
  ``kubectl get inferenceservice mlflow-wine-classifier -oyaml`` to see the detailed deployment status and logs.


Test the Deployment
~~~~~~~~~~~~~~~~~~~
Once the deployment is ready, you can send a request to the server.

First, create a test data in JSON file and saved it as ``test-input.json``. We need to send the request in the `V2 Inference Protocol <https://kserve.github.io/website/latest/modelserving/inference_api/#inference-request-json-object>`_,
because we created the model with ``protocolVersion: v2``. The request should look like this:

.. code-block:: json

  {
      "inputs": [
        {
          "name": "input",
          "shape": [13],
          "datatype": "FP32",
          "data": [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
        }
      ]
  }


Then send the request to your inference service:

.. tabs::

  .. tab:: Kubernetes Cluster

      Assuming your cluster is exposed via LoadBalancer, follow `this instruction <https://kserve.github.io/website/0.10/get_started/first_isvc/#4-determine-the-ingress-ip-and-ports>`_ to find the Ingress IP and port.

      .. code-block:: bash

        $ SERVICE_HOSTNAME=$(kubectl get inferenceservice mlflow-wine-classifier -n mlflow-kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
        $ curl -v \
          -H "Host: ${SERVICE_HOSTNAME}" \
          -H "Content-Type: application/json" \
          -d @./test-input.json \
          http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/mlflow-wine-classifier/infer


  .. tab:: Local Machine Emulation

      Normally Kubernetes cluster expose the service via LoadBalancer, but a local cluster created by ``kind`` doesn't have LoadBalancer.
      However, you can still access the service via port-forwarding.

      Open a new terminal and run the following command to forward the port:

      .. code-block:: bash

        $ INGRESS_GATEWAY_SERVICE=$(kubectl get svc -n istio-system --selector="app=istio-ingressgateway" -o jsonpath='{.items[0].metadata.name}')
        $ kubectl port-forward -n istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80

        Forwaring from 127.0.0.1:8080 -> 8080
        Forwarding from [::1]:8080 -> 8080

      Then switch back to your original terminal and send a test request to the server:

      .. code-block:: bash

        $ SERVICE_HOSTNAME=$(kubectl get inferenceservice mlflow-wine-classifier -n mlflow-kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
        $ curl -v \
          -H "Host: ${SERVICE_HOSTNAME}" \
          -H "Content-Type: application/json" \
          -d @./test-input.json \
          http://localhost:8080/v2/models/mlflow-wine-classifier/infer


Troubleshoot
------------

If you have any trouble with the deployment, please consult with the `KServe official documentation <https://kserve.github.io/website/>`_
and their `MLflow Deployment Guide <https://kserve.github.io/website/0.10/modelserving/v1beta1/mlflow/v2/>`_.

Conclusion
----------
Congratulations on finishing the guide! In this tutorial, you learned how to use MLflow to train a model, run hyper parameter tuning,
and deploy the model to Kubernetes cluster.

**Further readings**:

* `MLflow Tracking <../../../tracking.html>`_ - Learn more about MLflow Tracking and different ways of managing experiments and models e.g. team collaboration.
* `MLflow Model Registry <../../../model-registry.html>`_ - Learn more about MLflow Model Registry for how to manage model versions and stages in a centralized model store.
* `MLflow Deployment <../../deployments.html>`_ - Learn more about how MLflow deployment works and different deployment targets.
* `KServe official documentation <https://kserve.github.io/website/>`_ - Learn more about KServe and advanced features such as autoscaling, canary rollout, A/B testing, monitoring, explability, etc.
* `Seldon Core official documentation <https://docs.seldon.io/projects/seldon-core/en/latest/>`_ - Learn more about Seldon Core, another serverless model serving framework we support for Kubernetes.
