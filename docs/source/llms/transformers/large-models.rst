Working with Large Models in MLflow Transformers flavor
=======================================================

.. warning::

    The features described in this guide are intended for advanced users familiar with Transformers and MLflow. Please understand the limitations and potential risks associated with these features before use.

The `MLflow Transformers flavor <../index.html>`_ allows you to track various Transformers models in MLflow. However, logging large models such as Large Language Models (LLMs) can be resource-intensive due to their size and memory requirements. This guide outlines MLflow's features for reducing memory and disk usage when logging models, enabling you to work with large models in resource-constrained environments.

Overview
--------

The following table summarizes the different methods for logging models with the Transformers flavor. Please be aware that each method has certain limitations and requirements, as described in the following sections.

.. list-table::
    :header-rows: 1

    * - Save method
      - Description
      - Memory Usage
      - Disk Usage
      - Example

    * - Normal pipeline-based logging
      - Log a model using a pipeline instance or a dictionary of pipeline components.
      - High
      - High
      -
        .. code-block:: python

          import mlflow
          import transformers

          pipeline = transformers.pipeline(
              task="text-generation",
              model="meta-llama/Meta-Llama-3.1-70B",
          )

          with mlflow.start_run():
              mlflow.transformers.log_model(
                  transformers_model=pipeline,
                  artifact_path="model",
              )

    * - :ref:`Memory-Efficient Model Logging <transformers-memory-efficient-logging>`
      - Log a model by specifying a path to a local checkpoint, avoiding loading the model into memory.
      - **Low**
      - High
      -
        .. code-block:: python

          import mlflow

          with mlflow.start_run():
              mlflow.transformers.log_model(
                  # Pass a path to local checkpoint as a model
                  transformers_model="/path/to/local/checkpoint",
                  # Task argument is required for this saving mode.
                  task="text-generation",
                  artifact_path="model",
              )

    * - :ref:`Storage-Efficient Model Logging <transformers-save-pretrained-guide>`
      - Log a model by saving a reference to the HuggingFace Hub repository instead of the model weights.
      - High
      - **Low**
      -
        .. code-block:: python

          import mlflow
          import transformers

          pipeline = transformers.pipeline(
              task="text-generation",
              model="meta-llama/Meta-Llama-3.1-70B",
          )

          with mlflow.start_run():
              mlflow.transformers.log_model(
                  transformers_model=pipeline,
                  artifact_path="model",
                  # Set save_pretrained to False to save storage space
                  save_pretrained=False,
              )


.. _transformers-memory-efficient-logging:

Memory-Efficient Model Logging
------------------------------

Introduced in MLflow 2.16.1, this method allows you to log a model without loading it into memory:

.. code-block:: python

    import mlflow

    with mlflow.start_run():
        mlflow.transformers.log_model(
            # Pass a path to local checkpoint as a model to avoid loading the model instance
            transformers_model="path/to/local/checkpoint",
            # Task argument is required for this saving mode.
            task="text-generation",
            artifact_path="model",
        )


In the above example, we pass a path to the local model checkpoint/weight as the model argument in the  :py:func:`mlflow.transformers.log_model()` API, instead of a pipeline instance. MLflow will inspect the model metadata of the checkpoint and log the model weights without loading them into memory. This way, you can log an enormous multi-billion parameter model  to MLflow with minimal computational resources.


Important Notes
~~~~~~~~~~~~~~~

Please be aware of the following requirements and limitations when using this feature:

1. The checkpoint directory **must** contain a valid config.json file and the model weight files. If a tokenizer is required, its state file must also be present in the checkpoint directory. You can save the tokenizer state in your checkpoint directory by calling ``tokenizer.save_pretrained("path/to/local/checkpoint")`` method.
2. You **must** specify the ``task`` argument with the appropriate task name that the model is designed for.
3. MLflow may not accurately infer model dependencies in this mode. Please refer to `Managing Dependencies in MLflow Models <../../model/dependencies.html>`_ for more information on managing dependencies for your model.

.. warning::

    Ensure you specify the correct task argument, as an incompatible task will cause the model to **fail at the load time**. You can check the valid task type for your model on the HuggingFace Hub.


.. _transformers-save-pretrained-guide:

Storage-Efficient Model Logging
-------------------------------

Typically, when MLflow logs an ML model, it saves a copy of the model weight to the artifact store.
However, this is not optimal when you use a pretrained model from HuggingFace Hub and have no intention of fine-tuning or otherwise manipulating the model or its weights before logging it. For this very common case, copying the (typically very large) model weights is redundant while developing prompts, testing inference parameters, and otherwise is little more than an unnecessary waste of storage space.

To address this issue, MLflow 2.11.0 introduced a new argument ``save_pretrained`` in the :py:func:`mlflow.transformers.save_model()` and :py:func:`mlflow.transformers.log_model()` APIs. When with argument is set to ``False``, MLflow will forego saving the pretrained model weights, opting instead to store a reference to the underlying repository entry on the HuggingFace Hub; specifically, the repository name and the unique commit hash of the model weights are stored when your components or pipeline are logged. When loading back such a *reference-only* model, MLflow will check the repository name and commit hash from the saved metadata, and either download the model weight from the HuggingFace Hub or use the locally cached model from your HuggingFace local cache directory.

Here is the example of using ``save_pretrained`` argument for logging a model

.. code-block:: python

    import transformers

    pipeline = transformers.pipeline(
        task="text-generation",
        model="meta-llama/Meta-Llama-3.1-70B",
        torch_dtype="torch.float16",
    )

    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            # Set save_pretrained to False to save storage space
            save_pretrained=False,
        )

In the above example, MLflow will not save a copy of the **Llama-3.1-70B** model's weights and will instead log the following metadata as a reference to the HuggingFace Hub model. This will save roughly 150GB of storage space and reduce the logging latency significantly as well for each run that you initiate during development.

By navigating to the MLflow UI, you can see the model logged with the repository ID and commit hash:

.. code-block:: bash

    flavors:
        ...
        transformers:
            source_model_name: meta-llama/Meta-Llama-3.1-70B-Instruct
            source_model_revision: 33101ce6ccc08fa6249c10a543ebfcac65173393
            ...

Before production deployments, you may want to persist the model weight instead of the repository reference. To do so, you can use the :py:func:`mlflow.transformers.persist_pretrained_model()` API to download the model weight from the HuggingFace Hub and save it to the artifact location. Please refer to the :ref:`persist-pretrained-guide` section for more information.

Registering Reference-Only Models for Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The models logged with either of the above optimized methods are "reference-only", meaning that the model weight is not saved to the artifact store and only the reference to the HuggingFace Hub repository is saved. When you load the model back normally, MLflow will download the model weight from the HuggingFace Hub.

However, this may not be suitable for production use cases, as the model weight may be unavailable or the download may fail due to network issues. MLflow provides a solution to address this issue when registering reference-models to the Model Registry.


Databricks Unity Catalog
^^^^^^^^^^^^^^^^^^^^^^^^

Registering reference-only models to `Databricks Unity Catalog Model Registry <https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html>`_ requires **no additional steps** than the normal model registration process. MLflow automatically downloads and registers the model weights to Unity Catalog along with the model metadata.


.. code-block:: python

    import mlflow

    mlflow.set_registry_uri("databricks-uc")

    # Log the repository ID as a model. The model weight will not be saved to the artifact store
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            artifact_path="model",
        )

    # When registering the model to Unity Catalog Model Registry, MLflow will automatically
    # persist the model weight files. This may take a several minutes for large models.
    mlflow.register_model(model_info.model_uri, "your.model.name")

.. _persist-pretrained-guide:

OSS Model Registry or Legacy Workspace Model Registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For OSS Model Registry or the legacy Workspace Model Registry in Databricks, you need to manually persist the
model weight to the artifact store before registering the model. You can use the :py:func:`mlflow.transformers.persist_pretrained_model()` API to download the model weight from the HuggingFace Hub and save it to the artifact location. The process **does NOT require re-logging a model** but efficiently update the existing model and metadata in-place.

.. code-block:: python

    import mlflow

    # Log the repository ID as a model. The model weight will not be saved to the artifact store
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            artifact_path="model",
        )

    # Before registering the model to the non-UC model registry, persist the model weight
    # from the HuggingFace Hub to the artifact location.
    mlflow.transformers.persist_pretrained_model(model_info.model_uri)

    # Register the model
    mlflow.register_model(model_info.model_uri, "your.model.name")


.. _caveats-of-save-pretrained:

Caveats for Skipping Saving of Pretrained Model Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While these features are useful for saving computational resources and storage space for logging large models, there are some caveats to be aware of:

* **Change in Model Availability**: If you are using a model from other users' repository, the model may be deleted or become private in the HuggingFace Hub. In such cases, MLflow cannot load the model back. For production use cases, it is recommended to save a  copy of the model weights to the artifact store prior to moving from development or staging to production for your model.

* **HuggingFace Hub Access**: Downloading a model from the HuggingFace Hub might be slow or unstable due to the network latency or the HuggingFace Hub service status. MLflow doesn't provide any retry mechanism or robust error handling for model downloading from the HuggingFace Hub. As such, you should not rely on this functionality for your final production-candidate run.

By understanding these methods and their limitations, you can effectively work with large Transformers models in MLflow while optimizing resource usage.
