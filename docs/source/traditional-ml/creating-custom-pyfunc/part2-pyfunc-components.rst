Understanding PyFunc in MLflow
==============================

In the realm of MLflow, while named flavors offer specific functionalities tailored to popular frameworks, there are situations and 
requirements that fall outside these predefined paths. Enter the custom :py:mod:`pyfunc <mlflow.pyfunc>` (Python function), a universal interface, empowering you to 
encapsulate models from any framework into an MLflow Model by defining a custom Python function.

PyFunc versions of models are interacted with in the same way as any other MLflow model type, providing both :py:func:`save_model() <mlflow.pyfunc.save_model>` and
:py:func:`log_model() <mlflow.pyfunc.log_model>` interfaces in order to create (save) and access (load) the models respectively.

Because these custom models contain the ``python_function`` flavor, they can be deployed
to any of MLflow's supported production environments, such as SageMaker, AzureML, Databricks, Kubernetes, or local REST endpoints.

Why PyFunc?
-----------

1. **Flexibility**: It offers the freedom to work with any machine learning library or framework, ensuring MLflow's adaptability to a wide range of use cases.
2. **Unified Interface**: With `pyfunc`, you get a consistent API. Once your model conforms to this interface, you can leverage all of MLflow's deployment tools without worrying about the underlying framework.
3. **Custom Logic**: Beyond just the model, `pyfunc` allows for preprocessing and postprocessing, enhancing the model's deployment capabilities.

Components of PyFunc
--------------------

1. **Python Function Flavor**: 
   This is the default model interface for MLflow Python models. It ensures every MLflow Python model can be loaded and interacted with using a consistent API.

2. **Filesystem Format**:
   A structured directory that contains all required data, code, and configurations, ensuring the encapsulated model and its dependencies are self-contained and reproducible.

3. **MLModel Configuration**:
   An essential descriptor, the MLmodel file provides details about the model, including its loader module, code, data, and environment.

4. **Custom Pyfunc Models**:
   A powerful feature that goes beyond named flavors, allowing for the creation of models with custom logic, data transformations, and more.

The Power of Custom Pyfunc Models
---------------------------------

While MLflow's named flavors offer out-of-the-box solutions for many frameworks, they might not cater to every requirement. 
This is where custom `pyfunc` models shine. Whether you're working with a niche framework, need to implement specialized preprocessing, 
or want to integrate post-inference logic, custom `pyfunc` models provide the tools to do so.

By defining a Python class inheriting from `PythonModel` and implementing the necessary methods, you can create a custom `pyfunc` model 
tailored to your specific needs.

Conditions where a Custom Pyfunc might be best
----------------------------------------------

There are numerous scenarios where a custom Pyfunc becomes invaluable:

1. **Distributed Inference with Large Models**:
   
   - In distributed systems like Apache Spark or Ray, where inference is parallelized across multiple cores, there's a risk of loading multiple copies of a model, one for each core. This can significantly strain the system's resources, especially with large models.
   - With a custom Pyfunc, you can ensure that each worker node or executor loads only a single copy of the model, optimizing resource usage and speeding up inference.

2. **Unsupported Models**:

   - While MLflow offers a wide range of named flavors for popular frameworks, the machine learning ecosystem is vast. There might be niche or emerging frameworks that aren't yet supported.
   - Custom Pyfunc provides a way to encapsulate and manage models from any such unsupported frameworks seamlessly.

3. **Custom Inference Methods**:

   - The default `.predict()` method might not always cater to specific requirements. Perhaps you need a method that produces logits, uncertainties, or other metrics.
   - A custom Pyfunc can wrap around any inference method, ensuring that the deployed model behaves exactly as needed.

4. **Loading Ancillary Data or External Systems**:

   - Sometimes, a model's inference isn't just about the model itself. It might need to reference external data that wasn't saved with the model, or it might need to connect to other systems.
   - Consider a scenario where a model needs to look up entries in a vector database during prediction. A custom Pyfunc can utilize the `load_context` method to load a configuration file. This provides the custom `predict` method with configuration data, enabling it to connect to external services during inference.

Inner workings of Custom Pyfunc
-------------------------------

Understanding the sequence of events during the `mlflow.pyfunc.load_model()` call is crucial to harnessing the full power of custom Pyfuncs. 
Here's a step-by-step breakdown of the sequence of events that happens when loading a custom pyfunc and how declaring overrides during saving the model 
are accessed and referenced to control the behavior of the loaded model object.

.. figure:: ../../_static/images/guides/introductory/creating-custom-pyfunc/pyfunc_loading.svg
   :width: 90%
   :align: center
   :alt: Tags, experiments, and runs relationships

   Pyfunc loading process

1. **Initiation**:
   
   - The process starts when `mlflow.pyfunc.load_model()` is called, indicating the intention to load a custom Pyfunc model for use.

2. **Model Configuration Retrieval**:

   - The system fetches the `MLmodel` configuration file associated with the saved model. This descriptor provides essential details about the model, including its loader module, code, data, and environment.

3. **Artifact Mapping**:

   - The saved model artifacts, which could include serialized model objects, ancillary data, or other necessary files, are mapped. This mapping ensures that the custom Pyfunc knows where to find everything it needs.

4. **Python Model Initialization**:

   - The Python class that defines the custom Pyfunc (typically inheriting from `PythonModel`) is initialized. At this stage, the model isn't ready for inference yet but is prepared for the subsequent loading steps.

5. **Context Loading**:

   - The `load_context` method of the custom Pyfunc is invoked. This method is designed to load any external references or perform initialization tasks. For instance, it could deserialize a model object, load a configuration file for connecting to an external service, or prepare any other resources the model needs.

6. **Model Ready**:

   - With the context loaded, the custom Pyfunc model is now fully initialized and ready for inference. Any subsequent calls to its `predict` method will now execute the custom logic defined within, producing results as designed.

It's worth noting that this sequence ensures that the custom Pyfunc model, once loaded, is a fully self-contained unit, encapsulating not just the model but also any custom logic, data transformations, and external references it needs. This design ensures reproducibility and consistency, regardless of where the model is deployed.


Next Steps
----------

Now that you understand the importance and components of `pyfunc`, the next step is to dive into seeing how they can be built. 

.. raw:: html

    <a href="notebooks/index.html" class="download-btn">Explore the tutorial notebooks</a>
