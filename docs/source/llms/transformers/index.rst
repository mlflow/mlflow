MLflow Transformers Flavor
==========================

.. attention::
    The ``transformers`` flavor is in active development and is marked as Experimental. Public APIs may change and new features are
    subject to be added as additional functionality is brought to the flavor.

Introduction
------------

**Transformers** by 🤗 `Hugging Face <https://huggingface.co/docs/transformers/index>`_ represents a cornerstone in the realm of 
machine learning, offering state-of-the-art capabilities for a multitude of frameworks including `PyTorch <https://pytorch.org/>`_, 
`TensorFlow <https://www.tensorflow.org/>`_, and `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_. 
This library has become the de facto standard for natural language processing (NLP) and audio transcription processing. 
It also provides a compelling and advanced set of options for computer vision and multimodal AI tasks. 
Transformers achieves all of this by providing pre-trained models and accessible high-level APIs that are not only powerful 
but also versatile and easy to implement.

For instance, one of the cornerstones of the simplicity of the transformers library is the `pipeline API <https://huggingface.co/transformers/main_classes/pipelines.html>`_, 
an encapsulation of the most common NLP tasks into a single API call. This API allows users to perform a variety of tasks based on the specified task without 
having to worry about the underlying model or the preprocessing steps. 

.. figure:: ../../_static/images/tutorials/llms/transformers-pipeline-architecture.png
   :alt: Transformers Pipeline Architecture
   :width: 80%
   :align: center

   Transformers Pipeline Architecture for the Whisper Model

The integration of Transformers with MLflow brings a seamless experience for tracking and deploying these models. 
MLflow, a platform dedicated to the end-to-end machine learning lifecycle, now supports the Transformers library, 
enabling practitioners to manage experiments, track model versions, and serve models with unprecedented ease.

The Transformers library offers:

- **Pre-trained Models**: Access to a comprehensive collection of `pre-trained models <https://huggingface.co/models>`_, reducing compute costs, carbon footprint, and the time needed to train models from scratch.
- **Task Versatility**: Support for common tasks across different modalities such as text classification, entity recognition, question answering, summarization, translation, image classification, object detection, segmentation, speech recognition, and more.
- **Framework Interoperability**: Flexibility to train models in one framework and deploy in another, supporting PyTorch, TensorFlow, and JAX, with additional export options for ONNX and TorchScript.
- **Community**: A highly active and engaged community for collaboration and support through forums and the Hugging Face Hub.

MLflow's Transformers flavor is designed to harness the power of the Transformers library within the MLflow ecosystem, providing tools for:

- **Simplified Experiment Tracking**: Log parameters, metrics, and models with ease during the fine-tuning process.
- **Effortless Deployment**: Deploy trained models with simple API calls, catering to a variety of production environments.
- **Comprehensive Model Support**: Compatibility with a wide range of models from the Transformers library, ensuring that the latest advancements in AI are readily accessible.

Whether you are a data scientist, a machine learning engineer, or a researcher, MLflow's integration with Transformers offers a 
robust and efficient pathway to incorporate cutting-edge NLP and AI capabilities into your applications.

Features
--------

With MLflow's Transformers flavor, users can:

- Log and save Transformer models directly within MLflow.
- Track experiments, including parameters, metrics, and artifacts.
- Deploy models for inference with ease.
- Leverage the ``python_function`` flavor for generic Python function inference.

MLflow Flavor for Transformers
------------------------------

The MLflow Transformers flavor is designed to simplify the machine learning workflow. It enables users to:

- Fine-tune foundational models on custom datasets.
- Keep track of training metrics, parameters, and outputs.
- Deploy models for inference with minimal configuration.

Simplified Fine-Tuning and Experiment Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fine-tuning a pre-trained model is a common practice in machine learning. MLflow streamlines this process by allowing users to log 
the fine-tuning parameters, track the training metrics, and save the resulting model. This ensures reproducibility and facilitates 
comparison across different experiments.

Deployment Made Easy
^^^^^^^^^^^^^^^^^^^^

Once a model is trained, it needs to be deployed for inference. MLflow's integration with Transformers simplifies this by providing 
functions such as :py:func:`mlflow.transformers.load_model` and :py:func:`mlflow.pyfunc.load_model`, which allow for easy model serving.

Getting Started with the MLflow Transformers Flavor - Tutorials and Guides
--------------------------------------------------------------------------

Below, you will find a number of guides that focus on different use cases (`tasks`) using `transformers`  that leverage MLflow's 
APIs for tracking and inference capabilities. 

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorials/text-generation/text-generation.html">
                    <div class="header">
                        Text Generation with Transformers
                    </div>
                    <p>
                        Learn how to leverage the transformers integration with MLflow in this introductory tutorial.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/audio-transcription/whisper.html">
                    <div class="header">
                        Audio Transcription with Transformers
                    </div>
                    <p>
                        Learn how to leverage the Whisper Model with MLflow to generate accurate audio transcriptions.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/translation/component-translation.html">
                    <div class="header">
                        Translation with Transformers
                    </div>
                    <p>
                        Learn about the options for saving and loading transformers models in MLflow for customization of your workflows with a fun translation example!
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/conversational/conversational-model.html">
                    <div class="header">
                        Chat with Transformers
                    </div>
                    <p>
                        Learn the basics of stateful chat Conversational Pipelines with Transformers and MLflow.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../custom-pyfunc-for-llms/index.html">
                    <div class="header">
                        Custom PyFunc for Transformers
                    </div>
                    <p>
                        Learn how to define a custom PyFunc using transformers for advanced, state-of-the-art new models.
                    </p>
                </a>
            </div>
        </article>
    </section>

Download the Tutorial Notebooks to try them locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download the transformers tutorial notebooks to run in your environment, click the respective links below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/transformers/tutorials/text-generation/text-generation.ipynb" class="notebook-download-btn">Download the Introductory Text Generation Notebook</a><br>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/transformers/tutorials/audio-transcription/whisper.ipynb" class="notebook-download-btn">Download the Audio Transcription Notebook</a><br>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/transformers/tutorials/translation/component-translation.ipynb" class="notebook-download-btn">Download the Translation Notebook</a><br>        
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/transformers/tutorials/conversational/conversational-model.ipynb" class="notebook-download-btn">Download the Chat Conversational Notebook</a><br>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm.ipynb" class="notebook-download-btn">Download the Custom PyFunc transformers Notebook</a><br>

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorials/text-generation/text-generation.ipynb
    tutorials/audio-transcription/whisper.ipynb
    tutorials/translation/component-translation.ipynb
    tutorials/conversational/conversational-model.ipynb


Options for Logging Transformers Models - Pipelines vs. Component logging
-------------------------------------------------------------------------

The transformers flavor has two different primary mechanisms for saving and loading models: pipelines and components.

**Pipelines**

Pipelines in the context of the Transformers library are high-level objects that combine pre-trained models and tokenizers 
(as well as other components, depending on the task type) to perform a specific task. They abstract away much of the preprocessing 
and postprocessing work involved in using the models. 

For example, a text classification pipeline would handle the tokenization of text, passing the tokens through a model, and then 
interpreting the logits to produce a human-readable classification.

When logging a pipeline with MLflow, you're essentially saving this high-level abstraction, which can be loaded and used directly 
for inference with minimal setup. This is ideal for end-to-end tasks where the preprocessing and postprocessing steps are standard 
for the task at hand.

**Components**

Components refer to the individual parts that can make up a pipeline, such as the model itself, the tokenizer, and any additional 
processors, extractors, or configuration needed for a specific task. Logging components with MLflow allows for more flexibility and 
customization. You can log individual components when your project needs to have more control over the preprocessing and postprocessing 
steps or when you need to access the individual components in a bespoke manner that diverges from how the pipeline abstraction would call them.

For example, you might log the components separately if you have a custom tokenizer or if you want to apply some special postprocessing 
to the model outputs. When loading the components, you can then reconstruct the pipeline with your custom components or use the components 
individually as needed.


Important Details to be aware of with the transformers flavor
-------------------------------------------------------------

When working with the transformers flavor in MLflow, there are several important considerations to keep in mind:

- **Experimental Status**: The Transformers flavor in MLflow is marked as experimental, which means that APIs are subject to change, and new features may be added over time with potentially breaking changes.
- **PyFunc Limitations**: Not all output from a Transformers pipeline may be captured when using the python_function flavor. For example, if additional references or scores are required from the output, the native implementation should be used instead.
- **Supported Pipeline Types**: Not all Transformers pipeline types are currently supported for use with the python_function flavor. In particular, new model architectures may not be supported until the transformers library has a designated pipeline type in its supported pipeline implementations.
- **Input and Output Types**: The input and output types for the python_function implementation may differ from those expected from the native pipeline. Users need to ensure compatibility with their data processing workflows.
- **Model Configuration**: When saving or logging models, the `model_config` can be used to set certain parameters. However, if both model_config and a `ModelSignature` with parameters are saved, the default parameters in ModelSignature will override those in `model_config`.
- **Audio and Vision Models**: Audio and text-based large language models are supported for use with pyfunc, while other types like computer vision and multi-modal models are only supported for native type loading.

The currently supported pipeline types for Pyfunc can be seen `here <../../models.html#supported-transformers-pipeline-types-for-pyfunc>`_.

`Detailed Documentation <guide/index.html>`_
--------------------------------------------

To learn more about the nuances of the `transformers` flavor in MLflow, delve into the comprehensive guide, which covers:

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Comprehensive Guide</a>

- **Transformers within MLflow**: Explore the integration of the transformers library within MLflow and learn about its support for models, components, and pipelines.

- **Input and Output Types for PyFunc**: Understand the standardization of input and output formats in the pyfunc model implementation for the flavor, ensuring seamless integration with JSON and Pandas DataFrames.

- **Supported Transformers Pipeline Types for Pyfunc**: Familiarize yourself with the various ``transformers`` pipeline types compatible with the pyfunc model flavor and their respective input and output data types.

- **Using Model Config and Model Signature Params for `Transformers` Inference**: Learn how to leverage ``model_config`` and ``ModelSignature`` for flexible and customized model loading and inference.

- **Example of Loading a Transformers Model as a Python Function**: Walk through a practical example demonstrating how to log, load, and interact with a pre-trained conversational model in MLflow.

- **Save and Load Options for Transformers**: Explore the different approaches for saving model components or complete pipelines and understand the nuances of loading these models for various use cases.

- **Automatic Metadata and ModelCard Logging**: Discover the automatic logging features for model cards and other metadata, enhancing model documentation and transparency.

- **Automatic Signature Inference**: Learn about MLflow's capability within the ``transformers`` flavor to automatically infer and attach model signatures, facilitating easier model deployment.

- **Scalability for Inference**: Gain insights into optimizing ``transformers`` models for inference, focusing on memory optimization and data type configurations.

- **Input Data Types for Audio Pipelines**: Understand the specific requirements for handling audio data in ``transformers`` pipelines, including the handling of different input types like ``str``, ``bytes``, and ``np.ndarray``.


.. toctree::
    :maxdepth: 1
    :hidden:

    guide/index.rst
 