🤗 Transformers within MLflow
=============================

.. attention::
    The ``transformers`` flavor is in active development and is marked as Experimental. Public APIs may change and new features are
    subject to be added as additional functionality is brought to the flavor.

The ``transformers`` model flavor enables logging of
`transformers models, components, and pipelines <https://huggingface.co/docs/transformers/index>`_ in MLflow format via
the :py:func:`mlflow.transformers.save_model()` and :py:func:`mlflow.transformers.log_model()` functions. Use of these
functions also adds the ``python_function`` flavor to the MLflow Models that they produce, allowing the model to be
interpreted as a generic Python function for inference via :py:func:`mlflow.pyfunc.load_model()`.
You can also use the :py:func:`mlflow.transformers.load_model()` function to load a saved or logged MLflow
Model with the ``transformers`` flavor in the native transformers formats.

This page explains the detailed features and configurations of the MLflow ``transformers`` flavor. For the general introduction about the MLflow's Transformer integration, please refer to the `MLflow Transformers Flavor <../index.html>`_ page.

.. contents:: Table of Contents
  :local:
  :depth: 1

Loading a Transformers Model as a Python Function
-------------------------------------------------

Supported Transformers Pipeline types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``transformers`` :ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>` simplifies
and standardizes both the inputs and outputs of pipeline inference. This conformity allows for serving
and batch inference by coercing the data structures that are required for ``transformers`` inference pipelines
to formats that are compatible with json serialization and casting to Pandas DataFrames.

.. note::
    Certain `TextGenerationPipeline` types, particularly instructional-based ones, may return the original
    prompt and included line-formatting carriage returns `"\n"` in their outputs. For these pipeline types,
    if you would like to disable the prompt return, you can set the following in the `model_config` dictionary when
    saving or logging the model: `"include_prompt": False`. To remove the newline characters from within the body
    of the generated text output, you can add the `"collapse_whitespace": True` option to the `model_config` dictionary.
    If the pipeline type being saved does not inherit from `TextGenerationPipeline`, these options will not perform
    any modification to the output returned from pipeline inference.

.. attention::
    Not all ``transformers`` pipeline types are supported. See the table below for the list of currently supported Pipeline
    types that can be loaded as ``pyfunc``.

    In the current version, audio and text-based large language
    models are supported for use with ``pyfunc``, while computer vision, multi-modal, timeseries,
    reinforcement learning, and graph models are only supported for native type loading via :py:func:`mlflow.transformers.load_model()`

    Future releases of MLflow will introduce ``pyfunc`` support for these additional types.

The table below shows the mapping of ``transformers`` pipeline types to the :ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>`
data type inputs and outputs.

.. important::
    The inputs and outputs of the ``pyfunc`` implementation of these pipelines *are not guaranteed to match* the input types and output types that would
    return from a native use of a given pipeline type. If your use case requires access to scores, top_k results, or other additional references within
    the output from a pipeline inference call, please use the native implementation by loading via :py:func:`mlflow.transformers.load_model()` to
    receive the full output.

    Similarly, if your use case requires the use of raw tensor outputs or processing of outputs through an external ``processor`` module, load the
    model components directly as a ``dict`` by calling :py:func:`mlflow.transformers.load_model()` and specify the ``return_type`` argument as 'components'.


================================= ============================== ==========================================================================
Pipeline Type                     Input Type                     Output Type
================================= ============================== ==========================================================================
Instructional Text Generation     str or List[str]               List[str]
Conversational                    str or List[str]               List[str]
Summarization                     str or List[str]               List[str]
Text Classification               str or List[str]               pd.DataFrame (dtypes: {'label': str, 'score': double})
Text Generation                   str or List[str]               List[str]
Text2Text Generation              str or List[str]               List[str]
Token Classification              str or List[str]               List[str]
Translation                       str or List[str]               List[str]
ZeroShot Classification*          Dict[str, [List[str] | str]*   pd.DataFrame (dtypes: {'sequence': str, 'labels': str, 'scores': double})
Table Question Answering**        Dict[str, [List[str] | str]**  List[str]
Question Answering***             Dict[str, str]***              List[str]
Fill Mask****                     str or List[str]****           List[str]
Feature Extraction                str or List[str]               np.ndarray
AutomaticSpeechRecognition        bytes*****, str, or np.ndarray List[str]
AudioClassification               bytes*****, str, or np.ndarray pd.DataFrame (dtypes: {'label': str, 'score': double})
================================= ============================== ==========================================================================

\* A collection of these inputs can also be passed. The standard required key names are 'sequences' and 'candidate_labels', but these may vary.
Check the input requirments for the architecture that you're using to ensure that the correct dictionary key names are provided.

\** A collection of these inputs can also be passed. The reference table must be a json encoded dict (i.e. {'query': 'what did we sell most of?', 'table': json.dumps(table_as_dict)})

\*** A collection of these inputs can also be passed. The standard required key names are 'question' and 'context'. Verify the expected input key names match the
expected input to the model to ensure your inference request can be read properly.

\**** The mask syntax for the model that you've chosen is going to be specific to that model's implementation. Some are '[MASK]', while others are '<mask>'. Verify the expected syntax to
avoid failed inference requests.

\***** If using `pyfunc` in MLflow Model Serving for realtime inference, the raw audio in bytes format must be base64 encoded prior to submitting to the endpoint. String inputs will be interpreted as uri locations.

Example of loading a transformers model as a python function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the below example, a simple pre-trained model is used within a pipeline. After logging to MLflow, the pipeline is
loaded as a ``pyfunc`` and used to generate a response from a passed-in list of strings.

.. code-block:: python

    import mlflow
    import transformers

    # Read a pre-trained conversation pipeline from HuggingFace hub
    conversational_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

    # Define the signature
    signature = mlflow.models.infer_signature(
        "Hi there, chatbot!",
        mlflow.transformers.generate_signature_output(
            conversational_pipeline, "Hi there, chatbot!"
        ),
    )

    # Log the pipeline
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=conversational_pipeline,
            artifact_path="chatbot",
            task="conversational",
            signature=signature,
            input_example="A clever and witty question",
        )

    # Load the saved pipeline as pyfunc
    chatbot = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    # Ask the chatbot a question
    response = chatbot.predict("What is machine learning?")

    print(response)

    # >> [It's a new thing that's been around for a while.]


Saving Prompt Templates with Transformer Pipelines
--------------------------------------------------

.. note::

    This feature is only available in MLflow 2.10.0 and above.

MLflow supports specifying prompt templates for certain pipeline types:

- `feature-extraction <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FeatureExtractionPipeline>`_
- `fill-mask <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FillMaskPipeline>`_
- `summarization <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline>`_
- `text2text-generation <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.Text2TextGenerationPipeline>`_
- `text-generation <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextGenerationPipeline>`_

Prompt templates are strings that are used to format user inputs prior to ``pyfunc`` inference. To specify a prompt template,
use the ``prompt_template`` argument when calling :py:func:`mlflow.transformers.save_model()` or :py:func:`mlflow.transformers.log_model()`.
The prompt template must be a string with a single format placeholder, ``{prompt}``. 

For example:

.. code-block:: python

    import mlflow
    from transformers import pipeline

    # Initialize a pipeline. `distilgpt2` uses a "text-generation" pipeline
    generator = pipeline(model="distilgpt2")

    # Define a prompt template
    prompt_template = "Answer the following question: {prompt}"

    # Save the model
    mlflow.transformers.save_model(
        transformers_model=generator,
        path="path/to/model",
        prompt_template=prompt_template,
    )

When the model is then loaded with :py:func:`mlflow.pyfunc.load_model()`, the prompt
template will be used to format user inputs before passing them into the pipeline:

.. code-block:: python

    import mlflow

    # Load the model with pyfunc
    model = mlflow.pyfunc.load_model("path/to/model")

    # The prompt template will be used to format this input, so the
    # string that is passed to the text-generation pipeline will be:
    # "Answer the following question: What is MLflow?"
    model.predict("What is MLflow?")

.. note::

    ``text-generation`` pipelines with a prompt template will have the `return_full_text pipeline argument <https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters.return_full_text>`_
    set to ``False`` by default. This is to prevent the template from being shown to the users,
    which could potentially cause confusion as it was not part of their original input. To
    override this behaviour, either set ``return_full_text`` to ``True`` via ``params``, or by 
    including it in a ``model_config`` dict in ``log_model()``. See `this section <#using-model-config-and-model-signature-params-for-transformers-inference>`_ 
    for more details on how to do this.

For a more in-depth guide, check out the `Prompt Templating notebook <../tutorials/prompt-templating/prompt-templating.ipynb>`_!


Using model_config and Model Signature Params for Inference
-----------------------------------------------------------

For `transformers` inference, there are two ways to pass in additional arguments to the pipeline.

* Use ``model_config`` when saving/logging the model. Optionally, specify ``model_config`` when calling ``load_model``.
* Specify params at inference time when calling ``predict()``

Use ``model_config`` to control how the model is loaded and inference performed for all input samples. Configuration in
``model_config`` is not overridable at ``predict()`` time unless a ``ModelSignature`` is indicated with the same parameters.

Use ``ModelSignature`` with params schema, on the other hand, to allow downstream consumers to provide additional inference
params that may be needed to compute the predictions for their specific samples.

.. note::
    If both ``model_config`` and ``ModelSignature`` with parameters are saved when logging model, both of them
    will be used for inference. The default parameters in ``ModelSignature`` will override the params in ``model_config``.
    If extra ``params`` are provided at inference time, they take precedence over all params. We recommend using 
    ``model_config`` for those parameters needed to run the model in general for all the samples. Then, add 
    ``ModelSignature`` with parameters for those extra parameters that you want downstream consumers to indicated at
    per each of the samples.

* Using ``model_config``

.. code-block:: python

    import mlflow
    from mlflow.models import infer_signature
    from mlflow.transformers import generate_signature_output
    import transformers

    architecture = "mrm8488/t5-base-finetuned-common_gen"
    model = transformers.pipeline(
        task="text2text-generation",
        tokenizer=transformers.T5TokenizerFast.from_pretrained(architecture),
        model=transformers.T5ForConditionalGeneration.from_pretrained(architecture),
    )
    data = "pencil draw paper"

    # Infer the signature
    signature = infer_signature(
        data,
        generate_signature_output(model, data),
    )

    # Define an model_config
    model_config = {
        "num_beams": 5,
        "max_length": 30,
        "do_sample": True,
        "remove_invalid_values": True,
    }

    # Saving model_config with the model
    mlflow.transformers.save_model(
        model,
        path="text2text",
        model_config=model_config,
        signature=signature,
    )

    pyfunc_loaded = mlflow.pyfunc.load_model("text2text")
    # model_config will be applied
    result = pyfunc_loaded.predict(data)

    # overriding some inference configuration with diferent values
    pyfunc_loaded = mlflow.pyfunc.load_model(
        "text2text", model_config=dict(do_sample=False)
    )

.. note::
    Note that in the previous example, the user can't override the configuration ``do_sample``
    when calling ``predict``.

* Specifying params at inference time

.. code-block:: python

    import mlflow
    from mlflow.models import infer_signature
    from mlflow.transformers import generate_signature_output
    import transformers

    architecture = "mrm8488/t5-base-finetuned-common_gen"
    model = transformers.pipeline(
        task="text2text-generation",
        tokenizer=transformers.T5TokenizerFast.from_pretrained(architecture),
        model=transformers.T5ForConditionalGeneration.from_pretrained(architecture),
    )
    data = "pencil draw paper"

    # Define an model_config
    model_config = {
        "num_beams": 5,
        "remove_invalid_values": True,
    }

    # Define the inference parameters params
    inference_params = {
        "max_length": 30,
        "do_sample": True,
    }

    # Infer the signature including params
    signature_with_params = infer_signature(
        data,
        generate_signature_output(model, data),
        params=inference_params,
    )

    # Saving model with signature and model config
    mlflow.transformers.save_model(
        model,
        path="text2text",
        model_config=model_config,
        signature=signature_with_params,
    )

    pyfunc_loaded = mlflow.pyfunc.load_model("text2text")

    # Pass params at inference time
    params = {
        "max_length": 20,
        "do_sample": False,
    }

    # In this case we only override max_length and do_sample,
    # other params will use the default one saved on ModelSignature
    # or in the model configuration.
    # The final params used for prediction is as follows:
    # {
    #    "num_beams": 5,
    #    "max_length": 20,
    #    "do_sample": False,
    #    "remove_invalid_values": True,
    # }
    result = pyfunc_loaded.predict(data, params=params)


Pipelines vs. Component Logging
-------------------------------

The transformers flavor has two different primary mechanisms for saving and loading models: pipelines and components.

.. note::
    Saving transformers models with custom code (i.e. models that require ``trust_remote_code=True``) requires ``transformers >= 4.26.0``.

**Pipelines**

Pipelines, in the context of the Transformers library, are high-level objects that combine pre-trained models and tokenizers 
(as well as other components, depending on the task type) to perform a specific task. They abstract away much of the preprocessing 
and postprocessing work involved in using the models. 

For example, a text classification pipeline would handle the tokenization of text, passing the tokens through a model, and then interpret the logits to produce a human-readable classification.

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

.. note::
    MLflow by default uses a 500 MB `max_shard_size` to save the model object in :py:func:`mlflow.transformers.save_model()` or :py:func:`mlflow.transformers.log_model()` APIs. You can use the environment variable `MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE` to override the value.

.. note::
    For component-based logging, the only requirement that must be met in the submitted ``dict`` is that a model is provided. All other elements of the ``dict`` are optional.

Logging a components-based model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below shows logging components of a ``transformers`` model via a dictionary mapping of specific named components. The names of the keys within the submitted dictionary
must be in the set: ``{"model", "tokenizer", "feature_extractor", "image_processor"}``. Processor type objects (some image processors, audio processors, and multi-modal processors)
must be saved explicitly with the ``processor`` argument in the :py:func:`mlflow.transformers.save_model()` or :py:func:`mlflow.transformers.log_model()` APIs.

After logging, the components are automatically inserted into the appropriate ``Pipeline`` type for the task being performed and are returned, ready for inference.

.. note::
    The components that are logged can be retrieved in their original structure (a dictionary) by setting the attribute ``return_type`` to "components" in the ``load_model()`` API.

.. attention::
    Not all model types are compatible with the pipeline API constructor via component elements. Incompatible models will raise an
    ``MLflowException`` error stating that the model is missing the `name_or_path` attribute. In
    the event that this occurs, please construct the model directly via the ``transformers.pipeline(<repo name>)`` API and save the pipeline object directly.

.. code-block:: python

    import mlflow
    import transformers

    task = "text-classification"
    architecture = "distilbert-base-uncased-finetuned-sst-2-english"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

    # Define the components of the model in a dictionary
    transformers_model = {"model": model, "tokenizer": tokenizer}

    # Log the model components
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=transformers_model,
            artifact_path="text_classifier",
            task=task,
        )

    # Load the components as a pipeline
    loaded_pipeline = mlflow.transformers.load_model(
        model_info.model_uri, return_type="pipeline"
    )

    print(type(loaded_pipeline).__name__)
    # >> TextClassificationPipeline

    loaded_pipeline(["MLflow is awesome!", "Transformers is a great library!"])

    # >> [{'label': 'POSITIVE', 'score': 0.9998478889465332},
    # >>  {'label': 'POSITIVE', 'score': 0.9998030066490173}]


Saving a pipeline and loading components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some use cases can benefit from the simplicity of defining a solution as a pipeline, but need the component-level access for performing a micro-services based deployment strategy
where pre / post-processing is performed on containers that do not house the model itself. For this paradigm, a pipeline can be loaded as its constituent parts, as shown below.

.. code-block:: python

    import transformers
    import mlflow

    translation_pipeline = transformers.pipeline(
        task="translation_en_to_fr",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained(
            "t5-small", model_max_length=100
        ),
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=translation_pipeline,
            artifact_path="french_translator",
        )

    translation_components = mlflow.transformers.load_model(
        model_info.model_uri, return_type="components"
    )

    for key, value in translation_components.items():
        print(f"{key} -> {type(value).__name__}")

    # >> task -> str
    # >> model -> T5ForConditionalGeneration
    # >> tokenizer -> T5TokenizerFast

    response = translation_pipeline("MLflow is great!")

    print(response)

    # >> [{'translation_text': 'MLflow est formidable!'}]

    reconstructed_pipeline = transformers.pipeline(**translation_components)

    reconstructed_response = reconstructed_pipeline(
        "transformers makes using Deep Learning models easy and fun!"
    )

    print(reconstructed_response)

    # >> [{'translation_text': "Les transformateurs rendent l'utilisation de modèles Deep Learning facile et amusante!"}]



Automatic Metadata and ModelCard logging
----------------------------------------

In order to provide as much information as possible for saved models, the ``transformers`` flavor will automatically fetch the ``ModelCard`` for any model or pipeline that
is saved that has a stored card on the HuggingFace hub. This card will be logged as part of the model artifact, viewable at the same directory level as the ``MLmodel`` file and
the stored model object.

In addition to the ``ModelCard``, the components that comprise any Pipeline (or the individual components if saving a dictionary of named components) will have their source types
stored. The model type, pipeline type, task, and classes of any supplementary component (such as a ``Tokenizer`` or ``ImageProcessor``) will be stored in the ``MLmodel`` file as well.

In order to preserve any attached legal requirements to the usage of any  model that is hosted on the huggingface hub, a "best effort" attempt
is made when logging a transformers model to retrieve and persist any license information. A file will be generated (``LICENSE.txt``) within the root of
the model directory. Within this file you will either find a copy of a declared license, the name of a common license type that applies to the model's use (i.e., 'apache-2.0', 'mit'),
or, in the event that license information was never submitted to the huggingface hub when uploading a model repository, a link to the repository for you to use
in order to determine what restrictions exist regarding the use of the model.

.. note::
  Model license information was introduced in **MLflow 2.10.0**. Previous versions do not include license information for models.

Automatic Signature inference
-----------------------------

For pipelines that support ``pyfunc``, there are 3 means of attaching a model signature to the ``MLmodel`` file.

* Provide a model signature explicitly via setting a valid ``ModelSignature`` to the ``signature`` attribute. This can be generated via the helper utility :py:func:`mlflow.transformers.generate_signature_output()`

* Provide an ``input_example``. The signature will be inferred and validated that it matches the appropriate input type. The output type will be validated by performing inference automatically (if the model is a ``pyfunc`` supported type).

* Do nothing. The ``transformers`` flavor will automatically apply the appropriate general signature that the pipeline type supports (only for a single-entity; collections will not be inferred).


Scale Inference with Overriding Pytorch dtype
---------------------------------------------

A common configuration for lowering the total memory pressure for pytorch models within ``transformers`` pipelines is to modify the
processing data type. This is achieved through setting the ``torch_dtype`` argument when creating a ``Pipeline``.
For a full reference of these tunable arguments for configuration of pipelines, see the `training docs <https://huggingface.co/docs/transformers/v4.28.1/en/perf_train_gpu_one#floating-data-types>`_ .

.. note:: This feature does not exist in versions of ``transformers`` < 4.26.x

In order to apply these configurations to a saved or logged run, there are two options:

* Save a pipeline with the `torch_dtype` argument set to the encoding type of your choice.

Example:

.. code-block:: python

    import transformers
    import torch
    import mlflow

    task = "translation_en_to_fr"

    my_pipeline = transformers.pipeline(
        task=task,
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained(
            "t5-small", model_max_length=100
        ),
        framework="pt",
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=my_pipeline,
            artifact_path="my_pipeline",
            torch_dtype=torch.bfloat16,
        )

    # Illustrate that the torch data type is recorded in the flavor configuration
    print(model_info.flavors["transformers"])


Result:

.. code-block:: bash

    {'transformers_version': '4.28.1',
     'code': None,
     'task': 'translation_en_to_fr',
     'instance_type': 'TranslationPipeline',
     'source_model_name': 't5-small',
     'pipeline_model_type': 'T5ForConditionalGeneration',
     'framework': 'pt',
     'torch_dtype': 'torch.bfloat16',
     'tokenizer_type': 'T5TokenizerFast',
     'components': ['tokenizer'],
     'pipeline': 'pipeline'}


* Specify the `torch_dtype` argument when loading the model to override any values set during logging or saving.

Example:

.. code-block:: python

    import transformers
    import torch
    import mlflow

    task = "translation_en_to_fr"

    my_pipeline = transformers.pipeline(
        task=task,
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained(
            "t5-small", model_max_length=100
        ),
        framework="pt",
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=my_pipeline,
            artifact_path="my_pipeline",
            torch_dtype=torch.bfloat16,
        )

    loaded_pipeline = mlflow.transformers.load_model(
        model_info.model_uri, return_type="pipeline", torch_dtype=torch.float64
    )

    print(loaded_pipeline.torch_dtype)


Result:

.. code-block:: bash

    torch.float64

.. note:: MLflow 2.12.1 slightly changed the ``torch_dtype`` extraction logic.
    Previously it depended on the ``torch_dtype`` attribute of the pipeline instance, but now it is extracted from the underlying model via ``dtype`` property. This enables MLflow to capture the dtype change of the model after pipeline instantiation.


.. note:: Logging or saving a model in 'components' mode (using a dictionary to declare components) does not support setting the data type for a constructed pipeline.
    If you need to override the default behavior of how data is encoded, please save or log a `pipeline` object.

.. note:: Overriding the data type for a pipeline when loading as a :ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>` is not supported.
    The value set for ``torch_dtype`` during ``save_model()`` or ``log_model()`` will persist when loading as `pyfunc`.

Input Data Types for Audio Pipelines
------------------------------------

Note that passing raw data to an audio pipeline (raw bytes) requires two separate elements of the same effective library.
In order to use the bitrate transposition and conversion of the audio bytes data into numpy nd.array format, the library `ffmpeg` is required.
Installing this package directly from pypi (`pip install ffmpeg`) does not install the underlying `c` dll's that are required to make `ffmpeg` function.
Please consult with the documentation at `the ffmpeg website <https://ffmpeg.org/download.html>`_ for guidance on your given operating system.

The Audio Pipeline types, when loaded as a :ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>` have three input types available:

* ``str``

The string input type is meant for blob references (uri locations) that are accessible to the instance of the ``pyfunc`` model.
This input mode is useful when doing large batch processing of audio inference in Spark due to the inherent limitations of handling large ``bytes``
data in ``Spark`` ``DataFrames``. Ensure that you have ``ffmpeg`` installed in the environment that the ``pyfunc`` model is running in order
to use ``str`` input uri-based inference. If this package is not properly installed (both from ``pypi`` and from the ``ffmpeg`` binaries), an Exception
will be thrown at inference time.

.. warning:: If using a uri (`str`) as an input type for a `pyfunc` model that you are intending to host for realtime inference through the `MLflow Model Server`,
    you *must* specify a custom model signature when logging or saving the model.
    The default signature input value type of ``bytes`` will, in `MLflow Model serving`, force the conversion of the uri string to ``bytes``, which will cause an Exception
    to be thrown from the serving process stating that the soundfile is corrupt.

An example of specifying an appropriate uri-based input model signature for an audio model is shown below:

.. code-block:: python

    from mlflow.models import infer_signature
    from mlflow.transformers import generate_signature_output

    url = "https://www.mywebsite.com/sound/files/for/transcription/file111.mp3"
    signature = infer_signature(url, generate_signature_output(my_audio_pipeline, url))
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=my_audio_pipeline,
            artifact_path="my_transcriber",
            signature=signature,
        )


* ``bytes``

This is the default serialization format of audio files. It is the easiest format to utilize due to the fact that
Pipeline implementations will automatically convert the audio bitrate from the file with the use of ``ffmpeg`` (a required dependency if using this format) to the bitrate required by the underlying model within the `Pipeline`.
When using the ``pyfunc`` representation of the pipeline directly (not through serving), the sound file can be passed directly as ``bytes`` without any
modification. When used through serving, the ``bytes`` data *must be* base64 encoded.

* ``np.ndarray``

This input format requires that both the bitrate has been set prior to conversion to ``numpy.ndarray`` (i.e., through the use of a package like
``librosa`` or ``pydub``) and that the model has been saved with a signature that uses the ``np.ndarray`` format for the input.

.. note:: Audio models being used for serving that intend to utilize pre-formatted audio in ``np.ndarray`` format
    must have the model saved with a signature configuration that reflects this schema. Failure to do so will result in type casting errors due to the default signature for
    audio transformers pipelines being set as expecting ``binary`` (``bytes``) data. The serving endpoint cannot accept a union of types, so a particular model instance must choose one
    or the other as an allowed input type.

.. _transformers-save-pretrained-guide:

Storage-Efficient Model Logging with ``save_pretrained`` Option
---------------------------------------------------------------

.. warning::

    The ``save_pretrained`` argument is only available in MLflow 2.11.0 and above, and still in experimental stage. The API and behavior may change in future releases. Moreover, this feature is intended for advanced users who are familiar with Transformers and MLflow, understanding :ref:`the potential risks <caveats-of-save-pretrained>` of using this feature.

Avoiding Redundant Model Copy by Setting ``save_pretrained=False``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, when MLflow logs an ML model, it saves a copy of the model weight to the artifact store.
However, this is not optimal when you use a pretrained model from HuggingFace Hub and have no intention of fine-tuning or otherwise manipulating the model or its weights before logging it. For this very common case, copying the (typically very large) model weights is redundant while developing prompts, testing inference parameters, and otherwise is little more than an unnecessary waste of storage space.

To address this issue, MLflow 2.11.0 introduced a new argument ``save_pretrained`` in the :py:func:`mlflow.transformers.save_model()` and :py:func:`mlflow.transformers.log_model()` APIs. When with argument is set to ``False``, MLflow will forego saving the pretrained model weights, opting instead to store a reference to the underlying repository entry on the HuggingFace Hub; specifically, the  repository name and the unique commit hash of the model weights are stored when your components or pipeline are logged. When loading back such a *refernce-only* model, MLflow will check the repository name and commit hash from the saved metadata, and either download the model weight from the HuggingFace Hub or use the locally cached model from your HuggingFace local cache directory.

A good analogy for this feature is the comparison between a file *copy* and a *symlink* operation. The default behavior for the transformers flavor is to perform a copy, materializing the model weight files in your artifact store that is associated with the run that the model is logged to. By setting ``save_pretrained=False``, MLflow will log a link to the HuggingFace Hub repository, effectively building in symlink functionality to the run. This will save storage space and reduce the logging latency significantly, particularly for large models like LLMs.

.. note:

    By default, the ``save_pretrained`` argument is set to ``True`` and doesn't change the model saving behavior.

Example Usage
^^^^^^^^^^^^^

Here is the example of using ``save_pretrained`` argument for logging a model

.. code-block:: python

    import transformers

    pipeline = transformers.pipeline(
        task="text-generation", model="databricks/dolly-v2-7b", torch_dtype="torch.float16"
    )

    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="dolly",
            save_pretrained=False,
        )

In the above example, MLflow will not save a copy of the **Dolly-v2-7B** model's weights and will instead log the following metadata as a reference to the HuggingFace Hub model. This will save roughly 15GB of storage space and reduce the logging latency significantly as well for each run that you initiate during development.
```
source_model_name: "databricks/dolly-v2-7b"
source_model_revision: "d632f0c8b75b1ae5b26b250d25bfba4e99cb7c6f"
```

.. _caveats-of-save-pretrained:

Caveats of Reference-Only Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the ``save_pretrained`` argument is useful for saving storage space and reducing logging latency, it has the following caveats to be aware of:

* **Change in Model Unavailability**: If you are using a model from other users' repository, the model may be deleted or become private in the HuggingFace Hub. In such cases, MLflow cannot load the model back. For production use cases, it is recommended to save the copy model weight to the artifact store prior to moving from development or staging to production for your model.

* **HuggingFace Hub Access**: Downloading a model from the HuggingFace Hub might be slow or unstable due to the network condition or the HuggingFace Hub service status. MLflow doesn't provide any retry mechanism or robust error handling for the model downloading. As such, you should not rely on this functionality for your final production-candidate run.

* **Limited Databricks Integration**: If you are using Databricks, be aware that the model saved with `save_pretrained=False` cannot be registered to the legacy `Workspace Model Registry <https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/workspace-model-registry.html>`_. If you want to register the reference-only Transformer model, please use `Unity Catalog <https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html>`_ instead, or download the model weight in advance using :py:func:`mlflow.transformers.persist_pretrained_model()` API as described in the next section.

.. _persist-pretrained-guide:

Persist the Model Weight to the Existing Reference-Only Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to update the reference-only model to an instance that contains the model weight, you can use the :py:func:`mlflow.transformers.persist_pretrained_model()` API. This API will download the model weight from the HuggingFace Hub, save it to the artifact location, and update the metadata of the given reference-only model. After this operation, the model will be equivalent to the one saved with `save_pretrained=True` and be ready for the production use.

.. tip::

    The :py:func:`mlflow.transformers.persist_pretrained_model()` API **does NOT require re-logging a model** but efficiently update the existing model and metadata in-place.

.. code-block:: python

    import mlflow
    import transformers

    pipeline = transformers.pipeline(
        task="text-generation", model="databricks/dolly-v2-7b", torch_dtype="torch.float16"
    )

    # Save the reference-only Transformer model
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="dolly",
            save_pretrained=False,
        )

    # Model weight is not saved to the artifact store
    assert not os.path.exists(model_info.artifact_path + "/model")

    # This will download the model weight from the HuggingFace Hub and save it
    # to the artifact location
    mlflow.transformers.persist_pretrained_model(model_info.model_uri)

    assert os.path.exists(model_info.artifact_path + "/model")


PEFT Models in MLflow Transformers flavor
-----------------------------------------

.. warning::


    The PEFT model is supported in MLflow 2.11.0 and above and is still in the experimental stage. The API and behavior may change in future releases. Moreover, the `PEFT <https://huggingface.co/docs/peft/en/index>`_ library is under active development, so not all features
    and adapter types might be supported in MLflow.

`PEFT <https://huggingface.co/docs/peft/en/index>`_ is a library developed by HuggingFace🤗, that provides various optimization methods for pretrained models available on the HuggingFace Hub. With PEFT, you can easily apply various optimization techniques like LoRA and QLoRA to reduce the cost of fine-tuning Transformers models.

For example, `LoRA (Low-Rank Adaptation) <https://huggingface.co/docs/peft/main/en/conceptual_guides/lora>`_ is a method that approximate the weight updates of fine-tuning process with two smaller matrices through low-rank decomposition. LoRA typically shrinks the number of parameters to train to only 0.01% ~ a few % of the full model fine-tuning (depending on the configuration), which significantly accelerates the fine-tuning process and reduces the memory footprint, such that you can even `train a Mistral/Llama2 7B model on a single Nvidia A10G GPU in an hour <../tutorials/fine-tuning/transformers-peft.html>`_.
By using PEFT, you can apply LoRA to your Transformers model with only a few lines of code:

.. code-block:: python

    from peft import LoraConfig, get_peft_model

    base_model = AutoModelForCausalLM.from_pretrained(...)
    lora_config = LoraConfig(...)
    peft_model = get_peft_model(base_model, lora_config)


In MLflow 2.11.0, we introduced support for tracking PEFT models in the MLflow Transformers flavor. You can log and load PEFT models using the same APIs as other Transformers models, such as :py:func:`mlflow.transformers.log_model()` and :py:func:`mlflow.transformers.load_model()`.

.. code-block:: python

    import mlflow
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "databricks/dolly-v2-7b"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    peft_config = LoraConfig(...)
    peft_model = get_peft_model(base_model, peft_config)

    with mlflow.start_run():
        # Your training code here
        ...

        # Log the PEFT model
        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": peft_model,
                "tokenizer": tokenizer,
            },
            artifact_path="peft_model",
        )

    # Load the PEFT model
    loaded_model = mlflow.transformers.load_model(model_info.model_uri)

PEFT Models in MLflow Tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Check out the tutorial `Fine-Tuning Open-Source LLM using QLoRA with MLflow and PEFT <../tutorials/fine-tuning/transformers-peft.html>`_ for a more in-depth guide on how to use PEFT with MLflow,

Format of Saved PEFT Model
^^^^^^^^^^^^^^^^^^^^^^^^^^
When saving PEFT models, MLflow only saves the PEFT adapter and the configuration, but not the base model's weights. This is the same behavior as the Transformer's `save_pretrained() <https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained>`_ method and is highly efficient in terms of storage space and logging latency. One difference is that MLflow will also save the HuggingFace Hub repository name and version for the base model in the model metadata, so that it can load the same base model when loading the PEFT model. Concretely, the following artifacts are saved in MLflow for PEFT models:

* The PEFT adapter weight under the ``/peft`` directory.
* The PEFT configuration as a JSON file under the ``/peft`` directory.
* The HuggingFace Hub repository name and commit hash for the base model in the ``MLModel`` metadata file.

Limitations of PEFT Models in MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since the model saving/loading behavior for PEFT models is similar to that of ``save_pretrained=False``, :ref:`the same caveats <caveats-of-save-pretrained>` apply to PEFT models. For example, the base model weight may be deleted or become private in the HuggingFace Hub, and PEFT models cannot be registered to the legacy Databricks Workspace Model Registry.

To save the base model weight for PEFT models, you can use the :py:func:`mlflow.transformers.persist_pretrained_model()` API. This will download the base model weight from the HuggingFace Hub and save it to the artifact location, updating the metadata of the given PEFT model. Please refer to :ref:`this section <persist-pretrained-guide>` for the detailed usage of this API.
