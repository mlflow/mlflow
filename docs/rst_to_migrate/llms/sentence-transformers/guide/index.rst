Sentence Transformers within MLflow
====================================

.. attention::
    The ``sentence_transformers`` flavor is in active development and is marked as Experimental. Public APIs may change and new features are
    subject to be added as additional functionality is brought to the flavor.

The ``sentence_transformers`` model flavor enables logging of
`sentence-transformers models <https://www.sbert.net/>`_ in MLflow format via
the :py:func:`mlflow.sentence_transformers.save_model()` and :py:func:`mlflow.sentence_transformers.log_model()` functions. Using these
functions also adds the ``python_function`` flavor to the MLflow Models, enabling the model to be
interpreted as a generic Python function for inference via :py:func:`mlflow.pyfunc.load_model()`.
Additionally, :py:func:`mlflow.sentence_transformers.load_model()` can be used to load a saved or logged MLflow
Model with the ``sentence_transformers`` flavor in the native sentence-transformers format.

Tutorials for Sentence Transformers
-----------------------------------

Looking to get right in to some usable examples and tutorials that show how to leverage this library with MLflow? 

.. raw:: html

    <a href="../index.html#getting-started-with-the-mlflow-sentence-transformers-flavor-tutorials-and-guides" class="download-btn">See the Tutorials</a>

Input and Output Types for PyFunc
---------------------------------

The ``sentence_transformers`` :ref:`python_function (pyfunc) model flavor <pyfunc-model-flavor>` standardizes
the process of embedding sentences and computing semantic similarity. This standardization allows for serving
and batch inference by adapting the required data structures for ``sentence_transformers`` into formats compatible with JSON serialization and casting to Pandas DataFrames.

.. note::
    The ``sentence_transformers`` flavor supports various models for tasks such as embedding generation, semantic similarity, and paraphrase mining. The specific input and output types will depend on the model and task being performed.

Saving and Logging Sentence Transformers Models
-----------------------------------------------

You can save and log sentence-transformers models in MLflow. Here's an example of both saving and logging a model:

.. code-block:: python

    import mlflow
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("model_name")

    # Saving the model
    mlflow.sentence_transformers.save_model(model=model, path="path/to/save/directory")

    # Logging the model
    with mlflow.start_run():
        mlflow.sentence_transformers.log_model(
            sentence_transformers_model=model, artifact_path="model_artifact_path"
        )

Saving Sentence Transformers Models with an OpenAI-Compatible Inference Interface
---------------------------------------------------------------------------------

.. note::

    This feature is only available in MLflow 2.11.0 and above.

MLflow's ``sentence_transformers`` flavor allows you to pass in the ``task`` param with the string value ``"llm/v1/embeddings"`` 
when saving a model with :py:func:`mlflow.sentence_transformers.save_model()` and :py:func:`mlflow.sentence_transformers.log_model()`.

For example:

.. code-block:: python
    
    import mlflow
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    mlflow.sentence_transformers.save_model(
        model=model, path="path/to/save/directory", task="llm/v1/embeddings"
    )

When ``task`` is set as ``"llm/v1/embeddings"``, MLflow handles the following for you:

- Setting an embeddings compatible signature for the model
- Performing data pre- and post-processing to ensure the inputs and outputs conform to 
  the `Embeddings API spec <https://mlflow.org/docs/latest/llms/deployments/index.html#embeddings>`_, 
  which is compatible with OpenAI's API spec.

Note that these modifications only apply when the model is loaded with :py:func:`mlflow.pyfunc.load_model()` (e.g. when 
serving the model with the ``mlflow models serve`` CLI tool). If you want to load just the base pipeline, you can
always do so via :py:func:`mlflow.sentence_transformers.load_model()`.

Aside from the ``sentence-transformers`` flavor, the ``transformers`` flavor also support OpenAI-compatible inference interface (``"llm/v1/chat"`` and ``"llm/v1/completions"``). Refer to 
`the Transformers flavor guide <https://mlflow.org/docs/latest/llms/transformers/guide/index.html#saving-transformer-pipelines-with-an-openai-compatible-inference-interface>`_ for more information.

Custom Python Function Implementation
-------------------------------------

In addition to using pre-built models, you can create custom Python functions with the `sentence_transformers` flavor. Here's an example of a custom 
implementation for comparing the similarity between text documents:

.. code-block:: python

    import mlflow
    from mlflow.pyfunc import PythonModel
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util


    class DocumentSimilarityModel(PythonModel):
        def load_context(self, context):
            """Load the model context for inference."""
            self.model = SentenceTransformer.load(context.artifacts["model_path"])

        def predict(self, context, model_input):
            """Predict method for comparing similarity between documents."""
            if isinstance(model_input, pd.DataFrame) and model_input.shape[1] == 2:
                documents = model_input.values
            else:
                raise ValueError("Input must be a DataFrame with exactly two columns.")

            # Compute embeddings for each document separately
            embeddings1 = self.model.encode(documents[:, 0], convert_to_tensor=True)
            embeddings2 = self.model.encode(documents[:, 1], convert_to_tensor=True)

            # Calculate cosine similarity
            similarity_scores = util.cos_sim(embeddings1, embeddings2)

            return pd.DataFrame(similarity_scores.numpy(), columns=["similarity_score"])


    # Example model saving and loading
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model_path = "/tmp/sentence_transformers_model"
    model.save(model_path)

    # Example usage
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="document_similarity_model",
            python_model=DocumentSimilarityModel(),
            artifacts={"model_path": model_path},
        )

    loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    # Test prediction
    df = pd.DataFrame(
        {
            "doc1": ["Sentence Transformers is a wonderful package!"],
            "doc2": ["MLflow is pretty great too!"],
        }
    )

    result = loaded.predict(df)
    print(result)

Which will generate the similarity score for the documents passed, as shown below:

.. code-block:: bash

       similarity_score
    0          0.275423
