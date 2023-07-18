:orphan:

.. _gateway_databricks_model_serving_completions_example:

======================================================================
Databricks AI Gateway: Completions example with MPT-7B-Instruct
======================================================================
This example demonstrates how to use the Databricks AI Gateway
to deploy and query the open source
`MPT-7B-Instruct </Users/corey.zumar/mlflow/docs/source/gateway/databricks_model_serving.rst>`_ 
model.

This section assumes that you have installed the AI Gateway client in your Databricks
notebook. For reference, see :ref:`gateway-quickstart`.

For questions and information about using other open source LLMs with the AI Gateway,
please contact your Databricks representative. In particular, support for
`MPT-30B <https://huggingface.co/mosaicml/mpt-30b>`_ and
`Falcon 40B <https://huggingface.co/tiiuae/falcon-40b>`_ is coming soon.

Step 1: Download the MPT-7B-Instruct model snapshot
---------------------------------------------------
The first step is to download the MPT-7B-Instruct model snapshot from Hugging Face by running
the following code in your Databricks notebook:

.. code-block:: python

    from huggingface_hub import snapshot_download
    snapshot_location = snapshot_download(
        repo_id="mosaicml/mpt-7b-instruct",
        local_dir='mpt-7b'
    )

The snapshot may take several minutes to download.

Step 2: Define an MLflow PyFunc model wrapper for MPT-7B-Instruct 
-----------------------------------------------------------------
Next, define an `MLflow PyFunc model wrapper <https://mlflow.org/docs/latest/models.html#custom-python-models>`_
that makes MPT-7B-Instruct available for real-time serving with Databricks Model Serving. To do
this, run the following code in your Databricks notebook:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import transformers
    import mlflow
    import torch

    class MPT(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            """
            This method initializes the tokenizer and language model
            using the specified model repository.
            """
            # Initialize tokenizer and language model
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
              context.artifacts['repository'], padding_side="left")

            config = transformers.AutoConfig.from_pretrained(
                context.artifacts['repository'], 
                trust_remote_code=True
            )
            # support for flast-attn and openai-triton is coming soon
            config.attn_config['attn_impl'] = 'triton'
            
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                context.artifacts['repository'], 
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            self.model.to(device='cuda')
            
            self.model.eval()

        def _build_prompt(self, instruction):
            """
            This method generates the prompt for the model.
            """
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
            )

            return f"""{INTRO_BLURB}
            {INSTRUCTION_KEY}
            {instruction}
            {RESPONSE_KEY}
            """

        def predict(self, context, model_input):
            """
            This method generates prediction for the given input.
            """
            prompt = model_input["prompt"][0]
            temperature = model_input.get("temperature", [1.0])[0]
            max_tokens = model_input.get("max_tokens", [100])[0]

            # Build the prompt
            prompt = self._build_prompt(prompt)

            # Encode the input and generate prediction
            encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
            output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_tokens)
        
            # Decode the prediction to text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Removing the prompt from the generated text
            prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
            generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

            return generated_response


.. _compatible_signature_mpt7b:

Step 3: Define a compatible MLflow Model Signature 
--------------------------------------------------
Before the snapshot and model wrapper can be used with Databricks Model Serving, a compatible
:ref:`MLflow Model Signature <model-signature>` must also be defined.

The MLflow PyFunc wrapper defined in the previous step accepts three of the
AI Gateway :ref:`standard query parameters <standard_query_parameters>`: ``prompt``,
``temperature``, and ``max_tokens``.

Running the following code in your Databricks notebook will define a compatible schema that
includes these parameters:

.. code-block:: python

    from mlflow.models.signature import ModelSignature
    from mlflow.types import DataType, Schema, ColSpec

    # Define input and output schema
    input_schema = Schema([
        ColSpec(DataType.string, "prompt"), 
        ColSpec(DataType.double, "temperature"), 
        ColSpec(DataType.long, "max_tokens")
    ])
    output_schema = Schema([ColSpec(DataType.string)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)


Step 4: Register the wrapper and snapshot as an MLflow Model 
------------------------------------------------------------
Now that the snapshot has been loaded and the MLflow PyFunc wrapper has been defined with a
compatible :ref:`MLflow Model Signature <model-signature>`, the next
step is to register these assets with the MLflow Model Registry by running the following code:


.. code-block:: python

    with mlflow.start_run() as run:  
        mlflow.pyfunc.log_model(
            "model",
            python_model=MPT(),
            artifacts={"repository": snapshot_location},
            pip_requirements=["torch", "transformers", "accelerate", "einops", "sentencepiece"],
            input_example=input_example,
            signature=signature,
            registered_model_name="mpt-7b-instruct",
            await_registration_for=1200
        )

        # Define input example
        input_example=pd.DataFrame({
            "prompt": ["What is machine learning?"], 
            "temperature": [0.5],
            "max_tokens": [100]
        })

This will create an :ref:`MLflow Registered Model <registry>` called ``mpt-7b-instruct``
and create a new :ref:`MLflow Model Version <registry>` - version ``1`` - of that
registered model with the MPT-7B-Instruct snapshot and wrapper code defined above. You can reference
this model using the following URI: ``models:/mpt-7b-instruct/1``.

.. note::
    Due to its size, the model may take up to 20 minutes to register. On average, the upload time
    should be about 10 minutes.

For more information about model logging, see :py:func:`mlflow.pyfunc.log_model`.

Step 5: Create a Databricks Model Serving endpoint for MPT-7B-Instruct 
----------------------------------------------------------------------
Next, follow the instructions in
https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html
to create a Databricks Model Serving endpoint using the
:ref:`MLflow Model Version <registry>` that you created in the previous step.

.. important::
   We recommend specifying ``"Large"`` for the *compute* (also called *workload*) field when
   creating your Databricks Model Serving endpoint due to the large size of MPT-7B-Instruct.

   For improved performance, please contact your Databricks representative about support for GPUs
   in Databricks Model Serving.


Step 6: Create an AI Gateway completions route
----------------------------------------------
Now that a Databricks Model Serving endpoint has been created, the next step is to create a
completions route in the AI Gateway that forwards requests to the endpoint by running the
following code:

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, create_route

    set_gateway_uri("databricks")

    create_route(
        name="completions-mpt-7b-instruct",
        route_type="llm/v1/completions",
        model={
            "name": "<your_endpoint_name>",
            "provider": "databricks-model-serving",
            "config": {
                "databricks_api_token": "<your_databricks_access_token>"
                "databricks_workspace_url": "<your_databricks_workspace_url>"
            }
        }
    )

**Note: Before running the code, replace the following placeholder values**: 

* Replace ``<your_endpoint_name>`` with the name of the Databricks Model Serving endpoint that
  you created in the previous step.

* Replace ``<your_databricks_access_token>`` with a Databricks access token
  corresponding to a user or service principal that has **Can Query** access to the Databricks
  Model Serving endpoint.

* Replace ``<your_databricks_workspace_url>`` with the URL of the Databricks workspace
  containing the Databricks Model Serving endpoint that you created in the previous step.

For additional information about configuring the ``databricks-model-serving`` provider, see
:ref:`databricks_serving_provider_fields`.


Step 7: Query the AI Gateway completions route 
----------------------------------------------
Finally, now that the AI Gateway route has been created, the last step is to query it. The
following example code specifies the ``prompt``, ``temperature``, and ``max_tokens``
:ref:`standard query parameters <standard_query_parameters>` defined in the model's signature
from :ref:`compatible_signature_mpt7b`:

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        route="completions-mpt-7b-instruct",
        data={
            "prompt": "What is MLflow?",
            "temperature": 0.3,
            "max_tokens": 200
        }
    )
    print(response)

The structure of the ``response`` will be as follows
(the actual content and token values will likely be different):

.. code-block:: python

    {
         "candidates": [
           {
             "text": "MLflow is an open source machine learning platform...",
             "metadata": {}
           }
        ],
        "metadata": {
            "model": "<your_endpoint_name>",
            "route_type": "llm/v1/completions"
        }
    }

Step 8: Use the AI Gateway Route for model development
------------------------------------------------------

Now that you have created an AI Gateway route with MPT-7B-Instruct, you can create
MLflow Models that query this route to build application-specific logic using techniques
like prompt engineering. For more information, see
:ref:`AI Gateway and MLflow Models <gateway_mlflow_models>`.
