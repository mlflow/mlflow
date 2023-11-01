.. _llm-tracking:

MLflow's LLM Tracking Capabilities
==================================

MLflow's LLM Tracking system is an enhancement to the existing MLflow Tracking system, offerring additional capabilities for monitoring, 
managing, and interpreting interactions with Large Language Models (LLMs). 

At its core, MLflow's LLM suite builds upon the standard logging capabilities familiar to professionals working with traditional 
Machine Learning (ML) and Deep Learning (DL). However, it introduces distinct features tailored for the unique intricacies of LLMs. 

One such standout feature is the introduction of "prompts" – the queries or inputs directed towards an LLM – and the subsequent data 
the model generates in response. While MLflow's offerings for other model types typically exclude built-in mechanisms for preserving 
inference results, LLMs necessitate this due to their dynamic and generative nature. Recognizing this, MLflow introduces the term 
'predictions' alongside the existing tracking components of **artifacts**, **parameters**, **tags**, and **metrics**, ensuring comprehensive 
lineage and quality tracking for text-generating models.

.. _llm-tracking-introduction:

Parameters
-------------
Parameters are key-value pairs which are used typcially used to log lightweight 
metadata. Both the keys and values are stored as strings. Use 
:py:func:`mlflow.log_param` to log a single parameter 
:py:func:`mlflow.log_params` to log multiple parameters.

.. code-section::
    .. code-block:: python 

        # pip install mlflow langchain
        import mlflow
        from typing import Dict
        from langchain.prompts import PromptTemplate


        def log_model_run(prompt_template: str, prompt_values: Dict) -> str:
            with mlflow.start_run() as run:
                # Log the prompt template
                mlflow.log_param("prompt_template", prompt_template)

                # Log the prompt values
                mlflow.log_params(prompt_values)

                # Generate the prompt using the template and values
                prompt = PromptTemplate.from_template(prompt_template).format(**prompt_values)

                # Placeholder for model execution code
                # ...

                # Return the run ID for later reference
                return run.info.run_id


        # Example usage
        prompt_template = "Should we send {animal} to space using a {mode_of_transportation}?"
        prompt_values = {"animal": "squirrels", "mode_of_transportation": "weather baloon"}

        # Run the model with the given prompt and values
        run_id = log_model_run(prompt_template, prompt_values)

        # Retrieve and display parameters
        params_dict = mlflow.get_run(run_id).data.params
        for k, v in params_dict.items():
            print(f"Loggged parameter of type {type(v)} with key {k}: {v}")

Metrics
-------
Metrics are quantitative measures, often numeric, that give insights into the 
performance, accuracy, or any other measurable aspect of the LLM interaction. 
Metrics are dynamic and can be updated as the run progresses, offering a 
real-time or post-process insight into the model's behavior. Use 
:py:func:`mlflow.log_metric` to log a single metric and 
:py:func:`mlflow.log_metrics` to log multiple metrics.

.. code-section::
    .. code-block:: python 

        import mlflow
        import openai
        import pandas as pd

        # Prepare evaluation data
        eval_data = pd.DataFrame(
            {
                "inputs": [
                    "Are squirrels astronauts?",
                    "From a logistics perspective, do you think a single weather balloon could get a squirrel to space?",
                ],
                "ground_truth": [
                    "Squirrels are not astronauts as they have not been trained or equipped for space travel.",
                    "It is unlikely that a single weather balloon could safely carry a squirrel to space due to the complexities involved in space travel.",
                ],
            }
        )

        # Start an MLflow run
        with mlflow.start_run() as run:
            system_prompt = "Answer the following question in two sentences"

            # Log the OpenAI model as an MLflow model
            logged_model = mlflow.openai.log_model(
                model="gpt-4",
                task=openai.ChatCompletion,
                artifact_path="model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "{question}"},
                ],
            )

            # Evaluate the logged model
            results = mlflow.evaluate(
                logged_model.model_uri, eval_data, targets="ground_truth", model_type="question-answering"
            )

            # Explictely log ari_grade_level (this is already logged by default)
            ari_grade_level = results.metrics.get("ari_grade_level/v1/mean")
            if ari_grade_level is not None:
                mlflow.log_metric("manually logged ari grade level", ari_grade_level)

            # Get the current run ID
            run_id = run.info.run_id

        # Query metrics using the run ID with the fluent API
        metrics = mlflow.get_run(run_id).data.metrics

        # Display evaluation metrics
        print(f"\nLogged metrics of type {type(metrics)}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
  
Tables
-----------
Table logging refers to storing a set of complex information in the form of a 
dict or pandas DataFrame as a JSON artifact. With LLMs, typically you'd use 
log_table to store things like inputs to your model, model responses, evaluation
metrics, and anything else associated with a given run. With everything located
in one artifact, referencing related information becomes much easier. Use 
:py:func:`mlflow.log_table` to log a single dict or pandas DataFrame.

.. code-section::
    .. code-block:: python 

        # pip install mlflow
        import time
        import mlflow

        ARTIFACT_NAME = "important_information.json"


        def get_current_time_str():
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


        for _ in range(3):
            with mlflow.start_run() as run:
                table_dict = {
                    "Prompts": ["I am a squirrel and don't have a watch", "What time is it?"],
                    "Response": get_current_time_str(),
                    "We used an LLM?": False,
                }
                mlflow.log_table(data=table_dict, artifact_file=ARTIFACT_NAME)

                run_id = run.info.run_id

        loaded_table = mlflow.load_table(
            artifact_file=ARTIFACT_NAME,
            run_ids=[run_id],
        )

        print(f"\nLoaded table is of type: {type(loaded_table)}:")
        print(loaded_table.to_markdown())

Artifacts
----------
Artifacts are files that are not supported by the above functionality. Some 
examples that relate to LLMs are serialized models (e.g. an `openai` model),
images, data visualizations, structured data files (e.g., a 
`Parquet <https://parquet.apache.org/>`_ file), and much more. Use 
:py:func:`mlflow.log_artifact` to log artifacts.

.. code-section::
    .. code-block:: python 

        # pip install mlflow transformers
        import mlflow
        from transformers import AutoModel, AutoTokenizer

        MODEL_NAME = "bert-base-uncased"
        OUTPUT_DIR = "./local_model_directory"

        with mlflow.start_run():
            # Download the specified model and tokenizer from Hugging Face
            model = AutoModel.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            # Save the model and tokenizer to the specified output directory
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Model and tokenizer have been saved to {OUTPUT_DIR}")

            # Log model and tokenizer as artifacts
            mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")
