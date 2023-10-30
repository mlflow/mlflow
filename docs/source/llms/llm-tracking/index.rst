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

Introduction to LLM Tracking
----------------------------

The world of Large Language Models is vast, and as these models become more intricate and sophisticated, the need for a robust 
tracking system becomes paramount. MLflow's LLM Tracking is centered around the concept of *runs*. In essence, a run is a 
distinct execution or interaction with the LLM — whether it's a single query, a batch of prompts, or an entire fine-tuning session. 

Each run meticulously records:

- **Parameters**: Key-value pairs that detail the input parameters for the LLM. These could range from model-specific parameters like `top_k` and `temperature` to more generic ones. They provide context and configuration for each run. Parameters can be logged using both :py:func:`mlflow.log_param` for individual entries and :py:func:`mlflow.log_params` for bulk logging.
  
- **Metrics**: These are quantitative measures, often numeric, that give insights into the performance, accuracy, or any other measurable aspect of the LLM interaction. Metrics are dynamic and can be updated as the run progresses, offering a real-time or post-process insight into the model's behavior. Logging of metrics is facilitated through :py:func:`mlflow.log_metric` and :py:func:`mlflow.log_metrics`.
  
- **Predictions**: To understand and evaluate LLM outputs, MLflow allows for the logging of predictions. This encompasses the prompts or inputs sent to the LLM and the outputs or responses received. For structured storage and easy retrieval, these predictions are stored as artifacts in CSV format, ensuring that each interaction is preserved in its entirety. This logging is achieved using the dedicated :py:func:`mlflow.log_table`.
  
- **Artifacts**: Beyond predictions, MLflow's LLM Tracking can store a myriad of output files, ranging from visualization images (e.g., PNGs), serialized models (e.g., an `openai` model), to structured data files (e.g., a `Parquet <https://parquet.apache.org/>`_ file). The :py:func:`mlflow.log_artifact` function is at the heart of this, allowing users to log and organize their artifacts with ease.

Furthermore, to provide structured organization and comparative analysis capabilities, runs can be grouped into *experiments*. 
These experiments act as containers, grouping related runs, and providing a higher level of organization. This organization ensures 
that related runs can be compared, analyzed, and managed as a cohesive unit.

.. _how-llm-data-is-captured:

Detailed Logging of LLM Interactions
------------------------------------

MLflow's LLM Tracking doesn't just record data — it offers structured logging mechanisms tailored to the needs of LLM interactions:

- **Parameters**: Logging parameters is straightforward. Whether you're logging a single parameter using :py:func:`mlflow.log_param` or multiple parameters simultaneously with :py:func:`mlflow.log_params`, MLflow ensures that every detail is captured.

- **Metrics**: Quantitative insights are crucial. Whether it's tracking the accuracy of a fine-tuned LLM or understanding its response time, metrics provide this insight. They can be logged individually via :py:func:`mlflow.log_metric` or in bulk using :py:func:`mlflow.log_metrics`.

- **Predictions**: Every interaction with an LLM yields a result — a prediction. Capturing this prediction, along with the inputs that led to it, is crucial. The :py:func:`mlflow.log_table` function is specifically designed for this, ensuring that both inputs and outputs are logged cohesively.

- **Artifacts**: Artifacts act as the tangible outputs of an LLM run. They can be images, models, or any other form of data. Logging them is seamless with :py:func:`mlflow.log_artifact`, which ensures that every piece of data, regardless of its format, is stored and linked to its respective run.

.. _storage-of-llm-data:

Structured Storage of LLM Tracking Data
---------------------------------------

Every piece of data, every parameter, metric, prediction, and artifact is not just logged — it's structured and stored as part of an 
MLflow Experiment run. This organization ensures data integrity, easy retrieval, and a structured approach to analyzing and understanding 
LLM interactions in the grand scheme of machine learning workflows.
