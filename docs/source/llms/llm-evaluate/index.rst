.. _llm-eval:

MLflow LLM Evaluate
===================

With the emerging of ChatGPT, LLMs have shown its power of text generation in various fields, such as 
question answering, translating and text summarization. Evaluating LLMs' performance is slightly different 
from traditional ML models, as very often there is no single ground truth to compare against. 
MLflow provides an API :py:func:`mlflow.evaluate()` to help evaluate your LLMs.

MLflow's LLM evaluation functionality consists of 3 main components:

1. **A model to evaluate**: it can be an MLflow ``pyfunc`` model, a URI pointing to one registered 
   MLflow model, or any python callable that represents your model, e.g, a HuggingFace text summarization pipeline. 
2. **Metrics**: the metrics to compute, LLM evaluate will use LLM metrics. 
3. **Evaluation data**: the data your model is evaluated at, it can be a pandas Dataframe, a python list, a 
   numpy array or an :py:func:`mlflow.data.dataset.Dataset` instance.

Full Notebook Guides and Examples
---------------------------------
If you're interested in thorough use-case oriented guides that showcase the simplicity and power of MLflow's evaluate 
functionality for LLMs, please navigate to the notebook collection below:

.. raw:: html

    <a href="notebooks/index.html" class="download-btn">View the Notebook Guides</a><br/>

Quickstart
----------

Below is a simple example that gives an quick overview of how MLflow LLM evaluation works. The example builds
a simple question-answering model by wrapping "openai/gpt-4" with custom prompt. You can paste it to
your IPython or local editor and execute it, and install missing dependencies as prompted. Running the code 
requires OpenAI API key, if you don't have an OpenAI key, you can set it up [here](https://platform.openai.com/account/api-keys).

.. code-block:: shell

    export OPENAI_API_KEY='your-api-key-here'

.. code-block:: python

    import mlflow
    import openai
    import os
    import pandas as pd
    from getpass import getpass

    eval_data = pd.DataFrame(
        {
            "inputs": [
                "What is MLflow?",
                "What is Spark?",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
                "lifecycle. It was developed by Databricks, a company that specializes in big data and "
                "machine learning solutions. MLflow is designed to address the challenges that data "
                "scientists and machine learning engineers face when developing, training, and deploying "
                "machine learning models.",
                "Apache Spark is an open-source, distributed computing system designed for big data "
                "processing and analytics. It was developed in response to limitations of the Hadoop "
                "MapReduce computing model, offering improvements in speed and ease of use. Spark "
                "provides libraries for various tasks such as data ingestion, processing, and analysis "
                "through its components like Spark SQL for structured data, Spark Streaming for "
                "real-time data processing, and MLlib for machine learning tasks",
            ],
        }
    )

    with mlflow.start_run() as run:
        system_prompt = "Answer the following question in two sentences"
        # Wrap "gpt-4" as an MLflow model.
        logged_model_info = mlflow.openai.log_model(
            model="gpt-4",
            task=openai.ChatCompletion,
            artifact_path="model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "{question}"},
            ],
        )

        # Use predefined question-answering metrics to evaluate our model.
        results = mlflow.evaluate(
            logged_model_info.model_uri,
            eval_data,
            targets="ground_truth",
            model_type="question-answering",
        )
        print(f"See aggregated evaluation results below: \n{results.metrics}")

        # Evaluation result for each data record is available in `results.tables`.
        eval_table = results.tables["eval_results_table"]
        print(f"See evaluation table below: \n{eval_table}")


LLM Evaluation Metrics
----------------------

There are two types of LLM evaluation metrics in MLflow:

1. Metrics relying on SaaS model (e.g., OpenAI) for scoring, e.g., :py:func:`mlflow.metrics.genai.answer_relevance`. These  
   metrics are created via :py:func:`mlflow.metrics.genai.make_genai_metric` method. For each data record, these metrics under the hood sends 
   one prompt consisting of the following information to the SaaS model, and extract the score from model response:

   * Metrics definition.
   * Metrics grading criteria.
   * Reference examples.
   * Input data/context.
   * Model output.
   * [optional] Ground truth.

   More details of how these fields are set can be found in the section "Create your Custom LLM-evaluation Metrics".

2. Function-based per-row metrics. These metrics calculate a score for each data record (row in terms of Pandas/Spark dataframe),
   based on certain functions, like Rouge (:py:func:`mlflow.metrics.rougeL`) or Flesch Kincaid (:py:func:`mlflow.metrics.flesch_kincaid_grade_level`). 
   These metrics are similar to traditional metrics.


Select Metrics to Evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to select metrics to evaluate your model:

* Use **default** metrics for pre-defined model types.
* Use a **custom** list of metrics.

.. _llm-eval-default-metrics:

Use Default Metrics for Pre-defined Model Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLflow LLM evaluation includes default collections of metrics for pre-selected tasks, e.g, "question-answering". Depending on the 
LLM use case that you are evaluating, these pre-defined collections can greatly simplify the process of running evaluations. To use
defaults metrics for pre-selected tasks, specify the ``model_type`` argument in :py:func:`mlflow.evaluate`, as shown by the example 
below:

.. code-block:: python

    results = mlflow.evaluate(
        model,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
    )

The supported LLM model types and associated metrics are listed below:

* **question-answering**: ``model_type="question-answering"``:

    * exact-match
    * `toxicity <https://huggingface.co/spaces/evaluate-measurement/toxicity>`_ :sup:`1`
    * `ari_grade_level <https://en.wikipedia.org/wiki/Automated_readability_index>`_ :sup:`2`
    * `flesch_kincaid_grade_level <https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level>`_ :sup:`2`

* **text-summarization**: ``model_type="text-summarization"``: 

    * `ROUGE <https://huggingface.co/spaces/evaluate-metric/rouge>`_ :sup:`3`
    * `toxicity <https://huggingface.co/spaces/evaluate-measurement/toxicity>`_ :sup:`1`
    * `ari_grade_level <https://en.wikipedia.org/wiki/Automated_readability_index>`_ :sup:`2`
    * `flesch_kincaid_grade_level <https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level>`_ :sup:`2`

* **text models**: ``model_type="text"``:

    * `toxicity <https://huggingface.co/spaces/evaluate-measurement/toxicity>`_ :sup:`1`
    * `ari_grade_level <https://en.wikipedia.org/wiki/Automated_readability_index>`_ :sup:`2`
    * `flesch_kincaid_grade_level <https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level>`_ :sup:`2`


:sup:`1` Requires package `evaluate <https://pypi.org/project/evaluate>`_, `torch <https://pytorch.org/get-started/locally/>`_, and 
`transformers <https://huggingface.co/docs/transformers/installation>`_

:sup:`2` Requires package `textstat <https://pypi.org/project/textstat>`_

:sup:`3` Requires package `evaluate <https://pypi.org/project/evaluate>`_, `nltk <https://pypi.org/project/nltk>`_, and 
`rouge-score <https://pypi.org/project/rouge-score>`_

.. _llm-eval-custom-metrics:

Use a Custom List of Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the pre-defined metrics associated with a given model type is not the only way to generate scoring metrics 
for LLM evaluation in MLflow. You can specify a custom list of metrics in the `extra_metrics` argument in `mlflow.evaluate`:

* To add additional metrics to the default metrics list of pre-defined model type, keep the `model_type` and add your metrics to ``extra_metrics``:
  
  .. code-block:: python

        results = mlflow.evaluate(
            model,
            eval_data,
            targets="ground_truth",
            model_type="question-answering",
            extra_metrics=[mlflow.metrics.latency()],
        )

  The above code will evaluate your model using all metrics for "question-answering" model plus :py:func:`mlflow.metrics.latency()`.

* To disable default metric calculation and only calculate your selected metrics, remove the ``model_type`` argument and define the desired metrics. 

    .. code-block:: python

        results = mlflow.evaluate(
            model,
            eval_data,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency()],
        )


The full reference for supported evaluation metrics can be found `here <../../python_api/mlflow.html#mlflow.evaluate>`_.

Metrics with LLM as the Judge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLflow offers a few pre-canned metrics which uses LLM as the judge. Despite the difference under the hood, the usage
is the same - put these metrics in the ``extra_metrics`` argument in ``mlflow.evaluate()``. Here is the list of pre-canned
metrics:

* :py:func:`mlflow.metrics.genai.answer_similarity`: Use this metric when you want to evaluate how similar the model generated output is compared to the information in the ground_truth. High scores mean that your model outputs contain similar information as the ground_truth, while low scores mean that outputs may disagree with the ground_truth.
* :py:func:`mlflow.metrics.genai.answer_correctness`: Use this metric when you want to evaluate how factually correct the model generated output is based on the information in the ground_truth. High scores mean that your model outputs contain similar information as the ground_truth and that this information is correct, while low scores mean that outputs may disagree with the ground_truth or that the information in the output is incorrect. Note that this builds onto answer_similarity.
* :py:func:`mlflow.metrics.genai.answer_relevance`: Use this metric when you want to evaluate how relevant the model generated output is to the input (context is ignored). High scores mean that your model outputs are about the same subject as the input, while low scores mean that outputs may be non-topical.
* :py:func:`mlflow.metrics.genai.relevance`: Use this metric when you want to evaluate how relevant the model generated output is with respect to both the input and the context. High scores mean that the model has understood the context and correct extracted relevant information from the context, while low score mean that output has completely ignored the question and the context and could be hallucinating.
* :py:func:`mlflow.metrics.genai.faithfulness`: Use this metric when you want to evaluate how faithful the model generated output is based on the context provided. High scores mean that the outputs contain information that is in line with the context, while low scores mean that outputs may disagree with the context (input is ignored).

Selecting the LLM-as-judge Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, llm-as-judge metrics use ``openai:/gpt-4`` as the judge. You can change the default judge model by passing an override to the ``model`` argument within the metric definition, as shown below. In addition to OpenAI models, you can also use any endpoint via MLflow Deployments. Use :py:func:`mlflow.deployments.set_deployments_target` to set the target deployment client.

To use an endpoint hosted by a local MLflow Deployments Server, you can use the following code.

.. code-block:: python

    from mlflow.deployments import set_deployments_target

    set_deployments_target("http://localhost:5000")
    my_answer_similarity = mlflow.metrics.genai.answer_similarity(
        model="endpoints:/my-endpoint"
    )

To use an endpoint hosted on Databricks, you can use the following code.

.. code-block:: python

    from mlflow.deployments import set_deployments_target

    set_deployments_target("databricks")
    llama2_answer_similarity = mlflow.metrics.genai.answer_similarity(
        model="endpoints:/databricks-llama-2-70b-chat"
    )

For more information about how various models perform as judges, please refer to `this blog <https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG>`_.

Creating Custom LLM-evaluation Metrics
--------------------------------------

Create LLM-as-judge Evaluation Metrics (Category 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also create your own Saas LLM evaluation metrics with MLflow API :py:func:`mlflow.metrics.genai.make_genai_metric`, which 
needs the following information:

* ``name``: the name of your custom metric.
* ``definition``: describe what's the metric doing. 
* ``grading_prompt``: describe the scoring critieria. 
* ``examples``: a few input/output examples with score, they are used as a reference for LLM judge.
* ``model``: the identifier of LLM judge, in the format of "openai:/gpt-4" or "endpoints:/databricks-llama-2-70b-chat".  
* ``parameters``: the extra parameters to send to LLM judge, e.g., ``temperature`` for ``"openai:/gpt-3.5-turbo-16k"``.
* ``aggregations``: The list of options to aggregate the per-row scores using numpy functions.
* ``greater_is_better``: indicates if a higher score means your model is better.

Under the hood, ``definition``, ``grading_prompt``, ``examples`` together with evaluation data and model output will be 
composed into a long prompt and sent to LLM. If you are familiar with the concept of prompt engineering, 
SaaS LLM evaluation metric is basically trying to compose a "right" prompt containing instructions, data and model 
output so that LLM, e.g., GPT4 can output the information we want. 

Now let's create a custom GenAI metrics called "professionalism", which measures how professional our model output is. 

Let's first create a few examples with scores, these will be the reference samples LLM judge uses. To create such examples, 
we will use :py:func:`mlflow.metrics.genai.EvaluationExample` class, which has 4 fields:

* input: input text.
* output: output text.
* score: the score for output in the context of input. 
* justification: why do we give the `score` for the data. 

.. code-block:: python

    professionalism_example_score_2 = mlflow.metrics.genai.EvaluationExample(
        input="What is MLflow?",
        output=(
            "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
            "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
            "workflow smoother. It's like your Swiss Army knife for machine learning!"
        ),
        score=2,
        justification=(
            "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
            "exclamation points, which make it sound less professional. "
        ),
    )
    professionalism_example_score_4 = mlflow.metrics.genai.EvaluationExample(
        input="What is MLflow?",
        output=(
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was "
            "developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning engineers face when "
            "developing, training, and deploying machine learning models.",
        ),
        score=4,
        justification=("The response is written in a formal language and a neutral tone. "),
    )

Now let's define the ``professionalism`` metric, you will see how each field is set up.

.. code-block:: python

    professionalism = mlflow.metrics.genai.make_genai_metric(
        name="professionalism",
        definition=(
            "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
            "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
            "colloquialisms, and instead using clear, concise, and respectful language."
        ),
        grading_prompt=(
            "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
            "- Score 0: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
            "professional contexts."
            "- Score 1: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
            "some informal professional settings."
            "- Score 2: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
            "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
            "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
            "business or academic settings. "
        ),
        examples=[professionalism_example_score_2, professionalism_example_score_4],
        model="openai:/gpt-3.5-turbo-16k",
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance"],
        greater_is_better=True,
    )


Create heuristic-based LLM Evaluation Metrics (Category 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is very similar to creating custom traditional metrics, with the exception of returning a :py:func:`mlflow.metrics.MetricValue` instance.
Basically you need to:

1. Implement a ``eval_fn`` to define your scoring logic, it must take in 2 args ``predictions`` and ``targets``.
   ``eval_fn`` must return a :py:func:`mlflow.metrics.MetricValue` instance.
2. Pass ``eval_fn`` and other arguments to ``mlflow.metrics.make_metric`` API to create the metric. 

The following code creates a dummy per-row metric called ``"over_10_chars"``: if the model output is greater than 10, 
the score is "yes" otherwise "no".

.. code-block:: python

    def eval_fn(predictions, targets):
        scores = []
        for i in range(len(predictions)):
            if len(predictions[i]) > 10:
                scores.append("yes")
            else:
                scores.append("no")
        return MetricValue(
            scores=scores,
            aggregate_results=standard_aggregations(scores),
        )


    # Create an EvaluationMetric object.
    passing_code_metric = make_metric(
        eval_fn=eval_fn, greater_is_better=False, name="over_10_chars"
    )

If we want to create a custom metric that is dependent on other metrics, we can include those other metrics' names as an argument after ``predictions`` and ``targets``. This can be the name of a builtin metric or another custom metric.
Ensure that you do not accidentally have any circular dependencies in your metrics, or the evaluation will fail.

The following code creates a dummy per-row metric called ``"toxic_or_over_10_chars"``: if the model output is greater than 10 or the toxicity score is greater than 0.5, the score is "yes" otherwise "no".

.. code-block:: python

    def eval_fn(predictions, targets, toxicity, over_10_chars):
        scores = []
        for i in range(len(predictions)):
            if toxicity.scores[i] > 0.5 or over_10_chars.scores[i]:
                scores.append("yes")
            else:
                scores.append("no")
        return MetricValue(scores=scores)


    # Create an EvaluationMetric object.
    toxic_and_over_10_chars_metric = make_metric(
        eval_fn=eval_fn, greater_is_better=False, name="toxic_or_over_10_chars"
    )

Prepare Your LLM for Evaluating
-------------------------------

In order to evaluate your LLM with ``mlflow.evaluate()``, your LLM has to be one of the following type:

1. A :py:func:`mlflow.pyfunc.PyFuncModel` instance or a URI pointing to a logged `mlflow.pyfunc.PyFuncModel` model. In
   general we call that MLflow model. The 
2. A python function that takes in string inputs and outputs a single string. Your callable must match the signature of 
   :py:func:`mlflow.pyfunc.PyFuncModel.predict` (without `params` argument), briefly it should:

   * Has ``data`` as the only argument, which can be a ``pandas.Dataframe``, ``numpy.ndarray``, python list, dictionary or scipy matrix.
   * Returns one of ``pandas.DataFrame``, ``pandas.Series``, ``numpy.ndarray`` or list. 
3. Set ``model=None``, and put model outputs in `data`. Only applicable when the data is a Pandas dataframe.

Evaluating with an MLflow Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detailed instruction on how to convert your model into a ``mlflow.pyfunc.PyFuncModel`` instance, please read
`this doc <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models>`_. But in short,
to evaluate your model as an MLflow model, we recommend following the steps below:

1. Package your LLM as an MLflow model and log it to MLflow server by ``log_model``. Each flavor (``opeanai``, ``pytorch``, ...) 
   has its own ``log_model`` API, e.g., :py:func:`mlflow.openai.log_model()`:

   .. code-block:: python

        with mlflow.start_run():
            system_prompt = "Answer the following question in two sentences"
            # Wrap "gpt-3.5-turbo" as an MLflow model.
            logged_model_info = mlflow.openai.log_model(
                model="gpt-3.5-turbo",
                task=openai.ChatCompletion,
                artifact_path="model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "{question}"},
                ],
            )
2. Use the URI of logged model as the model instance in ``mlflow.evaluate()``:
   
   .. code-block:: python

        results = mlflow.evaluate(
            logged_model_info.model_uri,
            eval_data,
            targets="ground_truth",
            model_type="question-answering",
        )

Evaluating with a Custom Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of MLflow 2.8.0, :py:func:`mlflow.evaluate()` supports evaluating a python function without requiring 
logging the model to MLflow. This is useful when you don't want to log the model and just want to evaluate
it. The following example uses :py:func:`mlflow.evaluate()` to evaluate a function. You also need to set
up OpenAI authentication to run the code below.

.. code-block:: python

    eval_data = pd.DataFrame(
        {
            "inputs": [
                "What is MLflow?",
                "What is Spark?",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
                "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, offering improvements in speed and ease of use. Spark provides libraries for various tasks such as data ingestion, processing, and analysis through its components like Spark SQL for structured data, Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
            ],
        }
    )


    def openai_qa(inputs):
        answers = []
        system_prompt = "Please answer the following question in formal language."
        for index, row in inputs.iterrows():
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "{row}"},
                ],
            )
            answers.append(completion.choices[0].message.content)

        return answers


    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            openai_qa,
            eval_data,
            model_type="question-answering",
        )

.. _llm-eval-static-dataset:

Evaluating with a Static Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MLflow >= 2.8.0, :py:func:`mlflow.evaluate()` supports evaluating a static dataset without specifying a model.
This is useful when you save the model output to a column in a Pandas DataFrame or an MLflow PandasDataset, and
want to evaluate the static dataset without re-running the model.

If you are using a Pandas DataFrame, you must specify the column name that contains the model output using the
top-level ``predictions`` parameter in :py:func:`mlflow.evaluate()`:


.. code-block:: python

    import mlflow
    import pandas as pd

    eval_data = pd.DataFrame(
        {
            "inputs": [
                "What is MLflow?",
                "What is Spark?",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. "
                "It was developed by Databricks, a company that specializes in big data and machine learning solutions. "
                "MLflow is designed to address the challenges that data scientists and machine learning engineers "
                "face when developing, training, and deploying machine learning models.",
                "Apache Spark is an open-source, distributed computing system designed for big data processing and "
                "analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, "
                "offering improvements in speed and ease of use. Spark provides libraries for various tasks such as "
                "data ingestion, processing, and analysis through its components like Spark SQL for structured data, "
                "Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
            ],
            "predictions": [
                "MLflow is an open-source platform that provides handy tools to manage Machine Learning workflow "
                "lifecycle in a simple way",
                "Spark is a popular open-source distributed computing system designed for big data processing and analytics.",
            ],
        }
    )

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            data=eval_data,
            targets="ground_truth",
            predictions="predictions",
            extra_metrics=[mlflow.metrics.genai.answer_similarity()],
            evaluators="default",
        )
        print(f"See aggregated evaluation results below: \n{results.metrics}")

        eval_table = results.tables["eval_results_table"]
        print(f"See evaluation table below: \n{eval_table}")


Viewing Evaluation Results
--------------------------

View Evaluation Results via Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``mlflow.evaluate()`` returns the evaluation results as an :py:func:`mlflow.models.EvaluationResult` instance. 
To see the score on selected metrics, you can check:

* ``metrics``: stores the aggregated results, like average/variance across the evaluation dataset. Let's take a second
  pass on the code example above and focus on printing out the aggregated results.
  
  .. code-block:: python

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            data=eval_data,
            targets="ground_truth",
            predictions="predictions",
            extra_metrics=[mlflow.metrics.genai.answer_similarity()],
            evaluators="default",
        )
        print(f"See aggregated evaluation results below: \n{results.metrics}")

* ``tables["eval_results_table"]``: stores the per-row evaluation results. 

  .. code-block:: python

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            data=eval_data,
            targets="ground_truth",
            predictions="predictions",
            extra_metrics=[mlflow.metrics.genai.answer_similarity()],
            evaluators="default",
        )
        print(
            f"See per-data evaluation results below: \n{results.tables['eval_results_table']}"
        )



View Evaluation Results via the MLflow UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your evaluation result is automatically logged into MLflow server, so you can view your evaluation results directly from the
MLflow UI. To view the evaluation results on MLflow UI, please follow the steps below:

1. Go to the experiment view of your MLflow experiment.
2. Select the "Evaluation" tab.
3. Select the runs you want to check evaluation results.
4. Select the metrics from the dropdown menu on the right side. 

Please see the screenshot below for clarity:


.. figure:: ../../_static/images/llm_evaluate_experiment_view.png
    :width: 1024px
    :align: center
    :alt: Demo UI of MLflow evaluate
