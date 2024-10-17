Model Evaluation
================

Harnessing the Power of Automation
----------------------------------

In the evolving landscape of machine learning, the evaluation phase of model development is just as important as ever. 
Ensuring the accuracy, reliability, and efficiency of models is paramount to ensure that the model that has been trained has been as thoroughly 
validated as it can be prior to promoting it beyond the development phase. 

However, manual evaluation can be tedious, error-prone, and time-consuming. 

MLflow addresses these challenges head-on, offering a suite of automated tools that streamline the evaluation process, 
saving time and enhancing accuracy, helping you to have confidence that the solution that you've spent so much time working on will meet the 
needs of the problem you're trying to solve.

`LLM Model Evaluation <../llms/llm-evaluate/index.html>`_
---------------------------------------------------------

The rise of Large Language Models (LLMs) like ChatGPT has transformed the landscape of text generation, finding applications in question answering, translation, and text summarization. However, evaluating LLMs introduces unique challenges, primarily because there's often no single ground truth to compare against. MLflow's evaluation tools are tailored for LLMs, ensuring a streamlined and accurate evaluation process.

**Key Features**:

- **Versatile Model Evaluation**: MLflow supports evaluating various types of LLMs, whether it's an MLflow pyfunc model, a URI pointing to a registered MLflow model, or any python callable representing your model.

- **Comprehensive Metrics**: MLflow offers a range of metrics for LLM evaluation. From metrics that rely on SaaS models like OpenAI for scoring (e.g., :py:func:`mlflow.metrics.genai.answer_relevance`) to function-based per-row metrics such as Rouge (:py:func:`mlflow.metrics.rougeL`) or Flesch Kincaid (:py:func:`mlflow.metrics.flesch_kincaid_grade_level`).

- **Predefined Metric Collections**: Depending on your LLM use case, MLflow provides predefined metric collections, such as "question-answering" or "text-summarization", simplifying the evaluation process.

- **Custom Metric Creation**: Beyond the predefined metrics, MLflow allows users to create custom LLM evaluation metrics. Whether you're looking to evaluate the professionalism of a response or any other custom criteria, MLflow provides the tools to define and implement these metrics.

- **Evaluation with Static Datasets**: As of MLflow 2.8.0, you can evaluate a static dataset without specifying a model. This is especially useful when you've saved model outputs in a dataset and want a swift evaluation without rerunning the model.

- **Integrated Results View**: MLflow's :py:func:`mlflow.evaluate` returns comprehensive evaluation results, which can be viewed directly in the code or through the MLflow UI for a more visual representation.

Harnessing these features, MLflow's LLM evaluation tools eliminate the complexities and ambiguities associated with evaluating large language models. By automating these critical evaluation tasks, MLflow ensures that users can confidently assess the performance of their LLMs, leading to more informed decisions in the deployment and application of these models.

Guides and Tutorials for LLM Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To learn more about how you can leverage MLflow's evaluation features for your LLM-powered project work, see the tutorials below:

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="../llms/llm-evaluate/notebooks/rag-evaluation.html">
                    <div class="header">
                        RAG Evaluation
                    </div>
                    <p>
                        Learn how to evaluate a Retrieval Augmented Generation setup with MLflow Evaluate
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../llms/llm-evaluate/notebooks/question-answering-evaluation.html">
                    <div class="header">
                        Question-Answering Evaluation
                    </div>
                    <p>
                        See a working example of how to evaluate the quality of an LLM Question-Answering solution
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../llms/rag/notebooks/question-generation-retrieval-evaluation.html">
                    <div class="header">
                        RAG Question Generation Evaluation
                    </div>
                    <p>
                        See how to generate Questions for RAG generation and how to evaluate a RAG solution using MLflow
                    </p>
                </a>
            </div>
        </article>
    </section>


`Traditional ML Evaluation <../models.html#model-evaluation>`_
--------------------------------------------------------------

Traditional machine learning techniques, from classification to regression, have been the bedrock of many industries. MLflow recognizes 
their significance and offers automated evaluation tools tailored for these classic techniques. 

**Key Features**:

- `Evaluating a Function <../models.html#evaluating-with-a-function>`_: To get immediate results, you can evaluate a python function directly without logging the model. This is especially useful when you want a quick evaluation without the overhead of logging.
  
- `Evaluating a Dataset <../models.html#evaluating-with-a-static-dataset>`_: MLflow also supports evaluating a static dataset without specifying a model. This is invaluable when you've saved model outputs in a dataset and want a swift evaluation without having to rerun model inference.

- `Evaluating a Model <../models.html#performing-model-validation>`_: With MLflow, you can set validation thresholds for your metrics. If a model doesn't meet these thresholds compared to a baseline, MLflow will alert you. This automated validation ensures that only high-quality models progress to the next stages.

- `Common Metrics and Visualizations <../models.html#model-evaluation>`_: MLflow automatically logs common metrics like accuracy, precision, recall, and more. Additionally, visual graphs such as the confusion matrix, lift_curve_plot, and others are auto-logged, providing a comprehensive view of your model's performance.

- **SHAP Integration**: MLflow is integrated with SHAP, allowing for the auto-logging of SHAP's summarization importances validation visualizations when using the evaluate APIs.
