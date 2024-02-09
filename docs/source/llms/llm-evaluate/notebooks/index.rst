LLM Evaluation Examples
=======================

The notebooks listed below contain step-by-step tutorials on how to use MLflow to evaluate LLMs. 
The first notebook is centered around evaluating an LLM for question-answering with a 
prompt engineering approach. The second notebook is centered around evaluating a RAG system. 
Both notebooks will demonstrate how to use MLflow's builtin metrics such as token_count and 
toxicity as well as LLM-judged intelligent metrics such as answer_relevance. The third notebook 
is the same as the second notebook, but uses Databricks's served llama2-70b as the judge instead 
of gpt-4.

.. toctree::
    :maxdepth: 1
    :hidden:

    question-answering-evaluation.ipynb
    rag-evaluation.ipynb
    rag-evaluation-llama2.ipynb
    huggingface-evaluation.ipynb

QA Evaluation Notebook
----------------------

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/question-answering-evaluation.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="question-answering-evaluation.html" class="download-btn">View the Notebook</a><br/>


RAG Evaluation Notebook (using gpt-4-as-judge)
----------------------------------------------

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/rag-evaluation.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="rag-evaluation.html" class="download-btn">View the Notebook</a><br/>


RAG Evaluation Notebook (using llama2-70b-as-judge)
---------------------------------------------------

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/rag-evaluation-llama2.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="rag-evaluation-llama2.html" class="download-btn">View the Notebook</a><br/>

Evaluating a 🤗 Hugging Face LLM Notebook (using gpt-4-as-judge)
-----------------------------------------------------------------

Learn how to evaluate an Open-Source 🤗 Hugging Face LLM with MLflow evaluate by downloading the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/huggingface-evaluation.ipynb" class="notebook-download-btn">Download the notebook</a><br/>


Or follow along directly in the docs here:

.. raw:: html

    <a href="huggingface-evaluation.html" class="download-btn">View the Notebook</a><br/>
