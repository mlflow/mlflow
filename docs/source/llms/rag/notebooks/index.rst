=============
RAG Tutorials
=============



.. toctree::
    :maxdepth: 1
    :hidden:

    question-generation-retrieval-evaluation.ipynb
    retriever-evaluation-tutorial.ipynb

Question Generation for RAG Tutorial
------------------------------------

This notebook is a step-by-step tutorial on how to generate a question dataset with 
LLMs for retrieval evaluation within RAG. It will guide you through getting a document dataset,
generating diverse and relevant questions through prompt engineering on LLMs, and analyzing the 
question dataset. The question dataset can then be used for the subsequent task of evaluating the 
retriever model, which is a part of RAG that collects and ranks relevant document chunks based on
the user's question.

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/rag/notebooks/question-generation-retrieval-evaluation.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="question-generation-retrieval-evaluation.html" class="download-btn">View the Notebook</a><br/>


Retriever Evaluation Tutorial
-----------------------------

This tutorial is geared toward providing you insight into how to build a RAG application leveraging LangChain and how to adjudicate its effectiveness using MLflow evaluate. (probably want to explain it a bit better than this example)

In this tutorial you will learn:

- How to generate an embeddings corpus from encoded documents
- How to store the processed embeddings within a Vector DB (FAISS <link to what FAISS is>)
- How to evaluate a model's capacity for retrieving relevant document matches based on a series of queries using MLflow evaluate
- etc, etc, etc...

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/rag/notebooks/retriever-evaluation-tutorial.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="retriever-evaluation-tutorial.html" class="download-btn">View the Notebook</a><br/>
