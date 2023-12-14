=============
RAG Tutorials
=============

You can find a list of tutorials for RAG below. These tutorials are designed to help you
get started with RAG evaluation and walk you through a concrete example of how to evaluate
a RAG application that answers questions about MLflow documentation.

.. toctree::
    :maxdepth: 1
    :hidden:

    mlflow-e2e-evaluation.ipynb
    question-generation-retrieval-evaluation.ipynb
    retriever-evaluation-tutorial.ipynb


End-to-End LLM RAG Evaluation Tutorial
--------------------------------------

This notebook, intended for use with the Databricks platform, showcases a full end-to-end example of how to configure, create, and interface with 
a full RAG system. The example used in this tutorial uses the documentation of MLflow as the corpus of embedded documents that the RAG application will 
use to answer questions. Using ChromaDB to store the document embeddings and LangChain to orchestrate the RAG application, we'll use MLflow's `evaluate` 
functionality to evaluate the retrieved documents from our corpus based on a series of questions.

If you would like to try this notebook out on Databricks, you can import it directly from the Databricks Workspace. If you would like a local copy to 
manually import into your Workspace, you can download it here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/rag/notebooks/mlflow-e2e-evaluation.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="mlflow-e2e-evaluation.html" class="download-btn">View the Notebook</a><br/>


Question Generation for RAG Tutorial
------------------------------------

This notebook is a step-by-step tutorial on how to generate a question dataset with 
LLMs for retrieval evaluation within RAG. It will guide you through getting a document dataset,
generating relevant questions through prompt engineering on LLMs, and analyzing the 
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

This tutorial walks you through a concrete example of how to build and evaluate
a RAG application that answers questions about MLflow documentation.

In this tutorial you will learn:

- How to prepare an evaluation dataset for your RAG application.
- How to call your retriever in the MLflow evaluate API.
- How to evaluate a retriever's capacity for retrieving relevant documents based on a series of queries using MLflow evaluate.

If you would like a copy of this notebook to execute in your environment, download the notebook here:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/rag/notebooks/retriever-evaluation-tutorial.ipynb" class="notebook-download-btn">Download the notebook</a><br/>

To follow along and see the sections of the notebook guide, click below:

.. raw:: html

    <a href="retriever-evaluation-tutorial.html" class="download-btn">View the Notebook</a><br/>
