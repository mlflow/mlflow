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
functionality to evaluate the retrieved documents from our corpus based on a series of questions. You can click "Download this Notebook" button to 
download the ``.ipynb`` file locally and import it directly in the Databricks Workspace.


.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="mlflow-e2e-evaluation.html">
                    <div class="header">
                        End-to-End RAG Evaluation with MLflow
                    </div>
                    <p>
                        Comprehensive tutorial on evaluating Retrieval-Augmented Generation (RAG) systems using MLflow
                    </p>
                </a>
            </div>
        </article>
    </section>


Question Generation for RAG Tutorial
------------------------------------

This notebook is a step-by-step tutorial on how to generate a question dataset with 
LLMs for retrieval evaluation within RAG. It will guide you through getting a document dataset,
generating relevant questions through prompt engineering on LLMs, and analyzing the 
question dataset. The question dataset can then be used for the subsequent task of evaluating the 
retriever model, which is a part of RAG that collects and ranks relevant document chunks based on
the user's question.

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="question-generation-retrieval-evaluation.html">
                    <div class="header">
                        Question Generation for RAG Evaluation
                    </div>
                    <p>
                        Step-by-step demonstration for how to automatically generate a question-answering dataset for RAG evaluation
                    </p>
                </a>
            </div>
        </article>
    </section>

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

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="retriever-evaluation-tutorial.ipynb">
                    <div class="header">
                        Retriever Evaluation with MLflow
                    </div>
                    <p>
                        Learn how to leverage MLflow to evaluate the performance of a retriever in a RAG application,
                        leveraging built-in retriever metrics <code>precision_at_k</code> and <code>recall_at_k</code>.
                    </p>
                </a>
            </div>
        </article>
    </section>
