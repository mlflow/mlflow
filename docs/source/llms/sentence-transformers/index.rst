MLflow Sentence-Transformers Flavor
===================================

.. attention::
    The ``sentence-transformers`` flavor is under active development and is marked as Experimental. Public APIs are subject to change, 
    and new features may be added as the flavor evolves.

Introduction
------------

**Sentence-Transformers** is a groundbreaking Python library that specializes in producing high-quality, semantically rich embeddings 
for sentences and paragraphs. Developed as an extension of the well-known `Transformers <https://huggingface.co/docs/transformers/index>`_ library 
by 🤗 Hugging Face, Sentence-Transformers is tailored for tasks requiring a deep understanding of sentence-level context. This library is 
essential for NLP applications such as semantic search, text clustering, and similarity assessment.

Leveraging pre-trained models like BERT, RoBERTa, and DistilBERT, which are fine-tuned for sentence embeddings, Sentence-Transformers simplifies the process 
of generating meaningful vector representations of text. The library stands out for its simplicity, efficiency, and the quality of embeddings it produces.

The library features a number of powerful high-level utility functions for performing common follow-on tasks with sentence embeddings. 
These include:

- **Semantic Textual Similarity**: Assessing the `semantic similarity <https://www.sbert.net/docs/usage/semantic_textual_similarity.html>`_ between two sentences.
- **Semantic Search**: `Searching <https://www.sbert.net/examples/applications/semantic-search/README.html>`_ for the most semantically similar sentences in a corpus for a given query.
- **Clustering**: Grouping `similar sentences <https://www.sbert.net/examples/applications/clustering/README.html>`_ together.
- **Information Retrieval**: Finding the most relevant sentences for a given query via document `retrieval and ranking <https://www.sbert.net/examples/applications/retrieve_rerank/README.html>`_.
- **Paraphrase Mining**: Finding text entries that have similar (or identical) `meaning <https://www.sbert.net/examples/applications/paraphrase-mining/README.html>`_ in a large corpus of text.

What makes this Library so Special?
-----------------------------------
Let's take a look at a very basic representation of how the Sentence-Transformers library works and what you can do with it!

.. figure:: ../../_static/images/tutorials/llms/sentence-transformers-architecture.png
   :alt: Sentence-Transformers Model Architecture
   :width: 90%
   :align: center

   Sentence-Transformers Model Architecture Overview

Integrating Sentence-Transformers with MLflow, a platform dedicated to streamlining the entire machine learning lifecycle, enhances the experiment tracking and deployment 
capabilities for these specialized NLP models. MLflow's support for Sentence-Transformers enables practitioners to effectively manage experiments, track different model versions, 
and deploy models for various NLP tasks with ease.

Sentence-Transformers offers:

- **High-Quality Sentence Embeddings**: Efficient generation of sentence embeddings that capture the contextual and semantic nuances of language.
- **Pre-Trained Model Availability**: Access to a diverse range of pre-trained models fine-tuned for sentence embedding tasks, streamlining the process of embedding generation.
- **Ease of Use**: Simplified API, making it accessible for both NLP experts and newcomers.
- **Custom Training and Fine-Tuning**: Flexibility to fine-tune models on specific datasets or train new models from scratch for tailored NLP solutions.

With MLflow's Sentence-Transformers flavor, users benefit from:

- **Streamlined Experiment Tracking**: Easily log parameters, metrics, and sentence embedding models during the training and fine-tuning process.
- **Hassle-Free Deployment**: Deploy sentence embedding models for various applications with straightforward API calls.
- **Broad Model Compatibility**: Support for a range of sentence embedding models from the Sentence-Transformers library, ensuring access to the latest in embedding technology.

Whether you're working on semantic text similarity, clustering, or information retrieval, MLflow's integration with Sentence-Transformers provides a robust and efficient 
pathway for incorporating advanced sentence-level understanding into your applications.

Features
--------

With MLflow's Sentence-Transformers flavor, users can:

- **Save** and **log** Sentence-Transformer models within MLflow with the respective APIs: :py:func:`mlflow.sentence_transformers.save_model` and :py:func:`mlflow.sentence_transformers.log_model`.
- Track detailed experiments, including **parameters**, **metrics**, and **artifacts** associated with fine tuning runs.
- `Deploy <../../deployment/index.html>`_ sentence embedding models for practical applications.
- Utilize the :py:class:`mlflow.pyfunc.PythonModel` flavor for generic Python function inference, enabling complex and powerful custom ML solutions.

What can you do with Sentence Transformers and MLflow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the more powerful applications that can be built with these tools is a semantic search engine. By using readily available open source 
tooling, you can build a semantic search engine that can find the most semantically similar sentences in a corpus for a given query. This is 
a significant improvement over traditional keyword-based search engines, which are limited in their ability to understand the context of a query.

An example high-level architecture for such an application stack is shown below:

.. figure:: ../../_static/images/tutorials/llms/semantic-search-arch.png
   :alt: Semantic Search Architecture
   :width: 90%
   :align: center

   A basic architecture for a semantic search engine built with Sentence Transformers and MLflow


Deployment Made Easy
^^^^^^^^^^^^^^^^^^^^

Once a model is trained, it needs to be deployed for inference. MLflow's integration with Sentence Transformers simplifies this by providing 
functions such as :py:func:`mlflow.sentence_transformers.load_model` and :py:func:`mlflow.pyfunc.load_model`, which allow for easy model serving.
You can read more about `deploying models with MLflow <../../deployment/index.html>`_, find further information on 
`using the deployments API <../../cli.html#mlflow-deployments>`_, and `starting a local model serving endpoint <../../cli.html#mlflow-models-serve>`_ to get a 
deeper understanding of the deployment options that MLflow has available.

Getting Started with the MLflow Sentence Transformers Flavor - Tutorials and Guides
-----------------------------------------------------------------------------------

Below, you will find a number of guides that focus on different ways that you can leverage the power of the `sentence-transformers` library, leveraging MLflow's 
APIs for tracking and inference capabilities. 


.. toctree::
    :maxdepth: 2
    :hidden:

    tutorials/quickstart/sentence-transformers-quickstart.ipynb
    tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb
    tutorials/semantic-search/semantic-search-sentence-transformers.ipynb
    tutorials/semantic-similarity/semantic-similarity-sentence-transformers.ipynb

Introductory Tutorial
^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorials/quickstart/sentence-transformers-quickstart.html">
                    <div class="header">
                        Sentence Transformers Quickstart
                    </div>
                    <p>
                        Learn the very basics of using the Sentence Transformers package with MLflow to generate sentence embeddings from a logged model in 
                        both native and generic Python function formats.
                    </p>
                </a>
            </div>
        </article>
    </section>

Advanced Tutorials
^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorials/semantic-similarity/semantic-similarity-sentence-transformers.html">
                    <div class="header">
                        Semantic Similarity Tutorial
                    </div>
                    <p>
                        Learn how to leverage sentence embeddings to determine similarity scores between two sentences.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/semantic-search/semantic-search-sentence-transformers.html">
                    <div class="header">
                        Semantic Search Tutorial
                    </div>
                    <p>
                        Learn how to use sentence embeddings to find the most similar embedding within a corpus of text.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.html">
                    <div class="header">
                        Paraphrase Mining Tutorial
                    </div>
                    <p>
                        Explore the power of paraphrase mining to identify semantically similar sentences in a corpus of text.
                    </p>
                </a>
            </div>
        </article>
    </section>


`Detailed Documentation <guide/index.html>`_
--------------------------------------------

To learn more about the details of the MLflow flavor for sentence transformers, delve into the comprehensive guide below.

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Comprehensive Guide</a>

.. toctree::
   :maxdepth: 1
   :hidden:

   guide/index.rst


Learning More About Sentence Transformers
-----------------------------------------

Sentence Transformers is a versatile framework for computing dense vector representations of sentences, paragraphs, and images. Based on transformer networks like BERT, RoBERTa, and XLM-RoBERTa, it offers state-of-the-art performance across various tasks. The framework is designed for easy use and customization, making it suitable for a wide range of applications in natural language processing and beyond.

For those interested in delving deeper into Sentence Transformers, the following resources are invaluable:

Official Documentation and Source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Official Documentation**: For a comprehensive guide to getting started, advanced usage, and API references, visit the `Sentence Transformers Documentation <https://www.sbert.net>`_.

- **GitHub Repository**: The `Sentence Transformers GitHub repository <https://github.com/UKPLab/sentence-transformers>`_ is the primary source for the latest code, examples, and updates. Here, you can also report issues, contribute to the project, or explore how the community is using and extending the framework.


Official Guides and Tutorials for Sentence Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Training Custom Models**: The framework supports `fine-tuning of custom embedding models <https://www.sbert.net/docs/training/overview.html>`_ to achieve the best performance on specific tasks.

- **Publications and Research**: To understand the scientific foundations of Sentence Transformers, the `publications section <https://www.sbert.net/docs/publications.html>`_ offers a collection of research papers that have been integrated into the framework.

- **Application Examples**: Explore a variety of `application examples <https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications>`_ demonstrating the practical use of Sentence Transformers in different scenarios.


Library Resources
^^^^^^^^^^^^^^^^^

- **PyPI Package**: The `PyPI page for Sentence Transformers <https://pypi.org/project/sentence-transformers/>`_ provides information on installation, version history, and package dependencies.

- **Conda Forge Package**: For users preferring Conda as their package manager, the `Conda Forge page for Sentence Transformers <https://anaconda.org/conda-forge/sentence-transformers>`_ is the go-to resource for installation and package details.

- **Pretrained Models**: Sentence Transformers offers an extensive range of `pretrained models <https://www.sbert.net/docs/pretrained_models.html>`_ optimized for various languages and tasks. These models can be easily integrated into your projects.



Sentence Transformers is continually evolving, with regular updates and additions to its capabilities. Whether you're a researcher, developer, or enthusiast in the field of natural language processing, these resources will help you make the most of this powerful tool.
