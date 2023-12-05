Getting Started with MLflow Deployments for LLMs
================================================

MLflow provides a robust framework for deploying and managing machine learning models. In this tutorial, we will explore how to set up an
MLflow Deployments Server tailored for OpenAI's models, allowing seamless integration and querying of OpenAI's powerful language models.

What's in this tutorial?

This guide will cover:

- **Installation**: Setting up the necessary dependencies and tools to get your MLflow Deployments Server up and running.

- **Configuration**: How to expose your OpenAI token, configure the deployments server, and define routes for various OpenAI models.

- **Starting the deployments server**: Launching the deployments server and ensuring it's operational.

- **Querying the deployments server**: Interacting with the deployments server using fluent APIs to query various OpenAI models, including completions, chat, and embeddings.

By the end of this tutorial, you'll have a fully functional MLflow Deployments Server tailored for OpenAI, ready to handle and process requests.
You'll also gain insights into querying different types of routes, providers, and models through the deployments server.

.. toctree::
    :maxdepth: 1

    Setting Up the MLflow Deployments Server <step1-create-deployments>
    Querying the MLflow Deployments Server <step2-query-deployments>
