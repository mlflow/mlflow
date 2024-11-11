Getting Started with MLflow Deployments for LLMs
================================================

MLflow provides a robust framework for deploying and managing machine learning models. In this tutorial, we will explore how to set up an
MLflow AI Gateway tailored for OpenAI's models, allowing seamless integration and querying of OpenAI's powerful language models.

What's in this tutorial?

This guide will cover:

- **Installation**: Setting up the necessary dependencies and tools to get your MLflow AI Gateway up and running.

- **Configuration**: How to expose your OpenAI token, configure the gateway server, and define routes for various OpenAI models.

- **Starting the gateway server**: Launching the gateway server and ensuring it's operational.

- **Querying the gateway server**: Interacting with the gateway server using fluent APIs to query various OpenAI models, including completions, chat, and embeddings.

By the end of this tutorial, you'll have a fully functional MLflow AI Gateway tailored for OpenAI, ready to handle and process requests.
You'll also gain insights into querying different types of routes, providers, and models through the gateway server.

.. toctree::
    :maxdepth: 1

    Setting Up the MLflow AI Gateway <step1-create-deployments>
    Querying the MLflow AI Gateway <step2-query-deployments>
