Custom PyFuncs for Advanced LLMs with MLflow - Notebooks
========================================================

If you'd like to delve deeper into the notebooks in this guide, they can be viewed or downloaded directly below.

.. toctree::
    :maxdepth: 1
    :hidden:

    custom-pyfunc-advanced-llm.ipynb

Deploying Advanced LLMs with Custom PyFuncs
-------------------------------------------

Introduction
^^^^^^^^^^^^

In this tutorial, we'll explore the nuances of deploying advanced Large Language Models (LLMs) with MLflow, particularly focusing on models 
that can't be readily managed with MLflow's built-in functionality. We'll highlight the necessity of custom `pyfunc` definitions when 
dealing with such complex models, emphasizing its role in managing intricate model behaviors and dependencies. By the end, you'll understand 
the intricacies of deploying an LLM model using the MPT-7B instruct transformer, wrapped efficiently using a custom `pyfunc`.

What you will learn
^^^^^^^^^^^^^^^^^^^

- **LLM Deployment Challenges**: Recognize the complexities and challenges associated with deploying advanced LLMs in MLflow.
- **Custom PyFuncs for LLMs**: Understand the need and process of creating a custom `pyfunc` to effectively manage LLMs, particularly when default flavors fall short.
- **Prompt Management in Deployment**: Delve into how custom `pyfunc` allows manipulation of interface data to generate prompts, simplifying end-user interactions in a RESTful environment.
- **Leveraging Custom PyFunc for Enhanced Flexibility**: Witness how custom `pyfunc` definitions provide the flexibility needed for advanced model behaviors and dependencies.

Why Custom `pyfunc` for LLM Deployment?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deploying advanced LLMs isn't straightforward. Models like the MPT-7B instruct transformer have specific requirements and behaviors that don't align with traditional MLflow flavors. This section highlights the challenges faced and the importance of custom `pyfunc` definitions in addressing these challenges.

Crafting the Custom `pyfunc`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Venturing into the solution, we'll craft a custom `pyfunc` to efficiently wrap and manage our LLM. This custom definition serves as a bridge, ensuring our LLM can be deployed seamlessly while retaining its original capabilities and adhering to MLflow's standards.

Step-by-step Guide
^^^^^^^^^^^^^^^^^^

1. **LLM Introduction**: Understand the MPT-7B instruct transformer, its importance, and its intricacies.
2. **Challenges with Traditional Deployment**: Recognize the difficulties when attempting to deploy such an LLM using MLflow's default capabilities.
3. **Designing the Custom `pyfunc`**: Create a custom `pyfunc` that addresses the LLM's requirements and behaviors.
4. **Deploying the LLM**: Integrate with MLflow to deploy the LLM using the crafted custom `pyfunc`.
5. **Interface Simplification**: Examine how the custom `pyfunc` simplifies user interactions, particularly in RESTful deployments.

Wrap Up
^^^^^^^

With the complexities of advanced LLM deployment unraveled, this tutorial showcases the indispensable role of custom `pyfunc` in MLflow. Through a detailed, hands-on approach, you'll appreciate how custom `pyfunc` definitions can make seemingly insurmountable deployment challenges manageable and streamlined.

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="custom-pyfunc-advanced-llm.html" >
                    <div class="header">
                        Serving LLMs with MLflow: Leveraging Custom PyFunc
                    </div>
                    <p>
                        Learn how to use the MLflow Custom Pyfunc Model to serve Large Language Models (LLMs) in a RESTful environment.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. note::
    To execute the notebooks, ensure you either have a local MLflow Tracking Server running or adjust the ``mlflow.set_tracking_uri()`` to point to an active MLflow Tracking Server instance. 
    To engage with the MLflow UI, ensure you're either running the UI server locally or have a configured, accessible, deployed MLflow UI server.
