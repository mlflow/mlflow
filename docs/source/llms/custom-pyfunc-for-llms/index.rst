Deploying Advanced LLMs with Custom PyFuncs in MLflow
=====================================================

Advanced Large Language Models (LLMs) such as the MPT-7B instruct transformer are intricate and have requirements that don't align with 
traditional MLflow flavors. This demands a deeper understanding and the need for custom solutions.

.. tip::

    MLflow 2.12.2 introduced the feature "models from code", which greatly simplifies the process of serializing and deploying custom models through the use 
    of script serialization. While the tutorial here is valuable as a point of reference, we strongly recommend migrating custom model implementations to this 
    new paradigm. You can learn more about models from code within the `Models From Code Guide <../../model/models-from-code.html>`_.

What's in this tutorial?

This guide is designed to provide insights into the deployment of advanced LLMs with MLflow, with a focus on using custom PyFuncs to address challenges:

- **The World of LLMs**: An introduction to LLMs, particularly models like the MPT-7B instruct transformer. We'll delve into their intricacies, importance, and the challenges associated with their deployment.

- **Why Custom PyFuncs for LLM Deployment?**: We'll explore the reasons behind the need for custom PyFuncs in the context of LLMs. How do they provide a bridge, ensuring that LLMs can be seamlessly deployed while adhering to MLflow's standards?

    - **Managing Complex Behaviors**: How custom PyFuncs can help in handling intricate model behaviors and dependencies that aren't catered to by MLflow's default flavors.

    - **Interface Data Manipulation**: Delve into how custom PyFuncs allow the manipulation of interface data to generate prompts, thereby simplifying end-user interactions in a RESTful environment.

- **Crafting Custom PyFuncs for LLM Deployment**: A step-by-step walkthrough on how to define, manage, and deploy an LLM using a custom PyFunc. We'll look at how to design a `pyfunc` to address LLM requirements and behaviors, and then how to deploy it using MLflow.

- **Challenges with Traditional LLM Deployment**: Recognize the issues and limitations when trying to deploy an advanced LLM using MLflow's built-in capabilities. Understand why custom PyFuncs become essential in such scenarios.

By the conclusion of this guide, you'll possess a deep understanding of how to deploy advanced LLMs in MLflow using custom PyFuncs. You'll appreciate the role of custom PyFuncs in making complex deployments streamlined and efficient.

Explore the Tutorial
--------------------

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/custom-pyfunc-advanced-llm.html" >
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

.. toctree::
    :maxdepth: 1
    :hidden:

    Full Notebooks <notebooks/index>
