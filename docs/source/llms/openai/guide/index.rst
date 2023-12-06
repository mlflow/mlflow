OpenAI within MLflow
====================

.. attention::
    The ``openai`` flavor is under active development and is marked as Experimental. Public APIs may change and new features are
    subject to be added as additional functionality is brought to the flavor.

Overview
--------
The integration of OpenAI's advanced language models within MLflow opens up new frontiers in NLP applications. It enables users to harness 
the cutting-edge capabilities of models like GPT-3.5-turbo for varied tasks, ranging from conversational AI to complex text analysis 
and embeddings generation. This integration is a leap forward in making advanced NLP accessible and manageable within a robust framework like MLflow.

Expanding the Possibilities with OpenAI and MLflow
--------------------------------------------------
The ``openai`` model flavor within MLflow not only simplifies the logging and deployment of OpenAI models but also expands the possibilities of NLP applications:

- **Enhanced Text Analysis**: With OpenAI's models, users can perform in-depth text analysis, extract insights, and generate comprehensive summaries from large volumes of text.
- **Conversational AI**: The integration allows for the development of sophisticated conversational agents capable of understanding context, maintaining the flow of conversation, and providing human-like responses.
- **Embeddings for Advanced Retrieval**: Leverage OpenAI's models to generate embeddings that can be used for advanced document retrieval, enhancing search functionalities and content recommendations.

Key Features
------------
- **Model Logging and Deployment**: Utilize MLflow's ``save_model()`` and ``log_model()`` functions to log and deploy OpenAI models effortlessly.
- **Python Function Flavor**: The inclusion of the ``python_function`` flavor enables the models to be interpreted as generic Python functions, facilitating easy integration and inference.
- **Model Loading Capabilities**: The ``load_model()`` function provides an efficient way to load saved or logged models, offering direct access to their attributes and functionalities.

Advanced Prompt Engineering and Version Tracking with MLflow and OpenAI
-----------------------------------------------------------------------

Harnessing the Synergy of MLflow and OpenAI for Prompt Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MLflow, in conjunction with OpenAI, revolutionizes the way we approach prompt engineering for large language models (LLMs). This integration enables 
developers and data scientists to meticulously track, compare, and evaluate different versions of prompts, which are crucial in guiding LLMs to produce 
desired outcomes.

Prompt Engineering in LLM-Powered Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prompt engineering is an art and science in the realm of LLMs. It involves crafting and refining prompts that effectively direct the language model to generate 
specific, contextually relevant, and high-quality responses. With the vast capabilities of OpenAI's models, prompt engineering becomes a pivotal tool in 
harnessing their full potential.

MLflow's Role in Experimentation and Versioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Experiment Tracking**: MLflow excels in keeping detailed records of experiments, including various iterations of prompt engineering. This capability allows developers to log different prompt versions, along with corresponding model responses and performance metrics.
- **Version Comparison and Evaluation**: Through MLflow, it's straightforward to compare different versions of prompts and their effectiveness. This comparison is vital in evaluating how slight modifications in prompt structure or wording can lead to significantly different outcomes from the LLM.
- **Monitoring Deployed Models**: Once a prompt-driven model is deployed, MLflow continues to play a critical role in monitoring its performance. Any changes or updates to the prompts are tracked, ensuring full visibility and control over the deployed models.

Leveraging MLflow for Optimized Prompt Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Iterative Improvement**: MLflow's tracking system supports an iterative approach to prompt engineering. By logging each experiment, users can incrementally refine their prompts, driving towards the most effective model interaction.
- **Collaborative Experimentation**: MLflow's collaborative features enable teams to share and discuss prompt versions and experiment results, fostering a collaborative environment for prompt development.

Real-World Impact
^^^^^^^^^^^^^^^^^
In real-world applications, the ability to track and refine prompts using MLflow and OpenAI leads to more accurate, reliable, and efficient language model 
implementations. Whether in customer service chatbots, content generation, or complex decision support systems, the meticulous management of prompts 
and model versions directly translates to enhanced performance and user experience.

This integration not only simplifies the complexities of working with advanced LLMs but also opens up new avenues for innovation in NLP applications, 
ensuring that each prompt-driven interaction is as effective and impactful as possible.


Direct OpenAI Service Usage
---------------------------
Direct usage of OpenAI's service through MLflow allows for seamless interaction with the latest GPT models for a variety of NLP tasks.

.. literalinclude:: ../../../../../examples/openai/chat_completions.py
    :language: python

Azure OpenAI Service Integration
--------------------------------
The ``openai`` flavor supports logging models that use the `Azure OpenAI Service <https://azure.microsoft.com/en-us/products/ai-services/openai-service>`_. 
There are a few notable differences between the Azure OpenAI Service and the OpenAI Service that need to be considered when logging models that target Azure endpoints. 

Environment Configuration for Azure Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To successfully log a model targeting Azure OpenAI Service, specific environment variables are essential for authentication and functionality.

.. note::
    The following environment variables contain **highly sensitive access keys**. Ensure that you do not commit these values to source control or declare them in an interactive 
    environment. Environment variables should be set from within your terminal via an ``export`` command, an addition to your user profile configurations (i.e., .bashrc or .zshrc), 
    or set through your IDE's environment variable configuration. Please do not leak your credentials.

- **OPENAI_API_KEY**: The API key for the Azure OpenAI Service. This can be found in the Azure Portal under the "Keys and Endpoint" section of the "Keys and Endpoint" tab. You can use either ``KEY1`` or ``KEY2``.
- **OPENAI_API_BASE**: The base endpoint for your Azure OpenAI resource (e.g., ``https://<your-service-name>.openai.azure.com/``). Within the Azure OpenAI documentation and guides, this key is referred to as ``AZURE_OPENAI_ENDPOINT`` or simply ``ENDPOINT``.
- **OPENAI_API_VERSION**: The API version to use for the Azure OpenAI Service. More information can be found in the `Azure OpenAI documentation <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>`_, including up-to-date lists of supported versions.
- **OPENAI_API_TYPE**: If using Azure OpenAI endpoints, this value should be set to ``"azure"``.
- **DEPLOYMENT_ID**: The deployment name that you chose when you deployed the model in Azure. To learn more, visit the `Azure OpenAI deployment documentation <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal>`_.

Azure OpenAI Service in MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Integrating Azure OpenAI models within MLflow follows similar procedures to direct OpenAI service usage, with additional Azure-specific configurations.

.. literalinclude:: ../../../../../examples/openai/azure_openai.py
    :language: python

Deep Dive into MLflow and OpenAI Integration
--------------------------------------------
This integration empowers users to explore and innovate in the realm of AI:

1. **Dynamic Conversational Agents**: Develop AI agents that can engage in dynamic conversations, provide customer support, or even mimic specific personas for interactive experiences.
2. **Creative Text Generation**: Use OpenAI's models for creative writing, advertising copy generation, or even coding.
3. **Contextual Embeddings for Semantic Search**: Generate embeddings for sophisticated semantic search applications, improving information retrieval and content discoverability.

Value Proposition of OpenAI Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Ease of Use**: Simplify the complex process of working with state-of-the-art NLP models.
- **Experimentation and Tracking**: Leverage MLflow's tracking capabilities to experiment with different models and settings, ensuring optimal outcomes.
- **Scalability and Security**: With support for Azure OpenAI Service, the integration offers scalable and secure deployment options, catering to enterprise-level requirements.

Next Steps in Your NLP Journey
------------------------------
We invite you to harness the combined power of MLflow and OpenAI for developing innovative NLP applications. Whether it's creating interactive 
AI-driven platforms, enhancing data analysis with deep NLP insights, or exploring new frontiers in AI, this integration serves as a robust foundation 
for your explorations
