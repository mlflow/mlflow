MLflow OpenAI Flavor
====================

.. attention::
    The ``openai`` flavor is under active development and is marked as Experimental. Public APIs are 
    subject to change, and new features may be added as the flavor evolves.

Introduction
------------

**OpenAI's GPT Models** represent a significant leap in natural language processing (NLP) capabilities. 
The Generative Pre-trained Transformer (GPT) models are renowned for 
their ability to generate human-like text, comprehend complex queries, summarize extensive documents, 
and much more. OpenAI has been at the forefront of NLP technology, offering models that are 
versatile and widely applicable in various domains.

Leveraging MLflow's robust experiment tracking and model management framework, the integration with 
OpenAI's models enables practitioners to efficiently utilize these advanced NLP tools in their 
projects. From simple text generation to complex conversational AI applications, the MLflow-OpenAI 
integration brings a new level of ease and effectiveness to managing these powerful models.

The integration includes:

- **Text Analysis and Generation**: Utilizing models like GPT-3.5 and GPT-4 for diverse text-related tasks.
- **Conversational AI**: Exploring the capabilities of the Chat Completions API for interactive, context-aware applications.
- **Embeddings Generation**: Corpus and text embeddings generation capabilities for advanced document retrieval use cases.

What makes this Integration so Special?
---------------------------------------
The combination of MLflow's experiment tracking and model management with OpenAI's cutting-edge NLP 
models unlocks new potential for AI applications. 
This integration simplifies the process of deploying, monitoring, and scaling NLP models, making it 
accessible to a broader range of users and use cases.

Unique Aspects of OpenAI's Models: Reinforcement Learning from Human Feedback (RLHF)
------------------------------------------------------------------------------------

RLHF in GPT Models
^^^^^^^^^^^^^^^^^^
One of the defining features of OpenAI's GPT models is their training process, particularly the use of Reinforcement Learning from Human Feedback 
(RLHF). This methodology sets GPT models apart from traditional language models in several ways (although they are not the only organization to use this 
strategy, it is a key process component that greatly helps to enhance the quality of their services).

The RLHF Process
""""""""""""""""
1. **Supervised Fine-Tuning (SFT)**: Initially, GPT models undergo supervised fine-tuning using a large dataset of text. This process imparts the basic understanding of language and context.

2. **Reward Modeling (RM)**: Human trainers review the model's outputs and rate them based on criteria such as relevance, accuracy, and safety. This feedback is used to create a 'reward model'—a system that evaluates the quality of the model's responses.

3. **Proximal Policy Optimization (PPO)**: In this stage, the model is trained using reinforcement learning techniques, guided by the reward model. The model learns to generate responses that are more aligned with the values and preferences as judged by human trainers.

4. **Iterative Improvement**: The model undergoes continuous refinement through human feedback, ensuring that it evolves and adapts to produce responses that are not only accurate but also contextually appropriate and safe.

Why RLHF Matters
""""""""""""""""
- **Human-Like Responses**: RLHF enables GPT models to generate responses that closely mimic human thought processes, making them more relatable and effective in practical applications.
- **Safety and Relevance**: Through human feedback, GPT models learn to avoid generating harmful or irrelevant content, thereby increasing their reliability and applicability.
- **Cost-Effective Training**: RLHF allows for more efficient and cost-effective training compared to extensively curating fine-tuned data for each specific application.


.. figure:: ../../_static/images/tutorials/llms/RLHF-architecture.png
   :alt: MLflow OpenAI Integration Overview
   :width: 100%
   :align: center

   Simplified overview of RLHF

Features
--------

With the MLflow OpenAI flavor, users can:

- **Save** and **log** OpenAI models within MLflow using :py:func:`mlflow.openai.save_model` and :py:func:`mlflow.openai.log_model`.
- Seamlessly track detailed experiments, including **parameters**, **prompts**, and **artifacts** associated with model runs.
- `Deploy <../../deployment/index.html>`_ OpenAI models for various NLP applications with ease.
- Utilize :py:class:`mlflow.pyfunc.PythonModel` for flexible Python function inference, enabling custom and innovative ML solutions.

What can you do with OpenAI and MLflow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The integration of OpenAI's advanced NLP models with MLflow's robust model management capabilities opens up a vast array of potential real-world applications. Here are some powerful and impactful use cases:

- **Automated Customer Support**: Develop sophisticated chatbots that understand and respond to customer inquiries in a human-like manner, significantly improving customer service efficiency and satisfaction.
  
- **Content Generation and Curation**: Automatically generate high-quality, contextually relevant content for articles, blogs, or social media posts. Curate content by summarizing and categorizing large volumes of text data, enhancing content management strategies.
  
- **Language Translation Services**: Create advanced translation tools that not only convert text from one language to another but also capture nuances, idioms, and cultural context, bridging communication gaps more effectively.
  
- **Sentiment Analysis for Market Research**: Analyze customer feedback, social media posts, or product reviews to gauge public sentiment about brands, products, or services, providing valuable insights for marketing and product development teams.
  
- **Personalized Education and Training Tools**: Develop AI-driven educational platforms that can adapt content and teaching styles to individual learning preferences, making education more engaging and effective.
  
- **Legal and Compliance Document Analysis**: Automate the review and analysis of legal documents, contracts, and compliance materials, increasing accuracy and reducing the time and resources required for legal workflows.
  
- **Healthcare Assistance and Research**: Assist in medical research by summarizing and analyzing medical literature, patient records, or clinical trial data, contributing to faster and more informed decision-making in healthcare.
  
- **Financial Analysis and Forecasting**: Leverage NLP models to analyze financial reports, market trends, and news articles, providing deeper insights and predictions for investment strategies and economic forecasting.

With MLflow's integration, these applications not only benefit from the linguistic prowess of OpenAI's models but also gain from streamlined tracking, version control, and deployment processes. This synergy empowers developers and businesses to build sophisticated, AI-driven solutions that address complex challenges and create new opportunities in various industries.


Deployment Made Easy
^^^^^^^^^^^^^^^^^^^^

Deploying OpenAI models becomes a breeze with MLflow. Functions like :py:func:`mlflow.openai.load_model` and :py:func:`mlflow.pyfunc.load_model` facilitate easy model serving. 
Discover more about `deploying models with MLflow <../../deployment/index.html>`_, explore the `deployments API <../../cli.html#mlflow-deployments>`_, 
and learn about `starting a local model serving endpoint <../../cli.html#mlflow-models-serve>`_ to fully leverage the deployment capabilities of MLflow.

Getting Started with the MLflow OpenAI Flavor - Tutorials and Guides
--------------------------------------------------------------------

Below, you will find a number of guides that focus on different ways that you can leverage the power of the `openai` library, leveraging MLflow's 
APIs for tracking and inference capabilities. 

Introductory Tutorial
^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/openai-quickstart.html">
                    <div class="header">
                        OpenAI Quickstart
                    </div>
                    <p>
                        Learn the very basics of using the OpenAI package with MLflow with some simple prompt engineering and a fun use case to get 
                        started with this powerful integration.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/openai/notebooks/openai-quickstart.ipynb" class="notebook-download-btn">Download the Introductory Notebook</a><br>


Advanced Tutorials
^^^^^^^^^^^^^^^^^^

In these tutorials, the topics aren't any more advanced than the introductory tutorial, but the teaching text is much less. More code; less chatter.
If you're new to this flavor, please start with the Introductory Tutorial above, as it has information about environment configurations that you'll need 
to understand in order to get these notebooks to work.

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/openai-chat-completions.html">
                    <div class="header">
                        OpenAI ChatCompletions
                    </div>
                    <p>
                        Learn how to leverage the ChatCompletions endpoint in the OpenAI flavor to create a useful text messaging screening tool within MLflow.
                    </p>
                </a>
            </div>
        </article>
    </section>


Download the Advanced Tutorial Notebooks
----------------------------------------

To download the advanced OpenAI tutorial notebooks to run in your environment, click the respective links below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/openai/notebooks/openai-chat-completions.ipynb" class="notebook-download-btn">Download the ChatCompletions Notebook</a><br>
    

.. toctree::
    :maxdepth: 2
    :hidden:

    notebooks/openai-quickstart.ipynb
    notebooks/openai-chat-completions.ipynb

`Detailed Documentation <guide/index.html>`_
--------------------------------------------

To learn more about the details of the MLflow flavor for OpenAI, delve into the comprehensive guide below.

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Comprehensive Guide</a>

.. toctree::
   :maxdepth: 1
   :hidden:

   guide/index.rst
