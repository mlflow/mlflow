New Features
============

Looking to learn about new significant releases in MLflow? 

Find out about the details of major features, changes, and deprecations below.

.. raw:: html

    <section>
        <article class="new-content-grid">
            <div class="grid-card">
                <div class="content-container">
                    <div class="header" style="height: 4rem">
                        LangGraph Support
                    </div>
                    <img class="card-image" src="../_static/images/logos/langgraph-logo.png" style="max-height: 8rem" alt="LangGraph"></img>
                    <div class="body">
                        <a href="../llms/langchain/index.html">LangGraph</a>, the GenAI Agent authoring framework from LangChain, is now natively supported in MLflow using the <a href="../model/models-from-code.html">Models from Code</a> feature.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.16.0">released in 2.16.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header" style="height: 4rem">
                        AutoGen Tracing Integration
                    </div>
                    <img class="card-image" src="../_static/images/logos/autogen-logo.svg" style="max-height: 8rem" alt="AutoGen"></img>
                    <div class="body">
                        <a href="https://microsoft.github.io/autogen/">AutoGen</a>, a multi-turn agent framework from Microsoft, now has integrated automatic tracing integration with MLflow.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.16.0">released in 2.16.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header" style="height: 4rem">
                        LlamaIndex Support
                    </div>
                    <img class="card-image" src="../_static/images/logos/llamaindex-logo.svg" style="max-height: 8rem" alt="LlamaIndex"></img>
                    <div class="body">
                        <a href="../llms/llama-index/index.html">LlamaIndex</a>, the popular RAG and Agent authoring framework now has native support within MLflow for application logging and full support for tracing.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.15.0">released in 2.15.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header" style="height: 4rem">
                        MLflow Tracing
                    </div>
                    <img class="card-image" src="../_static/images/llms/tracing/trace-feature-card.png" style="max-height: 8rem" alt="MLflow Tracing"></img>
                    <div class="body">
                        <a href="../llms/tracing/index.html">MLflow Tracing</a> is powerful tool designed to enhance your ability to monitor, analyze, and debug GenAI applications by allowing you to inspect the intermediate outputs generated as your application handles a request.
                    </div>
                    <div class="doc"><a href="../llms/tracing/index.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.14.0">released in 2.14.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Unity Catalog Integration in MLflow Deployments Server
                    </div>
                    <img class="card-image" src="../_static/images/logos/unity-catalog-logo.png" alt="Unity Catalog"></img>
                    <div class="body">
                        The MLflow Deployments server now has <a href="../llms/deployments/uc_integration.html">an integration with Unity Catalog</a>, allowing you to leverage registered functions as tools for enhancing your chat application.
                    </div>
                    <div class="doc"><a href="../llms/deployments/uc_integration.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.14.0">released in 2.14.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        OpenAI Autologging
                    </div>
                    <img class="card-image" src="../_static/images/logos/openai-logo.svg" alt="OpenAI" style="max-height: 5rem"></img>
                    <div class="body">
                        Autologging support has now been added for the <a href="../llms/openai/guide/index.html#openai-autologging">OpenAI model flavor</a>. With this feature, MLflow will automatically log a model upon calling the OpenAI API.
                    </div>
                    <div class="doc"><a href="../llms/openai/guide/index.html#openai-autologging">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.14.0">released in 2.14.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Enhanced Code Dependency Management
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        The <code>infer_code_path</code> option when logging a model will determine which additional code modules are needed, ensuring the consistency between the training environment and production.
                    </div>
                    <div class="doc"><a href="../model/dependencies.html#saving-extra-code-dependencies-with-an-mlflow-model-automatic-inference">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.13.0">released in 2.13.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Fully Customizable GenAI Metrics
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        <p>The <a href="../llms/llm-evaluate/index.html">MLflow evaluate API</a> 
                        now supports fully customizable system prompts to create entirely novel evaluation metrics for GenAI use cases.</p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.13.0">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Together.ai added to MLflow Deployments Server
                    </div>
                    <img class="card-image" src="../_static/images/logos/togetherai-logo.png" alt="Together.ai"></img>
                    <div class="body">
                        <p>The <a href="../llms/deployments/index.html">MLflow Deployments Server</a> can now 
                        accept <a href="https://www.together.ai/">together.ai</a> endpoints.  
                        </p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.12.2">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Streaming Output support for LangChain and Python Models
                    </div>
                    <img class="card-image" src="../_static/images/logos/langchain-logo.png" alt="LangChain"></img>
                    <div class="body">
                        <p>
                        LangChain models and custom Python Models now support a <b>predict_stream</b> API, allowing for generator return types for streaming outputs.
                        </p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.12.2">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        LangChain Models as Code
                    </div>
                    <img class="card-image" src="../_static/images/logos/langchain-logo.png" alt="LangChain"></img>
                    <div class="body">
                        <p>The <a href="../llms/langchain/index.html">LangChain flavor</a> 
                        in MLflow now supports defining a model as a code file to simplify logging and loading of LangChain models.</p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.12.2">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Aynchronous Artifact Logging
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        <p>
                        MLflow now supports asynchronous artifact logging, allowing for faster and more efficient logging of models with many artifacts.
                        </p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.12.2">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Transformers Embedding Model Standardization
                    </div>
                    <img class="card-image" src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace transformers"></img>
                    <div class="body">
                        <p>The <a href="../llms/transformers/index.html">transformers flavor</a> 
                        has received standardization support for embedding models.</p>
                        <p>
                        Embedding models now return a standard <b>llm/v1/embeddings</b> output format to conform to OpenAI embedding response structures.
                        </p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.12.2">released in 2.12.2</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Transformers Feature Enhancements
                    </div>
                    <img class="card-image" src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace transformers"></img>
                    <div class="body">
                        <p>The <a href="../llms/transformers/index.html">transformers flavor</a> 
                        in MLflow has gotten a significant feature overhaul.</p>
                        <ul>
                            <li>All supported pipeline types can now be logged without restriction</li>
                            <li>Pipelines using foundation models can now be logged without copying the large model weights</li>
                        </ul>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        PEFT (Parameter-Efficient Fine-Tuning) support
                    </div>
                    <img class="card-image" src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace Logo"></img>
                    <div class="body">
                        MLflow now natively supports <a href="../llms/transformers/guide/index.html#peft-models-in-mlflow-transformers-flavor">PEFT (Parameter-Efficient Fine-Tuning)</a>
                        models in the Transformers flavor. PEFT unlocks significantly more efficient model fine-tuning processes such as LoRA, QLoRA, and Prompt Tuning. Check out 
                        <a href="../llms/transformers/tutorials/fine-tuning/transformers-peft.html">the new QLoRA fine-tuning tutorial</a> to learn how to 
                        build your own cutting-edge models with MLflow and PEFT!
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/transformers/guide/index.html#peft-models-in-mlflow-transformers-flavor">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        ChatModel Pyfunc Subclass Added
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        <p>
                        OpenAI-compatible chat models are now easier than ever to build in MLflow! 
                        <a href="../python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatModel">ChatModel</a> is a new
                        Pyfunc subclass that makes it easy to deploy and serve chat models with MLflow.</p>

                        <p>
                        Check out the
                        <a href="../llms/transformers/tutorials/conversational/pyfunc-chat-model.html">new tutorial</a> 
                        on building an OpenAI-compatible chat model using TinyLlama-1.1B-Chat!</p>
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Overhaul of MLflow Tracking UI for Deep Learning workflows
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        We've listened to your feedback and have put in a huge amount of new UI features designed to empower and 
                        simplify the process of evaluating DL model training runs. Be sure to upgrade your tracking server and 
                        benefit from all of the new UI enhancements today!
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Automated model checkpointing for Deep Learning model training
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        When performing training of Deep Learning models with <a href="../python_api/mlflow.pytorch.html#mlflow.pytorch.autolog">PyTorch Lightning</a> 
                        or <a href="../python_api/mlflow.tensorflow.html#mlflow.tensorflow.autolog">Tensorflow with Keras</a>, model checkpoint saving 
                        is enabled, allowing for state storage during long-running training events and the ability to resume if 
                        an issue is encountered during training. 
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Mistral AI added as an MLflow Deployments Provider
                    </div>
                    <img class="card-image" src="../_static/images/logos/mistral-ai-logo.svg" alt="Mistral AI" style="max-height: 5rem"></img>
                    <div class="body">
                        The <a href="../llms/deployments/index.html">MLflow Deployments Server</a> can now 
                        accept <a href="https://mistral.ai/">Mistral AI</a> endpoints. Give their models a try today! 
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Keras 3 is now supported in MLflow
                    </div>
                    <img class="card-image" src="../_static/images/logos/keras-logo.svg" alt="Keras"></img>
                    <div class="body">
                        You can now log and deploy models in the new <a href="https://keras.io/keras_3/">Keras 3 format</a>, allowing you 
                        to work with TensorFlow, Torch, or JAX models with a new high-level, easy-to-use suite of APIs.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow now has support for OpenAI SDK 1.x
                    </div>
                    <img class="card-image" src="../_static/images/logos/openai-logo.svg" alt="OpenAI" style="max-height: 5rem"></img>
                    <div class="body">
                        We've updated flavors that interact with the OpenAI SDK, bringing full support for the API changes with the 1.x release.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.11.0">released in 2.11.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Site Overhaul 
                    </div>
                    <img class="card-image" src="../_static/images/logos/homepage.png" alt="MLflow"></img>
                    <div class="body">
                        MLflow has a new <a href=https://mlflow.org>homepage</a> that has been completely modernized. Check it out today!
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.10.0">released in 2.10.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        LangChain Autologging Support
                    </div>
                    <img class="card-image" src="../_static/images/logos/langchain-logo.png" alt="LangChain" style="max-height: 5rem"></img>
                    <div class="body">
                        Autologging support for <a href="../llms/langchain/index.html">LangChain</a> is now available. Try it out the next time 
                        that you're building a Generative AI application with Langchain!
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.10.0">released in 2.10.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Object and Array Support for complex Model Signatures 
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        Complex input types for <a href="../models.html#model-signature-and-input-example">model signatures</a> are now supported with native 
                        support of Array and Object types.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.10.0">released in 2.10.0</a>
                    </div>
                </div>
            </div>
        </article>
    </section>
