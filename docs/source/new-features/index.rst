New Features
============

Looking to learn about new significant releases in MLflow? 

Find out about the details of major features, changes, and deprecations below.

.. raw:: html

    <section>
        <article class="new-content-grid">
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
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Direct Access to OpenAI through the MLflow Deployments API 
                    </div>
                    <img class="card-image" src="../_static/images/logos/openai-logo.png" alt="MLflow Deployments" style="max-height: 5rem"></img>
                    <div class="body">
                        MLflow Deployments now supports direct access to OpenAI services.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.9.0">released in 2.9.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Gateway renamed to MLflow Deployments Server
                    </div>
                    <img class="card-image" src="../_static/images/logos/gateway-header-image.png" alt="MLflow Deployments"></img>
                    <div class="body">
                        The previously known feature, MLflow Gateway has been refactored to the <a href="../llms/deployments/index.html">MLflow Deployments Server</a>.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.9.0">released in 2.9.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Docs Overhaul 
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        The MLflow docs are getting a facelift with added content, tutorials, and guides. Stay tuned for further improvements to the site!
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        New Features for LLM Evaluation
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        The functionality provided for LLM evaluation in MLflow is getting greatly expanded. Check out all of the new features in the 
                        <a href="../llms/llm-evaluate/index.html">guide</a> and the <a href="../llms/llm-evaluate/notebooks/index.html">tutorials</a>.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/llm-evaluate/index.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Updated Model Registry UI
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        A new opt-in Model Registry UI has been built that uses Aliases and Tags for managing model development. See 
                       more about the new <a href="../model-registry.html#ui-workflow">UI workflow</a> in the docs.
                    </div>
                    <div class="doc"><a class="icon bell" href="../model-registry.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Spark Connect support 
                    </div>
                    <img class="card-image" src="../_static/images/logos/spark-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now log, save, and load models trained using Spark Connect. Try out Spark 3.5 and the MLflow integration today!
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        AI21 Labs added as an MLflow Gateway provider 
                    </div>
                    <img class="card-image" src="../_static/images/logos/ai21labs-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now use the MLflow AI Gateway to connect to LLMs hosted by <a href="https://www.ai21.com/">AI21 Labs</a>.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/index.html#id1">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Amazon Bedrock added as an MLflow Gateway provider 
                    </div>
                    <img class="card-image" src="../_static/images/logos/aws-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now use the MLflow AI Gateway to connect to LLMs hosted by <a href="https://aws.amazon.com/bedrock/">AWS's Bedrock</a> service.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/index.html#id1">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        PaLM 2 added as an MLflow Gateway provider 
                    </div>
                    <img class="card-image" src="../_static/images/logos/PaLM-logo.png" alt="MLflow"></img>
                    <div class="body">
                        You can now use the MLflow AI Gateway to connect to LLMs hosted by <a href="https://ai.google/discover/palm2/">Google's PaLM 2</a> service.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/index.html#id1">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Hugging Face TGI added as an MLflow Gateway provider 
                    </div>
                    <img class="card-image" src="../_static/images/logos/huggingface-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can self-host your own transformers-based models from the Hugging Face Hub and directly connect to the models with the AI Gateway
                        with <a href="https://huggingface.co/docs/text-generation-inference/index">TGI</a>.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/index.html#id1">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.8.0">released in 2.8.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        LLM evaluation viewer added to MLflow UI
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can view your LLM evaluation results directly from the MLflow UI.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/llm-evaluate/index.html#view-evaluation-results-via-the-mlflow-ui">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.7.0">released in 2.7.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Introducting the Prompt Engineering UI
                    </div>
                    <img class="card-image" src="../_static/images/intro/new_features/prompt-eng-ui.png" alt="Prompt Engineering UI" style="max-height: 5rem"></img>
                    <div class="body">
                        Link your MLflow Tracking Server with your MLflow AI Gateway Server to experiment, evaluate, and construct 
                        prompts that can be compared amongst different providers without writing a single line of code.
                    </div>
                    <div class="doc"><a class="icon bell" href="../llms/prompt-engineering/index.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.7.0">released in 2.7.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MosaicML Support in AI Gateway
                    </div>
                    <img class="card-image" src="../_static/images/logos/mosaicml-logo.svg" alt="MosaicML"></img>
                    <div class="body">
                        MosaicML has now been added to the supported providers in MLflow AI Gateway.
                        You can now seamlessly interface with managed popular models like
                        <a href="https://www.mosaicml.com/blog/mpt-30b">MPT-30B</a> and other models in the MPT family.
                    </div>
                    <div class="body">
                        Try it out today with our <a href="https://github.com/mlflow/mlflow/blob/master/examples/gateway/mosaicml">example</a>.
                    </div>
                    <div class="doc"><a href="../llms/gateway/index.html#supported-provider-models">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.7.0">released in 2.7.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Cloudflare R2 now supported as an artifact store
                    </div>
                    <img class="card-image" src="../_static/images/intro/new_features/cloudflare-logo.svg" alt="cloudflare" style="max-height: 5rem"></img>
                    <div class="body">
                        Cloudflare's R2 storage backend is now supported for use as an artifact store. To learn more about 
                        R2, read the <a href="https://developers.cloudflare.com/r2/get-started/">Cloudflare docs</a> to get more information and to explore what is possible.
                    </div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.7.0">released in 2.7.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Params support for PyFunc Models
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        PyFunc models now support passing parameters at inference time. With this new feature, 
                        you can define the allowable keys, with default values, for any parameters that you would like 
                        consumers of your model to be able to override. This is particularly useful for LLMs, where you 
                        might want to let users adjust commonly modified parameters for a model, such as token counts and temperature. 
                    </div>
                    <div class="doc"><a href="../models.html#inference-params">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.6.0">released in 2.6.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Serving support added to MLflow AI Gateway
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        The MLflow AI Gateway now supports defining an MLflow serving endpoint as provider. With this 
                        new feature, you can serve any OSS transformers model that conforms to the 
                        <a href="../llms/deployments/index.html#completions">completions</a> or <a href="../llms/deployments/index.html#embeddings">embeddings</a> route type 
                        definitions. 
                    </div>
                    <div class="body">
                        Try it out today with our end-to-end <a href="https://github.com/mlflow/mlflow/tree/master/examples/deployments/mlflow_serving">example</a>.
                    </div>
                    <div class="doc"><a href="../llms/deployments/index.html#mlflow-models">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.6.0">released in 2.6.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Introducing the MLflow AI Gateway
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        We're excited to announce the newest top-level component in the MLflow ecosystem: <strong>The AI Gateway</strong>. 
                    </div>
                    <div class="body">
                        With this new feature, you can create a single access point to many of the most popular LLM SaaS services available now, 
                        simplifying interfaces, managing credentials, and providing a unified standard set of APIs to reduce the complexity of 
                        building products and services around LLMs. 
                    </div>
                    <div class="doc"><a href="../llms/deployments/index.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.5.0">released in 2.5.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        MLflow Evaluate now supports LLMs
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now use MLflow evaluate to compare results from your favorite LLMs on a fixed prompt.
                    </div>
                    <div class="body">
                        With support for many of the standard evaluation metrics for LLMs built in directly to the API, the featured 
                        LLM modeling tasks of text summarization, text classification, question answering, and text generation allows you 
                        to view the results of submitted text to multiple models in a single UI element. 
                    </div>
                    <div class="doc"><a href="../llms/llm-evaluate/index.html">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.4.0">released in 2.4.0</a>
                    </div>
                </div>
            </div>
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Chart View added to the MLflow UI
                    </div>
                    <img class="card-image" src="../_static/images/logos/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now visualize parameters and metrics across multiple runs as a chart on the runs table.
                    </div>
                    <div class="doc"><a href="../getting-started/quickstart-2/index.html#chart-view">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.2.0">released in 2.2.0</a>
                    </div>
                </div>
            </div>
        </article>
    </section>
