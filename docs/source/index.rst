MLflow: A Tool for Managing the Machine Learning Lifecycle
==========================================================

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in
handling the complexities of the machine learning process. Mlflow focuses on the full lifecycle for
machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.


Welcome to the Mlflow Documentation
-----------------------------------

.. raw:: html

    <section class="card-list">
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="introduction/index.html">
                    <h1>
                        Introduction to MLflow
                    </h1>
                </a>
                <img src="_static/MLflow-logo-final-black.png" alt="MLflow logo" class="card-header-image"></img>
                <h2>
                    Learn about how you can leverage MLflow to simplify your MLOps workloads.
                </h2>
                <p>Discover the core components of MLflow</p>
                <ul>
                    <li>
                        <a class="card-link" href="tracking.html">
                            Tracking
                        </a>
                    </li>
                    <li>
                        <a class="card-link" href="llm-tracking.html">
                            LLM Tracking
                        </a>
                    </li>
                    <li>
                        <a class="card-link" href="gateway/index.html">
                            AI Gateway
                        </a>
                    </li>
                    <li>
                        <a class="card-link" href="model-registry.html">
                            Model Registry
                        </a>
                    </li>
                    <li>
                        <a class="card-link" href="recipes.html">
                            Recipes
                        </a>
                    </li>
                </ul>
            </header>
            <div class="tags">
                <div><a href="introduction/index.html">What is MLflow?</a></div>
                <div><a href="concepts.html">MLflow Core Concepts</a></div>
            </div>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="tutorials/index.html">
                    <h1>
                        MLflow Tutorials
                    </h1>
                </a>
                <h2>
                    Get started with MLflow by diving into our step-by-step tutorials.
                </h2>
                <p>Whether you're new to MLflow or a seasoned user, this is a great place to start.</p>
                <p>From new features, to enhancements to existing tooling, learn in a hands-on and
                    guided step-by-step fashion with these tutorials.</p>
            </header>
            <div class="tags">
                <div><a class="bell" href="tutorials/introductory/logging-first-model/index.html">Logging your first MLflow Model</a></div>
            </div>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="guides/index.html">
                    <h1>
                        MLflow Guides
                    </h1>
                </a>
                <h2>
                    Learn in-depth processes from working examples focused on real-world MLOps tasks.
                </h2>
                <p>Explore MLflow features in greater depth with our task-oriented guides.</p>
                <p>Learn best practices of using MLflow to simplify and provide production-readiness to your ML workflows.</p>
            </header>
            <div class="tags">
                <div><a class="bell" href="guides/introductory/hyperparameter-tuning-with-child-runs/index.html">Hyperparameter Tuning with MLflow and Optuna</a></div>
                <div><a href="guides/introductory/deploy-model-to-kubernetes/index.html">Deploy a MLflow Model to Kubernetes</a></div>
            </div>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="https://github.com/mlflow/mlflow/tree/master/examples">
                    <h1>
                        MLflow Examples
                    </h1>
                </a>
                <img src="_static/images/intro/github-mark.svg" alt="GitHub logo" class="card-header-image"></img>
                <h2>
                    Get reference code examples.<br>No frills, just code.
                </h2>
                <p>Prefer code to tutorials and guides?</p>
                <p>Go straight to our GitHub repository to find examples for using the components of MLflow with supported popular ML libraries.</p>
            </header>
            <div class="tags">
                <div>
                    <a class="github" href="https://github.com/mlflow/mlflow/blob/master/examples/transformers/MLFlow_X_HuggingFace_Finetune_a_text_classification_model.ipynb">
                        Fine-tuning a transformers model with MLflow
                    </a>
                </div>
                <div>
                    <a class="github" href="https://github.com/mlflow/mlflow/tree/master/examples/gateway/mlflow_serving">
                        Using AI Gateway with MLflow served models
                    </a>
                </div>
                <div>
                    <a class="github" ref="https://github.com/mlflow/mlflow/tree/master/examples/evaluation">
                        How to use MLflow Evaluate
                    </a>
                </div>
                <div>
                    <a class="github" href="https://github.com/mlflow/mlflow/blob/master/examples/langchain/retrieval_qa_chain.py">
                        Langchain Retrieval QA Chain with MLflow
                    </a>
                </div>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="http://localhost:63342/mlflow/docs/build/html/python_api/index.html">
                    <h1>
                        Python API Docs
                    </h1>
                </a>
                <img src="_static/images/intro/python-logo-generic.svg" alt="Python Logo" class="card-header-image"></img>
                <h2>
                    References to all components of the MLflow Python APIs.
                </h2>
                <p>
                    Learn more about:
                </p>
                <ul>
                    <li>Function signatures</li>
                    <li>Plugin support</li>
                    <li>Integration points</li>
                </ul>
            </header>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="http://localhost:63342/mlflow/docs/build/html/r_api/index.html">
                    <h1>
                        R API Docs
                    </h1>
                </a>
                <img src="_static/images/intro/r-logo.svg" alt="R Logo" class="card-header-image"></img>
                <h2>
                    References to all components of the MLflow R APIs.
                </h2>
                <p>
                    Learn how to use MLflow with R
                </p>
            </header>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="http://localhost:63342/mlflow/docs/build/html/java_api/index.html">
                    <h1>
                        Java API Docs
                    </h1>
                </a>
                <img src="_static/images/intro/Java-Logo.svg" alt="Java Logo" class="card-header-image" style="max-height: 4.5rem;"></img>
                <h2>
                    References to all components of the MLflow Java APIs.
                </h2>
                <p>
                    Learn how to use MLflow with JVM-compatible languages
                </p>
            </header>
        </article>
    </section>


New Features and Notable Changes
--------------------------------

.. raw:: html

    <section>
        <article class="new-content-grid">
            <div class="grid-card">
                <div class="content-container">
                    <div class="header">
                        Introducting the Prompt Engineering UI
                    </div>
                    <img class="card-image" src="_static/images/intro/new_features/prompt-eng-ui.png" alt="Prompt Engineering UI" style="max-height: 5rem"></img>
                    <div class="body">
                        Link your MLflow Tracking Server with your MLflow AI Gateway Server to experiment, evaluate, and construct 
                        prompts that can be compared amongst different providers without writing a single line of code.
                    </div>
                    <div class="doc"><a href="llms/prompt-engineering.html">Learn more</a></div>
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
                    <img class="card-image" src="_static/images/intro/new_features/mosaicml-logo.svg" alt="MosaicML"></img>
                    <div class="body">
                        MosaicML has now been added to the supported providers in MLflow AI Gateway.
                        You can now seamlessly interface with managed popular models like
                        <a href="https://www.mosaicml.com/blog/mpt-30b">MPT-30B</a> and other models in the MPT family.
                    </div>
                    <div class="body">
                        Try it out today with our <a href="https://github.com/mlflow/mlflow/blob/master/examples/gateway/mosaicml">example</a>.
                    </div>
                    <div class="doc"><a href="gateway/index.html#supported-provider-models">Learn more</a></div>
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
                    <img class="card-image" src="_static/images/intro/new_features/cloudflare-logo.svg" alt="cloudflare" style="max-height: 5rem"></img>
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
                    <img class="card-image" src="_static/images/intro/new_features/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        PyFunc models now support passing parameters at inference time. With this new feature, 
                        you can define the allowable keys, with default values, for any parameters that you would like 
                        consumers of your model to be able to override. This is particularly useful for LLMs, where you 
                        might want to let users adjust commonly modified parameters for a model, such as token counts and temperature. 
                    </div>
                    <div class="doc"><a href="models.html#inference-params">Learn more</a></div>
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
                    <img class="card-image" src="_static/images/intro/new_features/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        The MLflow AI Gateway now supports defining an MLflow serving endpoint as provider. With this 
                        new feature, you can serve any OSS transformers model that conforms to the 
                        <a href="gateway/index.html#completions">completions</a> or <a href="gateway/index.html#embeddings">embeddings</a> route type 
                        definitions. 
                    </div>
                    <div class="body">
                        Try it out today with our end-to-end <a href="https://github.com/mlflow/mlflow/tree/master/examples/gateway/mlflow_serving">example</a>.
                    </div>
                    <div class="doc"><a href="gateway/index.html#mlflow-models">Learn more</a></div>
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
                    <img class="card-image" src="_static/images/intro/new_features/mlflow-logo.svg" alt="Mlflow"></img>
                    <div class="body">
                        We're excited to announce the newest top-level component in the MLflow ecosystem: <strong>The AI Gateway</strong>. 
                    </div>
                    <div class="body">
                        With this new feature, you can create a single access point to many of the most popular LLM SaaS services available now, 
                        simplifying interfaces, managing credentials, and providing a unified standard set of APIs to reduce the complexity of 
                        building products and services around LLMs. 
                    </div>
                    <div class="doc"><a href="gateway/index.html">Learn more</a></div>
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
                    <img class="card-image" src="_static/images/intro/new_features/mlflow-logo.svg" alt="MLflow"></img>
                    <div class="body">
                        You can now use MLflow evaluate to compare results from your favorite LLMs on a fixed prompt.
                    </div>
                    <div class="body">
                        With support for many of the standard evaluation metrics for LLMs built in directly to the API, the featured 
                        LLM modeling tasks of text summarization, text classification, question answering, and text generation allows you 
                        to view the results of submitted text to multiple models in a single UI element. 
                    </div>
                    <div class="doc"><a href="models.html#model-evaluation-llms">Learn more</a></div>
                    <div class="tag">
                        <a href="https://github.com/mlflow/mlflow/releases/tag/v2.4.0">released in 2.4.0</a>
                    </div>
                </div>
            </div>
        </article>
    </section>


.. toctree::
    :maxdepth: 1
    :hidden:

    introduction/index
    tutorials/index
    guides/index
    quickstart
    quickstart_mlops
    tutorials-and-examples/index
    concepts
    tracking
    llm-tracking
    projects
    models
    model-registry
    recipes
    gateway/index
    llms/prompt-engineering
    plugins
    auth/index
    cli
    search-runs
    search-experiments
    python_api/index
    R-api
    java_api/index
    rest-api
    docker
    community-model-flavors
