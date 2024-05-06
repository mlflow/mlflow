Traditional ML
==============

In the dynamic landscape of machine learning, traditional techniques remain foundational, playing pivotal roles across various industries 
and research institutions. From the precision of classification algorithms in healthcare diagnostics to the predictive prowess of regression 
models in finance, and from the forecasting capabilities of time-series analyses in supply chain management to the insights drawn from 
statistical modeling in social sciences, these core methodologies underscore many of the technological advancements we witness today.

MLflow recognizes the enduring significance of traditional machine learning. Designed with precision and a deep understanding of the 
challenges and intricacies faced by data scientists and ML practitioners, MLflow offers a comprehensive suite of tools tailor-made for 
these classic techniques. This platform not only streamlines the model development and deployment processes but also ensures reproducibility, 
scalability, and traceability.

As we delve further, we'll explore the multifaceted functionalities MLflow offers, showcasing how it enhances the efficacy, reliability, 
and insights derived from traditional ML models. Whether you're a seasoned expert looking to optimize workflows or a newcomer eager to make 
a mark, MLflow stands as an invaluable ally in your machine learning journey.

Native Library Support
----------------------
There are a number of natively supported traditional ML libraries within MLflow. Throughout the documentation, you may see these referred to as 
"flavors", as they are specific implementations of native support for saving, logging, loading, and generic python function representation for 
the models that are produced from these libraries. 

There are distinct benefits to using the native versions of these implementations, as many have auto-logging functionality built in, as well as 
specific custom handling with serialization and deserialization that can greatly simplify your MLOps experiences when using these libraries. 

The officially supported integrations for traditional ML libraries include:

.. raw:: html

    <section>
        <div class="logo-grid">
            <a href="../models.html#scikit-learn-sklearn">
                <div class="logo-card">
                    <img src="../_static/images/logos/scikit-learn-logo.svg" alt="scikit learn"/>
                </div>
            </a>
            <a href="../models.html#xgboost-xgboost">
                <div class="logo-card">
                    <img src="../_static/images/logos/xgboost-logo.svg" alt="XGBoost Logo"/>
                </div>
            </a>
            <a href="../models.html#spark-mllib-spark">
                <div class="logo-card">
                    <img src="../_static/images/logos/spark-logo.svg" alt="Spark Logo"/>
                </div>
            </a>
            <a href="../models.html#lightgbm-lightgbm">
                <div class="logo-card">
                    <img src="../_static/images/logos/lightgbm-logo.png" alt="LightGBM Logo"/>
                </div>
            </a>
            <a href="../models.html#catboost-catboost">
                <div class="logo-card">
                    <img src="../_static/images/logos/catboost-logo.png" alt="CatBoost Logo"/>
                </div>
            </a>
            <a href="../models.html#statsmodels-statsmodels">
                <div class="logo-card">
                    <img src="../_static/images/logos/statsmodels-logo.svg" alt="Statsmodels Logo"/>
                </div>
            </a>
            <a href="../models.html#prophet-prophet">
                <div class="logo-card">
                    <img src="../_static/images/logos/prophet-logo.png" alt="Prophet Logo"/>
                </div>
            </a>
        </div>
    </section>


Tutorials and Guides
--------------------

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="hyperparameter-tuning-with-child-runs/index.html">
                    <div class="header">
                        Hyperparameter Tuning with MLflow and Optuna
                    </div>
                    <p>
                        Explore the integration of MLflow Tracking with Optuna for hyperparameter tuning. Dive into the capabilities of MLflow, 
                        understand parent-child run relationships, and compare different tuning runs to optimize model performance.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="creating-custom-pyfunc/index.html">
                    <div class="header">
                        Custom Pyfunc Models with MLflow
                    </div>
                    <p>
                        Dive deep into the world of MLflow's Custom Pyfunc. Starting with basic model definitions, embark on a journey that
                        showcases the versatility and power of Pyfunc. From simple mathematical curves to complex machine learning integrations,
                        discover how Pyfunc offers standardized, reproducible, and efficient workflows for a variety of use cases.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="serving-multiple-models-with-pyfunc/index.html">
                    <div class="header">
                        Multi-Model Endpoints with PyFunc
                    </div>
                    <p>
                        Dive deep into custom multi-model inference via MLflow's custom PyFunc models. Learn how to
                        simplify low-latency inference by passing additional inference parameters to a simple custom PyFunc implementation. 
                        This tutorial can serve as a jumping off point for many multi-model endpoing (MME) use cases!
                    </p>
                </a>
            </div>
        </article>
    </section>


.. toctree::
    :maxdepth: 1
    :hidden:

    hyperparameter-tuning-with-child-runs/index
    creating-custom-pyfunc/index
    serving-multiple-models-with-pyfunc/index

MLflow Tracking
---------------
Tracking is central to the MLflow ecosystem, facilitating the systematic organization of experiments and runs:

- **Experiments and Runs**: Each experiment encapsulates a specific aspect of your research, and each experiment can house multiple runs. Runs document critical data like metrics, parameters, and the code state.
- **Artifacts**: Store crucial output from runs, be it models, visualizations, datasets, or other metadata. This repository of artifacts ensures traceability and easy access.
- **Metrics and Parameters**: By allowing users to log parameters and metrics, MLflow makes it straightforward to compare different runs, facilitating model optimization.
- **Dependencies and Environment**: The platform automatically captures the computational environment, ensuring that experiments are reproducible across different setups.
- **Input Examples and Model Signatures**: These features allow developers to define the expected format of the model's inputs, making validation and debugging more straightforward.
- **UI Integration**: The integrated UI provides a visual overview of all runs, enabling easy comparison and deeper insights.
- **Search Functionality**: Efficiently sift through your experiments using MLflow's robust search functionality.
- **APIs**: Comprehensive APIs are available, allowing users to interact with the tracking system programmatically, integrating it into existing workflows.

MLflow Recipes
---------------
Recipes in MLflow are predefined templates tailored for specific tasks:

- **Reduced Boilerplate**: These templates help eliminate repetitive setup or initialization code, speeding up development.
- **Best Practices**: MLflow's recipes are crafted keeping best practices in mind, ensuring that users are aligned with industry standards right from the get-go.
- **Customizability**: While recipes provide a structured starting point, they're designed to be flexible, accommodating tweaks and modifications as needed.

MLflow Evaluate
---------------
Ensuring model quality is paramount:

- **Auto-generated Metrics**: MLflow automatically evaluates models, providing key metrics for regression (like RMSE, MAE) and classification (such as F1-score, AUC-ROC).
- **Visualization**: Understand your model better with automatically generated plots. For instance, MLflow can produce confusion matrices, precision-recall curves, and more for classification tasks.
- **Extensibility**: While MLflow provides a rich set of evaluation tools out of the box, it's also designed to accommodate custom metrics and visualizations.

Model Registry
--------------
This feature acts as a catalog for models:

- **Versioning**: As models evolve, keeping track of versions becomes crucial. The Model Registry handles versioning, ensuring that users can revert to older versions or compare different iterations.
- **Annotations**: Models in the registry can be annotated with descriptions, use-cases, or other relevant metadata.
- **Lifecycle Stages**: Track the stage of each model version, be it 'staging', 'production', or 'archived'. This ensures clarity in deployment and maintenance processes.

Deployment
----------
MLflow simplifies the transition from development to production:

- **Consistency**: By meticulously recording dependencies and the computational environment, MLflow ensures that models behave consistently across different deployment setups.
- **Docker Support**: Facilitate deployment in containerized environments using Docker, encapsulating all dependencies and ensuring a uniform runtime environment.
- **Scalability**: MLflow is designed to accommodate both small-scale deployments and large, distributed setups, ensuring that it scales with your needs.
