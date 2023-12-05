Deep Learning
=============

The realm of deep learning has witnessed an unprecedented surge, revolutionizing numerous sectors with its ability to 
process vast amounts of data and capture intricate patterns. From the real-time object detection in autonomous vehicles 
to the generation of art through Generative Adversarial Networks, and from natural language processing applications in 
chatbots to predictive analytics in e-commerce, deep learning models are at the forefront of today's AI-driven innovations.

MLflow acknowledges the profound impact and complexity of deep learning. With a keen focus on the unique challenges 
posed by deep learning workflows, such as iterative model training and hyperparameter tuning, MLflow introduces a robust 
suite of tools specifically designed for these advanced models. MLflow helps to facilitate seamless model development, 
ensuring reproducibility, and provides enhanced monitoring capabilities with the concept of 'steps' for recording metrics at 
various training iterations, with integrated UI features that enable you to easily visualize the iterative improvements 
of key metrics during training epochs. 

Key Benefits:
-------------
* **Iterative Model Training**: With the concept of 'steps', MLflow allows users to log metrics at various training iterations, offering a granular view of the model's progress.
* **Reproducibility**: Ensure that every model training run can be replicated with the exact same conditions.
* **Scalability**: Handle projects ranging from small-scale models to enterprise-level deployments with ease.
* **Traceability**: Keep track of every detail, from hyperparameters to the final model output.

Deep Autologging Integrations
-----------------------------
One of the standout features of MLflow's deep learning support is its deep autologging integrations. These integrations automatically 
capture and log intricate details during the training of deep learning models, ensuring that every nuance, from model parameters to 
evaluation metrics, is meticulously recorded. This is especially prominent in frameworks like TensorFlow, PyTorch Lightning, base PyTorch, 
and Keras, making the iterative training process more insightful and manageable.

Native Library Support
----------------------
Deep learning in MLflow is enriched by its native support for a number of the most popular libraries. The native integration with 
each of these libraries within MLflow help to streamline and simplify the training process, as well as saving, logging, loading, and 
representing models as generic Python functions for inference use anywhere. 

Opting for these native integrations brings forth a myriad of advantages:

* **Auto-logging Capabilities**: Automatically capture details without manual intervention.
* **Custom Serialization**: Streamline the model saving and loading process with custom methods tailored for each library.
* **Unified Interface**: Regardless of the underlying library, interact with a consistent MLflow interface.

The officially supported integrations for deep learning libraries in MLflow encompass:

.. raw:: html

    <section>
        <div class="logo-grid">
            <a href="../models.html#pytorch-pytorch">
                <div class="logo-card">
                    <img src="../_static/images/logos/pytorch-logo.svg" alt="pytorch Logo"/>
                </div>
            </a>
            <a href="../models.html#keras-keras">
                <div class="logo-card">
                    <img src="../_static/images/logos/keras-logo.svg" alt="keras Logo"/>
                </div>
            </a>
            <a href="../models.html#tensorflow-tensorflow">
                <div class="logo-card">
                    <img src="../_static/images/logos/TensorFlow-logo.svg" alt="TensorFlow Logo"/>
                </div>
            </a>
            <a href="../models.html#spacy-spacy">
                <div class="logo-card">
                    <img src="../_static/images/logos/spacy-logo.svg" alt="spaCy Logo"/>
                </div>
            </a>
            <a href="../models.html#fastai-fastai">
                <div class="logo-card">
                    <img src="../_static/images/logos/fastai-logo.png" alt="fast.ai Logo"/>
                </div>
            </a>
        </div>
    </section>

Harness the power of these integrations and elevate your deep learning projects with MLflow's comprehensive support.
For detailed guide on how to integrate MLflow with these libraries, refer to the following pages:

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tensorflow/index.html">
                    <div class="header">
                        Tensorflow
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Tensorflow library and see example notebooks that leverage
                        MLflow and Tensorflow to build deep learning workflows.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    tensorflow/index

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="keras/index.html">
                    <div class="header">
                        Keras
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Keras library and see example notebooks that leverage
                        MLflow and Keras to build deep learning workflows.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    keras/index
    tensorflow/index

MLflow Tracking for Deep Learning
---------------------------------
Tracking remains a cornerstone of the MLflow ecosystem, especially vital for the iterative nature of deep learning:

- **Experiments and Runs**: Organize your deep learning projects into experiments, with each experiment containing multiple runs. Each run captures essential data like metrics at various training steps, hyperparameters, and the code state.
- **Artifacts**: Store vital outputs such as deep learning models, visualizations, or even tensorboard logs. This artifact repository ensures traceability and easy access.
- **Metrics at Steps**: With deep learning's iterative nature, MLflow allows logging metrics at various training steps, offering a granular view of the model's progress.
- **Dependencies and Environment**: Capture the computational environment, including deep learning frameworks' versions, ensuring reproducibility.
- **Input Examples and Model Signatures**: Define the expected format of the model's inputs, crucial for complex data like images or sequences.
- **UI Integration**: The enhanced UI provides a visual overview of deep learning runs, facilitating comparison and insights into training progress.
- **Search Functionality**: Efficiently navigate through your deep learning experiments using robust search capabilities.
- **APIs**: Interact with the tracking system programmatically, integrating deep learning workflows seamlessly.


Model Registry
--------------
A centralized repository for your deep learning models:

- **Versioning**: Handle multiple iterations and versions of deep learning models, facilitating comparison or reversion.
- **Annotations**: Attach notes, training datasets, or other relevant metadata to models.
- **Lifecycle Stages**: Clearly define the stage of each model version, ensuring clarity in deployment and further fine-tuning.

Deployment for Deep Learning Models
-----------------------------------
Transition deep learning models from training to real-world applications:

- **Consistency**: Ensure models, especially those with GPU dependencies, behave consistently across different deployment environments.
- **Docker and GPU Support**: Deploy in containerized environments, ensuring all dependencies, including GPU support, are encapsulated.
- **Scalability**: From deploying a single model to serving multiple distributed deep learning models, MLflow scales as per your requirements.
