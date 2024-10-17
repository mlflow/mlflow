Deep Learning
=============

The realm of deep learning has witnessed an unprecedented surge, revolutionizing numerous sectors with its ability to
process vast amounts of data and capture intricate patterns. From the real-time object detection in autonomous vehicles
to the generation of art through Generative Adversarial Networks, and from natural language processing applications in
chatbots to predictive analytics in e-commerce, deep learning models are at the forefront of today's AI-driven innovations.

In the deep learning realm, libraries such as PyTorch, Keras, Tensorflow provide handy tools to build and train deep learning
models. MLflow, on the other hand, targets the problem of experiment tracking in deep learning, including logging your
experiment setup (learning rate, batch size, etc) along with training metrics (loss, accuracy, etc) and the model
(architecture, weights, etc). MLflow provides native integrations with deep learning libraries, so you can plug MLflow
into your existing deep learning workflow with minimal changes to your code, and view your experiments in the MLflow UI.

Why MLflow for Deep Learning?
-----------------------------
MLflow offers a list of features that power your deep learning workflows:

* **Experiments Tracking**: MLflow tracks your deep learning experiments, including parameters, metrics, and models.
  Your experiments will be stored in the MLflow server, so you can compare across different experiments and share them.
* **Model Registry**: You can register your trained deep learning models in the MLflow server, so you can easily
  retrieve them later for inference.
* **Model Deployment**: After training, you can serve the trained model with MLflow as a REST API endpoint, so you can
  easily integrate it with your application.

Experiments Tracking
^^^^^^^^^^^^^^^^^^^^
Tracking is the cornerstone of the MLflow ecosystem, and especially vital for the iterative nature of deep learning:

- **Experiments and Runs**: Organize your deep learning projects into experiments, with each experiment containing multiple runs.
  Each run captures essential data like metrics at various training steps, hyperparameters, and the code state.
- **Artifacts**: Store vital outputs such as deep learning models, visualizations, or even tensorboard logs. This artifact
  repository ensures traceability and easy access.
- **Metrics at Steps**: With deep learning's iterative nature, MLflow allows logging metrics at various training steps,
  offering a granular view of the model's progress.
- **Dependencies and Environment**: Capture the computational environment, including deep learning frameworks' versions,
  ensuring reproducibility.
- **Input Examples and Model Signatures**: Define the expected format of the model's inputs, crucial for complex data like
  images or sequences.
- **UI Integration**: The enhanced UI provides a visual overview of deep learning runs, facilitating comparison and insights
  into training progress.
- **Search Functionality**: Efficiently navigate through your deep learning experiments using robust search capabilities.
- **APIs**: Interact with the tracking system programmatically, integrating deep learning workflows seamlessly.


.. |chart-comparison| raw:: html

        <div class="tracking-responsive-tab-panel">
            <div>
                <h4>Easier DL Model Comparison with Charts</h4>
                <p>Use charts to compare deep learning (DL) model training convergence easily. Quickly identify superior 
                configuration sets across training iterations.</p>
            </div>
            <img src="../_static/images/deep-learning/dl-run-selection.gif" style="width: 90%; height: auto; object-fit: cover;"/>
        </div>

.. |chart-customization| raw:: html

        <div class="tracking-responsive-tab-panel">
            <div>
                <h4>Chart Customization for DL Models</h4>
                <p>Easily customize charts for DL training run comparisons. Adjust visualizations to pinpoint optimal parameter 
                settings, displaying optimization metrics across iterations in a unified view.</p>
            </div>
            <img src="../_static/images/deep-learning/dl-run-navigation.gif" style="width: 90%; height: auto; object-fit: cover;"/>
        </div>

.. |run-comparison| raw:: html

        <div class="tracking-responsive-tab-panel">
            <div>
                <h4>Enhanced Parameter and Metric Comparison</h4>
                <p>Analyze parameter relationships from a unified interface to refine tuning parameters, optimizing your DL models efficiently.</p>
            </div>
            <img src="../_static/images/deep-learning/dl-run-comparison.gif" style="width: 90%; height: auto; object-fit: cover;"/>
        </div>

.. |parameter-evaluation| raw:: html

        <div class="tracking-responsive-tab-panel">
            <div>
                <h4>Statistical Evaluation of Categorical Parameters</h4>
                <p>Leverage boxplot visualizations for categorical parameter evaluation. Quickly discern the most effective 
                settings for hyperparameter tuning.</p>
            </div>
            <img src="../_static/images/deep-learning/dl-boxplot.gif" style="width: 90%; height: auto; object-fit: cover;"/>
        </div>

.. |realtime-tracking| raw:: html

        <div class="tracking-responsive-tab-panel">
            <div>
                <h4>Real-Time Training Tracking</h4>
                <p>Automatically monitor DL training progress over epochs with the MLflow UI. Instantly track results to validate 
                your hypotheses, eliminating constant manual updates.</p>
            </div>
            <img src="../_static/images/deep-learning/dl-tracking.gif" style="width: 90%; height: auto; object-fit: cover;"/>
        </div>


.. container:: tracking-responsive-tabs

    .. tabs::

        .. tab:: Chart Comparison

            |chart-comparison|
        
        .. tab:: Chart Customization

            |chart-customization|

        .. tab:: Run Comparison

            |run-comparison|
 
        .. tab:: Statistical Evaluation

            |parameter-evaluation|

        .. tab:: Realtime Tracking
            
            |realtime-tracking|
        

Model Registry
^^^^^^^^^^^^^^
A centralized repository for your deep learning models:

- **Versioning**: Handle multiple iterations and versions of deep learning models, facilitating comparison or reversion.
- **Annotations**: Attach notes, training datasets, or other relevant metadata to models.
- **Lifecycle Stages**: Clearly define the stage of each model version, ensuring clarity in deployment and further fine-tuning.

Model Deployment
^^^^^^^^^^^^^^^^
Transition deep learning models from training to real-world applications:

- **Consistency**: Ensure models, especially those with GPU dependencies, behave consistently across different deployment environments.
- **Docker and GPU Support**: Deploy in containerized environments, ensuring all dependencies, including GPU support, are encapsulated.
- **Scalability**: From deploying a single model to serving multiple distributed deep learning models, MLflow scales as per
  your requirements.

Native Library Support
----------------------
MLflow has native integrations with common deep learning libraries, such as PyTorch, Keras and Tensorflow, so you can plug
MLflow into your workflow easily to elevate your deep learning projects.

For detailed guide on how to integrate MLflow with these libraries, refer to the following pages:

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tensorflow/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/TensorFlow-logo.svg" alt="TensorFlow Logo"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Tensorflow library and see example notebooks that leverage
                        MLflow and Tensorflow to build deep learning workflows.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="pytorch/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/pytorch-logo.svg" alt="Pytorch Logo" style="width: 90%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the PyTorch library and see example notebooks that leverage
                        MLflow and PyTorch to build deep learning workflows.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="keras/index.html">
                   <div class="header-with-image">
                        <img src="../_static/images/logos/keras-logo.svg" alt="Keras Logo" style="width: 20%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Keras library and see example notebooks that leverage
                        MLflow and Keras to build deep learning workflows.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../models.html#spacy-spacy">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/spacy-logo.svg" alt="spaCy Logo" style="width: 60%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Spacy library and see example code.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../models.html#fastai-fastai">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/fastai-logo.png" alt="fast.ai Logo"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the FastAI library and see example code.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../llms/transformers/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace Logo"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Transformers 🤗 library and see example notebooks that leverage 
                        MLflow and Transformers to build Open-Source powered solutions.
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
    pytorch/index
