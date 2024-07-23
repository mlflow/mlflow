MLflow Keras 3.0 Integration
============================

Introduction
------------

Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. 
It was developed with a focus on enabling fast experimentation.

Keras 3.0 makes it possible to run Keras workflows on top of TensorFlow, JAX, and PyTorch. 
It also enables you to seamlessly integrate Keras components (like layers, models, or metrics) as part of 
low-level TensorFlow, JAX, and PyTorch workflows.

MLflow provides built-in support for Keras 3.0 workflows. It provides a callback that allows you to 
log parameters and metrics during model training. Model logging is not currently supported.

5 Minute Quick Start with MLflow + Keras 3.0
--------------------------------------------

To get a quick overview of how to use MLflow + Keras 3.0, please read the quickstart guide.


.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="quickstart/quickstart_keras.html">
                    <div class="header">
                        Get Started with Keras 3.0 + MLflow
                    </div>
                    <p>
                        Learn how to leverage the MLflow Keras callback for tracking experiments, as well as how to customize it.
                    </p>
                </a>
            </div>
        </article>
    </section>


.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart/quickstart_keras.ipynb
