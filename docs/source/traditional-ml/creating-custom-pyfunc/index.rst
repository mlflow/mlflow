Building Custom Python Function Models with MLflow
==================================================

.. raw:: html

   <div class="no-toc"></div>

MLflow offers a wide range of pre-defined model flavors, but there are instances where you'd want to go 
beyond these and craft something tailored to your needs. That's where custom PyFuncs come in handy.

What's in this tutorial?

This guide aims to walk you through the intricacies of PyFuncs, explaining the why, the what, and the how:

- **Named Model Flavors**: Before we dive into the custom territory, it's essential to understand the existing named flavors in MLflow. These pre-defined flavors simplify model tracking and deployment, but they might not cover every use case.

- **Custom PyFuncs Demystified**: What exactly is a custom PyFunc? How is it different from the named flavors, and when would you want to use one? We'll cover:

    - **Pre/Post Processing**: Integrate preprocessing or postprocessing steps as part of your model's prediction pipeline.

    - **Unsupported Libraries**: Maybe you're using a niche or newly-released ML library that MLflow doesn't support yet. No worries, custom PyFuncs have you covered.

    - **External References**: Avert serialization issues and simplify model deployment by externalizing references.

- **Getting Started with Custom PyFuncs**: We'll begin with the simplest of examples, illuminating the core components and abstract methods essential for your custom PyFunc.

- **Tackling Unsupported Libraries**: Stepping up the complexity, we'll demonstrate how to integrate a model from an unsupported library into MLflow using custom PyFuncs.

- **Overriding Default Inference Methods**: Sometimes, the default isn't what you want. We'll show you how to override a model's inference method, for example, using ``predict_proba`` instead of ``predict``.

By the end of this tutorial, you'll have a clear understanding of how to leverage custom PyFuncs in MLflow to cater to specialized needs, 
ensuring flexibility without compromising on the ease of use.

.. toctree::
    :maxdepth: 1

    Models, Flavors, and PyFuncs in MLflow <part1-named-flavors>
    Understanding Pyfunc Components <part2-pyfunc-components>
    Full Notebooks <notebooks/index>
    