MLflow Documentation
====================

MLflow is an open source platform for managing the end-to-end machine learning lifecycle.
It tackles four primary functions:

* Tracking experiments to record and compare parameters and results (:ref:`tracking`).
* Packaging ML code in a reusable, reproducible form in order to share with other data
  scientists or transfer to production (:ref:`projects`).
* Managing and deploying models from a variety of ML libraries to a variety of model serving and
  inference platforms (:ref:`models`).
* Providing a central model store to collaboratively manage the full lifecycle of an MLflow Model,
  including model versioning, stage transitions, and annotations (:ref:`registry`).

MLflow is library-agnostic. You can use it with any machine learning library, and in any
programming language, since all functions are accessible through a :ref:`rest-api`
and :ref:`CLI<cli>`. For convenience, the project also includes a :ref:`python-api`, :ref:`R-api`,
and :ref:`java_api`.

Get started using the :ref:`quickstart` or by reading about the :ref:`key concepts<concepts>`.

.. toctree::
    :hidden:

    what-is-mlflow
    concepts
    tracking
    python_api/index

.. grid::  1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0

    .. grid-item-card::  API reference
        :link: python_api/index.html

        The reference guide contains a detailed description of the MLflow API.
        The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.

    .. grid-item-card::  Tutorials

        ...


    .. grid-item-card::  Another card

        ...

    .. grid-item-card::  Another card

        ...
