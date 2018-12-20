Hyperparameter Tuning Example
------------------------------

Example of image classification with MLflow and Keras.

This example builds an image classifier for flower photos available from tensorflow.org. It uses
VGG16 and InceptionV3 models from Keras with an option to use pretrained weights. The image
preprocessing required  in order to train or score the model is packaged with the model using custom
python function.

The MLflow model produced by running this example can be deployed at any of supported mlflow
endpoints. It accepts pandas DataFrame with a single column containing the (jpeg) image as base64
encoded binary data.



Running this Example
^^^^^^^^^^^^^^^^^^^^

You can run the example as a standard MLflow project.


.. code:: bash

    mlflow run -e train examples/flower_classifier

Will download the training dataset from tensorflow org, train a classifier using Keras and log
result with Mlflow.


You can experiment with the model paramaters and compare these results by using ``mlflow ui``.
