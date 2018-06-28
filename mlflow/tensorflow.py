"""MLflow integration for TensorFlow.

Manages logging and loading TensorFlow models as Python Functions. You are expected to save your own
``saved_models`` and pass their paths to ``log_saved_model()`` 
so that MLflow can track the models. 

In order to load the model to predict on it again, you can call
``model = mlflow.pyfunc.load_pyfunc(saved_model_dir)``, followed by 
``prediction= model.predict(pandas DataFrame)`` in order to obtain a prediction in a pandas DataFrame.

Note that the loaded PyFunc model does not expose any APIs for model training.
"""

from __future__ import absolute_import

import os

import pandas
import tensorflow as tf

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


class _TFWrapper(object):
    """
    Wrapper class that creates a predict function such that 
    predict(data: pandas.DataFrame) -> pandas.DataFrame
    """
    def __init__(self, saved_model_dir):
        model = Model.load(os.path.join(saved_model_dir, "MLmodel"))
        assert "tensorflow" in model.flavors
        if not "signature_def_key" in model.flavors["tensorflow"]:
            self._signature_def_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        else:
            self._signature_def_key = model.flavors["tensorflow"]["signature_def_key"]
        self._saved_model_dir = model.flavors["tensorflow"]["saved_model_dir"]

    def predict(self, df):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, 
                                                        [tf.saved_model.tag_constants.SERVING], 
                                                        self._saved_model_dir)
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, 
                                                                      self._signature_def_key)

            # Determining output tensors.
            fetch_mapping = {sigdef_output: graph.get_tensor_by_name(tnsr_info.name)
                             for sigdef_output, tnsr_info in sig_def.outputs.items()}

            # Build the feed dict, mapping input tensors to DataFrame column values.
            # We assume that input arguments to the signature def correspond to DataFrame column
            # names
            feed_dict = {graph.get_tensor_by_name(tnsr_info.name): df[sigdef_input].values
                        for sigdef_input, tnsr_info in sig_def.inputs.items()}
            raw_preds = sess.run(fetch_mapping, feed_dict=feed_dict)
            pred_dict = {fetch_name: list(values) for fetch_name, values in raw_preds.items()}
            return pandas.DataFrame(data=pred_dict)


def log_saved_model(saved_model_dir, signature_def_key, artifact_path):
    """Log a TensorFlow model as an MLflow artifact for the current run.

    :param saved_model_dir: Directory where the exported tf model is saved.
    :param signature_def_key: Which signature definition to use when loading the model again. See https://www.tensorflow.org/serving/signature_defs for details.
    :param artifact_path: Path (within the artifact directory for the current run) to which artifacts of the model will be saved.
    """
    run_id = mlflow.tracking.active_run().info.run_uuid
    mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.tensorflow")
    mlflow_model.add_flavor("tensorflow", 
                            saved_model_dir=saved_model_dir, 
                            signature_def_key=signature_def_key)
    mlflow_model.save(os.path.join(saved_model_dir, "MLmodel"))
    mlflow.tracking.log_artifacts(saved_model_dir, artifact_path)


def load_pyfunc(saved_model_dir):
    """Load model stored in python-function format.
    The loaded model object exposes a ``predict(pandas DataFrame)`` method that returns a Pandas DataFrame 
    containing the model's inference output on an input DataFrame.
    
    :param saved_model_dir: Directory where the model is saved.
    :rtype: Pyfunc format model with function `model.predict(pandas DataFrame) -> pandas DataFrame)`.

    """
    return _TFWrapper(saved_model_dir)
