from __future__ import print_function

import os
import dill as pickle
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

from mlflow import tensorflow, pyfunc
import mlflow.pyfunc.cli
from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir

def load_pyfunc(path):
    model_fn = None
    model_dir = None
    for filename in os.listdir(path):
        if filename == "model_fn.pkl":
            with open(os.path.join(path, filename), "rb") as f:
                model_fn = pickle.load(f)
                print("found function")
        elif filename == "model_dir.pkl":
            with open(os.path.join(path, filename), "rb") as f:
                model_dir = pickle.load(f)
                print("found dir")
    return tf.estimator.Estimator(model_fn, model_dir=model_dir)


class TestModelExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        iris = datasets.load_iris()
        self._X = iris.data[:, :2]  # we only take the first two features.
        self._y = iris.target
        self._trainingFeatures = {}
        for i in range(0, 2):
            tab = str.maketrans(dict.fromkeys(' ()'))
            name = iris.feature_names[i].translate(tab)
            self._trainingFeatures[name] = iris.data[:, i:i+1]
        tf_feat_cols = []
        for col in iris.feature_names[:2]:
            tab = str.maketrans(dict.fromkeys(' ()'))
            name = col.translate(tab)
            tf_feat_cols.append(tf.feature_column.numeric_column(name))
        with tf.Session() as session:
            self._input_train = tf.estimator.inputs.numpy_input_fn(self._trainingFeatures, 
                                                            self._y, 
                                                            shuffle=False, 
                                                            batch_size=1)
        self._dnn = tf.estimator.DNNRegressor(
                                feature_columns=tf_feat_cols, 
                                hidden_units=[1])
        self._dnn.train(self._input_train, steps=100)
        self._dnn_predict = self._dnn.predict(self._input_train)
        # self._linear_lr = glm.LogisticRegression()
        # self._linear_lr.fit(self._X, self._y)
        # self._linear_lr_predict = self._linear_lr.predict(self._X)

    def test_model_save_load(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            path = tmp.path("dnn")
            tensorflow.save_model(tf_model=self._dnn, path=path)
            x = tensorflow.load_model(path)
            xpred = x.predict(self._input_train)
            saved = []
            for s in self._dnn_predict:
                saved.append(s['predictions'])
            loaded = []
            for l in xpred:
                loaded.append(l['predictions'])
            np.testing.assert_array_equal(saved, loaded)
            # sklearn should also be stored as a valid pyfunc model
            # test pyfunc compatibility
            y = pyfunc.load_pyfunc(path)
            ypred = y.predict(self._input_train)
            loaded = []
            for l in ypred:
                loaded.append(l['predictions'])
            np.testing.assert_array_equal(saved, loaded)

    def test_model_log(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            path = tmp.path("model")
            os.makedirs(path)
            model_fn_file = os.path.join(path, "model_fn.pkl")
            model_dir_file = os.path.join(path, "model_dir.pkl")
            with open(model_fn_file, "wb") as out:
                pickle.dump(self._dnn.model_fn, out)
            with open(model_dir_file, "wb") as out:
                pickle.dump(self._dnn.model_dir, out)
            tracking_dir = os.path.abspath(tmp.path("mlruns"))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            tracking.start_run()
            try:
                pyfunc.log_model(artifact_path="dnn",
                                 data_path="model",
                                 loader_module="mlflow.tensorflow")
                                 #code_path=[__file__])
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                run_id = tracking.active_run().info.run_uuid
                path = tracking._get_model_log_dir("dnn", run_id)
                print(path)
                m = Model.load(os.path.join(path, "MLmodel"))
                print(m.__dict__)
                x = pyfunc.load_pyfunc("dnn", run_id=run_id)
                xpred = x.predict(self._input_train)
                saved = []
                for s in self._dnn_predict:
                    saved.append(s['predictions'])
                loaded = []
                for l in xpred:
                    loaded.append(l['predictions'])
                np.testing.assert_array_equal(saved, loaded)
            finally:
                tracking.end_run()
                # Remove the log directory in order to avoid adding new tests to pytest...
                shutil.rmtree(tracking_dir)

    # def test_model_serve(self):
    #     with TempDir() as tmp:
    #         model_path = tmp.path("knn.pkl")
    #         with open(model_path, "wb") as f:
    #             pickle.dump(self._knn, f)
    #         path = tmp.path("knn")
    #         pyfunc.save_model(dst_path=path,
    #                           data_path=model_path,
    #                           loader_module=os.path.basename(__file__)[:-3],
    #                           code_path=[__file__],
    #                           )

    # def test_cli_predict(self):
    #     with TempDir() as tmp:
    #         model_path = tmp.path("knn.pkl")
    #         with open(model_path, "wb") as f:
    #             pickle.dump(self._knn, f)
    #         path = tmp.path("knn")
    #         pyfunc.save_model(dst_path=path,
    #                           data_path=model_path,
    #                           loader_module=os.path.basename(__file__)[:-3],
    #                           code_path=[__file__],
    #                           )
    #         input_csv_path = tmp.path("input.csv")
    #         pandas.DataFrame(self._X).to_csv(input_csv_path, header=True, index=False)
    #         output_csv_path = tmp.path("output.csv")
    #         runner = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"})
    #         result = runner.invoke(mlflow.pyfunc.cli.commands,
    #                                ['predict', '--model-path', path, '-i',
    #                                 input_csv_path, '-o', output_csv_path])
    #         print("result", result.output)
    #         print(result.exc_info)
    #         print(result.exception)
    #         assert result.exit_code == 0
    #         result_df = pandas.read_csv(output_csv_path)
    #         np.testing.assert_array_equal(result_df.values.transpose()[0],
    #                                       self._knn.predict(self._X))


if __name__ == '__main__':
    unittest.main()
