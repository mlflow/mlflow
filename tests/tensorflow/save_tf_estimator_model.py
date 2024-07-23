import argparse
import collections
import copy
import os
import pickle

import iris_data_utils
import numpy as np
import pandas as pd
import pandas.testing
import tensorflow as tf
from tensorflow import estimator as tf_estimator

import mlflow
from mlflow.utils.file_utils import TempDir

assert mlflow.__version__ == "1.28.0"


parser = argparse.ArgumentParser()

parser.add_argument("--tracking_uri")
parser.add_argument("--mlflow_repo_path")
parser.add_argument("--model_type")
parser.add_argument("--task_type")
parser.add_argument("--save_path")

args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    [
        "path",
        "meta_graph_tags",
        "signature_def_key",
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
    ],
)


def save_tf_iris_model(tmp_path):
    # Following code from
    # https://github.com/tensorflow/models/blob/v1.13.0/samples/core/get_started/premade_estimator.py
    train_x, train_y = iris_data_utils.load_data()[0]

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    estimator = tf_estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
    )

    # Train the Model.
    batch_size = 100
    train_steps = 1000
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps,
    )

    # Generate predictions from the model
    predict_x = {
        "SepalLength": [5.1, 5.9, 6.9],
        "SepalWidth": [3.3, 3.0, 3.1],
        "PetalLength": [1.7, 4.2, 5.4],
        "PetalWidth": [0.5, 1.5, 2.1],
    }

    estimator_preds = estimator.predict(
        lambda: iris_data_utils.eval_input_fn(predict_x, None, batch_size)
    )

    estimator_preds_dict = next(estimator_preds)
    for row in estimator_preds:
        for key in row.keys():
            estimator_preds_dict[key] = np.vstack((estimator_preds_dict[key], row[key]))

    # Building a pandas DataFrame out of the prediction dictionary.
    estimator_preds_df = copy.deepcopy(estimator_preds_dict)
    for col in estimator_preds_df.keys():
        if all(len(element) == 1 for element in estimator_preds_df[col]):
            estimator_preds_df[col] = estimator_preds_df[col].ravel()
        else:
            estimator_preds_df[col] = estimator_preds_df[col].tolist()

    # Building a DataFrame that contains the names of the flowers predicted.
    estimator_preds_df = pandas.DataFrame.from_dict(data=estimator_preds_df)
    estimator_preds_results = [
        iris_data_utils.SPECIES[id[0]] for id in estimator_preds_dict["class_ids"]
    ]
    estimator_preds_results_df = pd.DataFrame({"predictions": estimator_preds_results})

    # Define a function for estimator inference
    feature_spec = {}
    for name in my_feature_columns:
        feature_spec[name.key] = tf.Variable([], dtype=tf.float64, name=name.key)

    receiver_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = os.path.join(tmp_path, "saved_model")
    os.makedirs(saved_estimator_path)
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(
            data=predict_x, columns=[name.key for name in my_feature_columns]
        ),
        expected_results_df=estimator_preds_results_df,
        raw_results=estimator_preds_dict,
        raw_df=estimator_preds_df,
    )


def save_tf_categorical_model(tmp_path):
    path = os.path.join(args.mlflow_repo_path, "tests/datasets/uci-autos-imports-85.data")
    # Order is important for the csv-readers, so we use an OrderedDict here
    defaults = collections.OrderedDict(
        [("body-style", [""]), ("curb-weight", [0.0]), ("highway-mpg", [0.0]), ("price", [0.0])]
    )
    types = collections.OrderedDict((key, type(value[0])) for key, value in defaults.items())
    df = pd.read_csv(path, names=list(types.keys()), dtype=types, na_values="?")
    df = df.dropna()

    # Extract the label from the features dataframe
    y_train = df.pop("price")

    # Create the required input training function
    training_features = df.to_dict(orient="series")

    # Create the feature columns required for the DNNRegressor
    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        key="body-style", vocabulary_list=body_style_vocab
    )
    feature_columns = [
        tf.feature_column.numeric_column(key="curb-weight"),
        tf.feature_column.numeric_column(key="highway-mpg"),
        # Since this is a DNN model, convert categorical columns from sparse to dense.
        # Then, wrap them in an `indicator_column` to create a one-hot vector from the input
        tf.feature_column.indicator_column(body_style),
    ]

    # Build a DNNRegressor, with 20x20-unit hidden layers, with the feature columns
    # defined above as input
    estimator = tf_estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)

    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(training_features, y_train, 1), steps=10
    )
    estimator_preds = np.array(
        [
            s["predictions"]
            for s in estimator.predict(
                lambda: iris_data_utils.eval_input_fn(training_features, None, 1)
            )
        ]
    ).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {
        "body-style": tf.Variable([], dtype=tf.string, name="body-style"),
        "curb-weight": tf.Variable([], dtype=tf.float64, name="curb-weight"),
        "highway-mpg": tf.Variable([], dtype=tf.float64, name="highway-mpg"),
    }
    receiver_fn = tf_estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = os.path.join(tmp_path, "saved_model")
    os.makedirs(saved_estimator_path)
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(training_features),
        expected_results_df=estimator_preds_df,
        raw_results=None,
        raw_df=None,
    )


if args.model_type == "iris":
    gen_model_fn = save_tf_iris_model
elif args.model_type == "categorical":
    gen_model_fn = save_tf_categorical_model
else:
    raise ValueError("Illegal argument.")

output_data_file_path = "output_data.pkl"

with TempDir() as tmp:
    saved_model = gen_model_fn(tmp.path())

    if args.task_type == "log_model":
        with mlflow.start_run() as run:
            mlflow.tensorflow.log_model(
                tf_saved_model_dir=saved_model.path,
                tf_meta_graph_tags=saved_model.meta_graph_tags,
                tf_signature_def_key=saved_model.signature_def_key,
                artifact_path="model",
                extra_pip_requirements=["protobuf<4.0.0"],
            )
            run_id = run.info.run_id
    elif args.task_type == "save_model":
        mlflow.tensorflow.save_model(
            tf_saved_model_dir=saved_model.path,
            tf_meta_graph_tags=saved_model.meta_graph_tags,
            tf_signature_def_key=saved_model.signature_def_key,
            path=args.save_path,
            extra_pip_requirements=["protobuf<4.0.0"],
        )
        run_id = None
    else:
        raise ValueError("Illegal argument.")

output_data_info = (
    saved_model.inference_df,
    saved_model.expected_results_df,
    saved_model.raw_results,
    saved_model.raw_df,
    run_id,
)

with open(output_data_file_path, "wb") as f:
    pickle.dump(output_data_info, f)
