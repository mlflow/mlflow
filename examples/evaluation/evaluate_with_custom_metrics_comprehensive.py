import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import os

# loading the California housing dataset
cali_housing = fetch_california_housing(as_frame=True)

# split the dataset into train and test partitions
X_train, X_test, y_train, y_test = train_test_split(
    cali_housing.data, cali_housing.target, test_size=0.2, random_state=123
)

# train the model
lin_reg = LinearRegression().fit(X_train, y_train)

# creating the evaluation dataframe
eval_data = X_test.copy()
eval_data["target"] = y_test


def metrics_only_fn(eval_df, builtin_metrics):
    """
    This example demonstrates an example custom metric function that does not
    produce any artifacts. Also notice that for computing its metrics, it can either
    directly use the eval_df or build upon existing metrics supplied by builtin_metrics
    """
    return {
        "squared_diff_plus_one": np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2),
        "sum_on_label_divided_by_two": builtin_metrics["sum_on_label"] / 2,
    }


def file_artifacts_fn(eval_df, builtin_metrics, artifacts_dir):
    """
    This example shows how you can return file paths as representation
    of the produced artifacts. For a full list of supported file extensions
    refer to https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
    """
    example_np_arr = np.array([1, 2, 3])
    np.save(os.path.join(artifacts_dir, "example.npy"), example_np_arr, allow_pickle=False)

    example_df = pd.DataFrame({"test": [2.2, 3.1], "test2": [3, 2]})
    example_df.to_csv(os.path.join(artifacts_dir, "example.csv"), index=False)
    example_df.to_parquet(os.path.join(artifacts_dir, "example.parquet"))

    example_json = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_json.update(builtin_metrics)
    with open(os.path.join(artifacts_dir, "example.json"), "w") as f:
        json.dump(example_json, f)

    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plt.savefig(os.path.join(artifacts_dir, "example.png"))
    plt.savefig(os.path.join(artifacts_dir, "example.jpeg"))

    with open(os.path.join(artifacts_dir, "example.txt"), "w") as f:
        f.write("hello world!")

    return {}, {
        "example_np_arr_from_npy_file": os.path.join(artifacts_dir, "example.npy"),
        "example_df_from_csv_file": os.path.join(artifacts_dir, "example.csv"),
        "example_df_from_parquet_file": os.path.join(artifacts_dir, "example.parquet"),
        "example_dict_from_json_file": os.path.join(artifacts_dir, "example.json"),
        "example_image_from_png_file": os.path.join(artifacts_dir, "example.png"),
        "example_image_from_jpeg_file": os.path.join(artifacts_dir, "example.jpeg"),
        "example_string_from_txt_file": os.path.join(artifacts_dir, "example.txt"),
    }


class ExampleClass:
    def __init__(self, x):
        self.x = x


def object_artifacts_fn(eval_df, builtin_metrics):
    """
    This example shows how you can return python objects as artifacts
    without the need to save them to file system.
    """
    example_np_arr = np.array([1, 2, 3])
    example_df = pd.DataFrame({"test": [2.2, 3.1], "test2": [3, 2]})
    example_dict = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_dict.update(builtin_metrics)
    example_dict_2 = '{"a": 3, "b": [1, 2, 3]}'
    example_image = plt.figure()
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    example_custom_class = ExampleClass(10)

    return {}, {
        "example_np_arr_from_obj_saved_as_npy": example_np_arr,
        "example_df_from_obj_saved_as_csv": example_df,
        "example_dict_from_obj_saved_as_json": example_dict,
        "example_image_from_obj_saved_as_png": example_image,
        "example_dict_from_json_str_saved_as_json": example_dict_2,
        "example_class_from_obj_saved_as_pickle": example_custom_class,
    }


def mixed_example_fn(eval_df, builtin_metrics, artifacts_dir):
    """
    This example mixes together some of the different ways to return metrics and artifacts
    """
    metrics = {
        "squared_diff_divided_two": np.sum(
            np.abs(eval_df["prediction"] - eval_df["target"]) ** 2 / 2
        ),
        "sum_on_label_multiplied_by_three": builtin_metrics["sum_on_label"] * 3,
    }
    example_dict = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_dict.update(builtin_metrics)
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")

    plt.savefig(os.path.join(artifacts_dir, "example2.png"))
    artifacts = {
        "example_dict_2_from_obj_saved_as_csv": example_dict,
        "example_image_2_from_png_file": os.path.join(artifacts_dir, "example2.png"),
    }
    return metrics, artifacts


with mlflow.start_run() as run:
    mlflow.sklearn.log_model(lin_reg, "model")
    model_uri = mlflow.get_artifact_uri("model")
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="target",
        model_type="regressor",
        dataset_name="cali_housing",
        evaluators=["default"],
        custom_metrics=[
            metrics_only_fn,
            file_artifacts_fn,
            object_artifacts_fn,
            mixed_example_fn,
        ],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
