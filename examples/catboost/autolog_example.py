"""
Example demonstrating MLflow autologging for CatBoost models.

This script shows how to use mlflow.catboost.autolog() to automatically log:
- Model parameters
- Per-iteration metrics
- Trained model artifacts
- Dataset inputs (training and evaluation data)

Includes examples for both CatBoostClassifier and CatBoostRegressor.
"""

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

import mlflow

# Enable CatBoost autologging
# This will automatically log parameters, metrics, models, and datasets for all CatBoost fit() calls
mlflow.catboost.autolog()


def classifier_example():
    """Example of autologging with CatBoostClassifier for binary classification."""
    print("\n" + "=" * 60)
    print("CatBoostClassifier Autologging Example")
    print("=" * 60)

    # Load the breast cancer dataset from sklearn
    # Binary classification: 569 samples, 30 features (malignant vs benign)
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    # Split into train (80%) and evaluation (20%) sets
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create CatBoost Pool objects (best practice for CatBoost)
    # Pool objects efficiently handle data and allow for feature names, categorical features, etc.
    train_pool = Pool(
        data=X_train,
        label=y_train,
        feature_names=dataset.feature_names.tolist(),  # Include feature names for interpretability
    )
    eval_pool = Pool(
        data=X_eval,
        label=y_eval,
        feature_names=dataset.feature_names.tolist(),
    )

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Evaluation set: {X_eval.shape[0]} samples")

    # Initialize classifier with parameters suitable for the dataset
    # Include custom metrics: F1, MCC, AUC, PRAUC
    model = CatBoostClassifier(
        iterations=50,  # More iterations for better convergence
        learning_rate=0.1,
        depth=4,  # Deeper trees for more complex patterns
        custom_metric=["F1", "MCC", "AUC", "PRAUC"],  # Additional metrics to track
        verbose=False,  # Suppress CatBoost training output
        allow_writing_files=False,  # Prevent CatBoost from writing temporary files
    )

    # Start MLflow run
    with mlflow.start_run(run_name="catboost_classifier_autolog") as run:
        # Fit the model using Pool objects - autolog will automatically log:
        # - All model parameters (iterations, learning_rate, depth, etc.)
        # - Per-iteration metrics (Logloss, F1, MCC, AUC, PRAUC on train and eval sets)
        # - The trained model artifact
        # - Training and evaluation datasets
        model.fit(train_pool, eval_set=eval_pool)

        # Make predictions
        predictions = model.predict(X_eval)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Predictions: {predictions}")
        print(f"\nView this run in the MLflow UI at: {mlflow.get_tracking_uri()}")


def regressor_example():
    """Example of autologging with CatBoostRegressor for regression."""
    print("\n" + "=" * 60)
    print("CatBoostRegressor Autologging Example")
    print("=" * 60)

    # Load the diabetes dataset from sklearn
    # Regression: 442 samples, 10 features (predicting disease progression)
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target

    # Split into train (80%) and evaluation (20%) sets
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create CatBoost Pool objects (best practice for CatBoost)
    # Pool objects efficiently handle data and allow for feature names, categorical features, etc.
    train_pool = Pool(
        data=X_train,
        label=y_train,
        feature_names=dataset.feature_names,  # Include feature names for interpretability
    )
    eval_pool = Pool(
        data=X_eval,
        label=y_eval,
        feature_names=dataset.feature_names,
    )

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Evaluation set: {X_eval.shape[0]} samples")

    # Initialize regressor with parameters suitable for the dataset
    # Include custom metrics: R2, MAPE
    model = CatBoostRegressor(
        iterations=50,  # More iterations for better convergence
        learning_rate=0.1,
        depth=4,  # Deeper trees for more complex patterns
        custom_metric=["R2", "MAPE"],  # Additional metrics to track
        verbose=False,  # Suppress CatBoost training output
        allow_writing_files=False,  # Prevent CatBoost from writing temporary files
    )

    # Start MLflow run
    with mlflow.start_run(run_name="catboost_regressor_autolog") as run:
        # Fit the model using Pool objects - autolog will automatically log:
        # - All model parameters (iterations, learning_rate, depth, etc.)
        # - Per-iteration metrics (RMSE, R2, MAPE on train and eval sets)
        # - The trained model artifact
        # - Training and evaluation datasets
        model.fit(train_pool, eval_set=eval_pool)

        # Make predictions
        predictions = model.predict(X_eval)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Predictions: {predictions}")
        print(f"\nView this run in the MLflow UI at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    # Set the MLflow tracking URI (defaults to local file-based store)
    mlflow.set_tracking_uri("file:./mlruns")

    # Run both examples
    classifier_example()
    regressor_example()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nTo view the logged runs, start the MLflow UI:")
    print("uv run mlflow ui --backend-store-uri ./mlruns")
    print("\nThen navigate to http://localhost:5000")
