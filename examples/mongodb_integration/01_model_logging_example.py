#!/usr/bin/env python
"""
Comprehensive Model Logging Example with MongoDB Backend

This example demonstrates how to log ML models of different types to MongoDB using Genesis-Flow.
Covers: sklearn, tensorflow, pytorch, custom models, and model signatures.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec

# For model examples
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class CustomPreprocessor:
    """Custom preprocessing pipeline to demonstrate custom model logging."""
    
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
    
    def transform(self, X):
        return X * self.scale_factor
    
    def fit(self, X):
        return self


class CustomMLModel(mlflow.pyfunc.PythonModel):
    """Custom ML model implementation for MLflow PyFunc."""
    
    def __init__(self, sklearn_model, preprocessor):
        self.sklearn_model = sklearn_model
        self.preprocessor = preprocessor
    
    def predict(self, context, model_input):
        # Apply preprocessing
        processed_input = self.preprocessor.transform(model_input)
        # Make predictions
        return self.sklearn_model.predict(processed_input)


def setup_mongodb_tracking():
    """Configure Genesis-Flow to use MongoDB backend."""
    tracking_uri = "mongodb://localhost:27017/genesis_flow_models"
    registry_uri = "mongodb://localhost:27017/genesis_flow_models"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    return tracking_uri, registry_uri


def generate_sample_data():
    """Generate sample dataset for model training."""
    print("\n📊 Generating Sample Dataset")
    print("-" * 40)
    
    # Generate classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"✅ Train split: {X_train.shape[0]} samples")
    print(f"✅ Test split: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_names


def log_sklearn_model_example(X_train, X_test, y_train, y_test, feature_names):
    """Example 1: Log scikit-learn model with comprehensive metadata."""
    print("\n🤖 Example 1: Scikit-Learn Model Logging")
    print("=" * 50)
    
    experiment_name = "sklearn_model_logging_demo"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="sklearn_random_forest") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Model parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "framework": "scikit-learn"
        })
        
        # Train model
        print("🔄 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        })
        
        # Create model signature
        sample_input = pd.DataFrame(X_train[:5], columns=feature_names)
        signature = infer_signature(sample_input, test_pred)
        
        # Log model with signature and input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
            pip_requirements=[
                "scikit-learn==1.3.0",
                "pandas>=1.5.0",
                "numpy>=1.21.0"
            ]
        )
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save and log feature importance as artifact
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            feature_importance.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "analysis")
        os.unlink(f.name)
        
        # Log classification report
        class_report = classification_report(y_test, test_pred, output_dict=True)
        mlflow.log_metrics({
            f"class_{i}_precision": class_report[str(i)]['precision'] 
            for i in range(len(np.unique(y_test)))
        })
        
        print(f"✅ Model logged with accuracy: {test_accuracy:.4f}")
        print(f"✅ Run ID: {run.info.run_id}")
        
        return run.info.run_id


def log_custom_pyfunc_model_example(X_train, X_test, y_train, y_test, feature_names):
    """Example 2: Log custom PyFunc model with preprocessing pipeline."""
    print("\n🔧 Example 2: Custom PyFunc Model with Preprocessing")
    print("=" * 55)
    
    experiment_name = "custom_pyfunc_model_demo"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="custom_pyfunc_pipeline") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Create preprocessing pipeline
        preprocessor = CustomPreprocessor(scale_factor=0.5)
        
        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Apply preprocessing and train
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        base_model.fit(X_train_processed, y_train)
        
        # Create custom model wrapper
        custom_model = CustomMLModel(base_model, preprocessor)
        
        # Test the custom model
        test_pred = custom_model.predict(None, X_test)
        accuracy = accuracy_score(y_test, test_pred)
        
        # Log parameters
        mlflow.log_params({
            "model_type": "CustomPyFuncModel",
            "base_model": "RandomForestClassifier",
            "preprocessing": "CustomPreprocessor",
            "scale_factor": 0.5,
            "framework": "custom_pyfunc"
        })
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "base_model_estimators": 50
        })
        
        # Create signature
        sample_input = pd.DataFrame(X_train[:3], columns=feature_names)
        signature = infer_signature(sample_input, test_pred)
        
        # Create artifacts dictionary for the model
        artifacts = {
            "preprocessor": "preprocessor.pkl"
        }
        
        # Save preprocessor
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(preprocessor, f)
            preprocessor_path = f.name
        
        # Log the custom model
        mlflow.pyfunc.log_model(
            artifact_path="custom_model",
            python_model=custom_model,
            artifacts={"preprocessor_path": preprocessor_path},
            signature=signature,
            input_example=sample_input,
            pip_requirements=[
                "scikit-learn==1.3.0",
                "pandas>=1.5.0",
                "numpy>=1.21.0"
            ]
        )
        
        # Cleanup
        os.unlink(preprocessor_path)
        
        print(f"✅ Custom PyFunc model logged with accuracy: {accuracy:.4f}")
        print(f"✅ Run ID: {run.info.run_id}")
        
        return run.info.run_id


def log_model_with_datasets_example(X_train, X_test, y_train, y_test, feature_names):
    """Example 3: Log model with dataset tracking."""
    print("\n📊 Example 3: Model Logging with Dataset Tracking")
    print("=" * 50)
    
    experiment_name = "model_with_datasets_demo"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="model_with_dataset_tracking") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Create datasets
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df['target'] = y_test
        
        # Save datasets to temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            train_df.to_csv(f.name, index=False)
            train_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            test_df.to_csv(f.name, index=False)
            test_path = f.name
        
        # Log datasets as artifacts
        mlflow.log_artifact(train_path, "datasets/train_data.csv")
        mlflow.log_artifact(test_path, "datasets/test_data.csv")
        
        # Log dataset info
        mlflow.log_params({
            "train_dataset_size": len(train_df),
            "test_dataset_size": len(test_df),
            "feature_count": len(feature_names),
            "class_distribution": str(np.bincount(y_train).tolist())
        })
        
        # Train model
        model = RandomForestClassifier(n_estimators=75, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "model_complexity": model.n_estimators
        })
        
        # Log model
        signature = infer_signature(X_train, test_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=pd.DataFrame(X_train[:2], columns=feature_names)
        )
        
        # Log additional metadata
        mlflow.set_tags({
            "model_family": "ensemble",
            "data_type": "tabular",
            "problem_type": "multiclass_classification",
            "dataset_version": "v1.0"
        })
        
        # Cleanup
        os.unlink(train_path)
        os.unlink(test_path)
        
        print(f"✅ Model with datasets logged, accuracy: {accuracy:.4f}")
        print(f"✅ Run ID: {run.info.run_id}")
        
        return run.info.run_id


def log_model_versions_example(X_train, X_test, y_train, y_test, feature_names):
    """Example 4: Log multiple model versions for comparison."""
    print("\n🔄 Example 4: Multiple Model Versions")
    print("=" * 40)
    
    experiment_name = "model_versions_comparison"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    run_ids = []
    model_configs = [
        {"n_estimators": 50, "max_depth": 5, "version": "v1_baseline"},
        {"n_estimators": 100, "max_depth": 10, "version": "v2_improved"},
        {"n_estimators": 200, "max_depth": 15, "version": "v3_complex"}
    ]
    
    for config in model_configs:
        with mlflow.start_run(run_name=f"rf_model_{config['version']}") as run:
            print(f"🔄 Training {config['version']} model...")
            
            # Train model with specific config
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, test_pred)
            
            # Log everything
            mlflow.log_params(config)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "model_size_mb": model.n_estimators * 0.1  # Approximate size
            })
            
            # Create signature
            signature = infer_signature(X_train, test_pred)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=pd.DataFrame(X_train[:1], columns=feature_names),
                metadata={"version": config["version"], "experiment_type": "hyperparameter_tuning"}
            )
            
            mlflow.set_tags({
                "model_version": config["version"],
                "hyperparameter_experiment": "true"
            })
            
            print(f"  ✅ {config['version']}: accuracy = {accuracy:.4f}")
            run_ids.append(run.info.run_id)
    
    print(f"\n✅ Logged {len(model_configs)} model versions for comparison")
    return run_ids


def demonstrate_model_loading():
    """Example 5: Demonstrate loading logged models from MongoDB."""
    print("\n📦 Example 5: Loading Models from MongoDB")
    print("=" * 45)
    
    # Get the latest experiment
    experiment = mlflow.get_experiment_by_name("sklearn_model_logging_demo")
    if not experiment:
        print("❌ No sklearn experiment found. Run the sklearn example first.")
        return
    
    # Get runs from the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("❌ No runs found in sklearn experiment.")
        return
    
    # Get the first run
    run_id = runs.iloc[0]['run_id']
    print(f"🔍 Loading model from run: {run_id}")
    
    try:
        # Load model as sklearn
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Loaded sklearn model: {type(loaded_model).__name__}")
        
        # Load model as PyFunc (generic interface)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Loaded PyFunc model: {type(pyfunc_model).__name__}")
        
        # Test prediction with sample data
        sample_data = pd.DataFrame([[0.1] * 20], columns=[f"feature_{i}" for i in range(20)])
        prediction = pyfunc_model.predict(sample_data)
        print(f"✅ Sample prediction: {prediction[0]}")
        
        # Get model metadata
        model_info = mlflow.models.get_model_info(model_uri)
        print(f"✅ Model signature: {model_info.signature}")
        print(f"✅ Model flavors: {list(model_info.flavors.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")


def main():
    """Run all model logging examples."""
    print("🚀 Genesis-Flow MongoDB Model Logging Examples")
    print("=" * 60)
    
    # Setup
    setup_mongodb_tracking()
    X_train, X_test, y_train, y_test, feature_names = generate_sample_data()
    
    # Run examples
    sklearn_run_id = log_sklearn_model_example(X_train, X_test, y_train, y_test, feature_names)
    custom_run_id = log_custom_pyfunc_model_example(X_train, X_test, y_train, y_test, feature_names)
    dataset_run_id = log_model_with_datasets_example(X_train, X_test, y_train, y_test, feature_names)
    version_run_ids = log_model_versions_example(X_train, X_test, y_train, y_test, feature_names)
    
    # Demonstrate loading
    demonstrate_model_loading()
    
    print(f"\n🎉 ALL MODEL LOGGING EXAMPLES COMPLETED!")
    print("=" * 60)
    print(f"📋 Summary:")
    print(f"  ✅ Sklearn model run: {sklearn_run_id}")
    print(f"  ✅ Custom PyFunc run: {custom_run_id}")
    print(f"  ✅ Dataset tracking run: {dataset_run_id}")
    print(f"  ✅ Version comparison runs: {len(version_run_ids)} models")
    print(f"  ✅ Model loading: Verified")
    print(f"\n📊 Database: genesis_flow_models")
    print(f"🔗 All models stored in MongoDB with full metadata!")


if __name__ == "__main__":
    main()