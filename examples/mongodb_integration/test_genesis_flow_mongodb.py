#!/usr/bin/env python3
"""
Genesis-Flow MongoDB Integration Test

This example demonstrates how to use Genesis-Flow with MongoDB as the tracking store,
eliminating the need for a separate MLflow server.

Prerequisites:
1. MongoDB running (local or Azure Cosmos DB)
2. Genesis-Flow installed
3. Optional: Azure Blob Storage for artifacts

Usage:
    python test_genesis_flow_mongodb.py
"""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def setup_mongodb_tracking():
    """Configure MLflow to use MongoDB tracking store."""
    
    # MongoDB connection configurations
    mongodb_configs = {
        "local": "mongodb://localhost:27017/genesis_flow_test",
        "azure_cosmos": "mongodb+srv://username:password@cluster.cosmos.azure.com/genesis_flow_test"
    }
    
    # Choose configuration (default to local for testing)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mongodb_configs["local"])
    
    print(f"🔗 Setting up Genesis-Flow with MongoDB tracking URI: {tracking_uri}")
    
    # Genesis-Flow already has MongoDB support built-in, no need for registry reloading
    
    # Set tracking URI - Genesis-Flow will handle MongoDB integration directly
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set registry URI to use the same MongoDB backend
    mlflow.set_registry_uri(tracking_uri)
    
    # Optional: Set artifact root (can be local, Azure Blob, or S3)
    artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "file:///tmp/genesis_flow_artifacts")
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = artifact_root
    
    print(f"📁 Artifact storage: {artifact_root}")
    
    return tracking_uri, artifact_root


def create_sample_dataset():
    """Create a sample dataset for ML experiments."""
    print("📊 Creating sample dataset...")
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame for better handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✅ Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")
    
    return df


def test_experiment_management():
    """Test Genesis-Flow experiment management with MongoDB."""
    print("\n🧪 Testing Experiment Management...")
    
    # Create experiment with Genesis-Flow
    experiment_name = f"genesis_flow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            tags={
                "framework": "genesis-flow",
                "storage": "mongodb",
                "test_type": "integration",
                "created_at": datetime.now().isoformat()
            }
        )
        print(f"✅ Experiment created: {experiment_name} (ID: {experiment_id})")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        return experiment_name, experiment_id
        
    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")
        return None, None


def test_model_training_and_logging(df, experiment_name):
    """Test model training and logging with Genesis-Flow."""
    print("\n🤖 Testing Model Training and Logging...")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"🔄 Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_run") as run:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param, value in params.items():
                    mlflow.log_param(param, value)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("n_samples", len(X_train))
            mlflow.log_metric("n_features", X_train.shape[1])
            
            # Log model with Genesis-Flow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"genesis_flow_{model_name}",
                signature=mlflow.models.infer_signature(X_train, y_pred)
            )
            
            # Create and log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log additional artifacts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(classification_report(y_test, y_pred))
                f.flush()
                mlflow.log_artifact(f.name, "reports/classification_report.txt")
                os.unlink(f.name)
            
            # Log tags
            mlflow.set_tags({
                "model_type": model_name,
                "algorithm": type(model).__name__,
                "framework": "scikit-learn",
                "genesis_flow": "true"
            })
            
            results.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                "accuracy": accuracy,
                "training_time": training_time,
                "model_uri": model_info.model_uri
            })
            
            print(f"✅ {model_name} logged - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
    
    return results


def test_model_loading_and_inference(results):
    """Test model loading and inference with Genesis-Flow."""
    print("\n🔮 Testing Model Loading and Inference...")
    
    for result in results:
        model_name = result["model_name"]
        model_uri = result["model_uri"]
        
        try:
            print(f"🔄 Loading {model_name} from Genesis-Flow...")
            
            # Load model using Genesis-Flow
            loaded_model = mlflow.sklearn.load_model(model_uri)
            
            # Test inference
            test_data = np.random.randn(5, 20)  # 5 samples, 20 features
            predictions = loaded_model.predict(test_data)
            probabilities = loaded_model.predict_proba(test_data) if hasattr(loaded_model, 'predict_proba') else None
            
            print(f"✅ {model_name} loaded successfully")
            print(f"   Predictions: {predictions}")
            if probabilities is not None:
                print(f"   Probabilities shape: {probabilities.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")


def test_experiment_search_and_comparison():
    """Test experiment search and run comparison with Genesis-Flow."""
    print("\n🔍 Testing Experiment Search and Comparison...")
    
    try:
        # Get current experiment
        current_experiment = mlflow.get_experiment_by_name(mlflow.get_experiment(mlflow.active_run().info.experiment_id).name) if mlflow.active_run() else None
        
        if current_experiment:
            experiment_id = current_experiment.experiment_id
        else:
            # Get the most recent experiment
            experiments = mlflow.search_experiments()
            if experiments:
                experiment_id = experiments[0].experiment_id
            else:
                print("❌ No experiments found")
                return
        
        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
        
        print(f"✅ Found {len(runs)} runs in experiment")
        
        if len(runs) > 0:
            print("\n📊 Top runs by accuracy:")
            for i, (idx, run) in enumerate(runs.head(3).iterrows()):
                accuracy = run.get('metrics.accuracy', 'N/A')
                model_type = run.get('tags.model_type', 'Unknown')
                training_time = run.get('metrics.training_time', 'N/A')
                print(f"  {i+1}. {model_type}: Accuracy={accuracy}, Time={training_time}s")
        
        return runs
        
    except Exception as e:
        print(f"❌ Failed to search experiments: {e}")
        return None


def test_genesis_flow_features():
    """Test specific Genesis-Flow features."""
    print("\n🚀 Testing Genesis-Flow Specific Features...")
    
    # Test direct MongoDB connection (this is handled internally by Genesis-Flow)
    try:
        # Get tracking store info
        tracking_uri = mlflow.get_tracking_uri()
        print(f"✅ Tracking URI: {tracking_uri}")
        
        # Check if we're using MongoDB
        if "mongodb" in tracking_uri.lower():
            print("✅ Using MongoDB tracking store (Genesis-Flow)")
        else:
            print("ℹ️  Using file-based tracking store")
        
        # Test artifact storage
        with mlflow.start_run(run_name="genesis_flow_feature_test") as run:
            # Create a test artifact
            test_data = {"message": "Genesis-Flow MongoDB integration test", "timestamp": datetime.now().isoformat()}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                import json
                json.dump(test_data, f, indent=2)
                f.flush()
                
                # Log artifact
                mlflow.log_artifact(f.name, "test_artifacts/test_data.json")
                os.unlink(f.name)
            
            # Log some metrics to test MongoDB storage
            mlflow.log_metrics({
                "test_metric_1": 0.95,
                "test_metric_2": 42.0,
                "test_metric_3": 3.14159
            })
            
            # Log parameters
            mlflow.log_params({
                "test_param_1": "genesis_flow",
                "test_param_2": "mongodb",
                "test_param_3": True
            })
            
            print(f"✅ Genesis-Flow features test completed - Run ID: {run.info.run_id}")
    
    except Exception as e:
        print(f"❌ Genesis-Flow features test failed: {e}")


def main():
    """Main test function."""
    print("🎯 Genesis-Flow MongoDB Integration Test")
    print("=" * 50)
    
    # Setup MongoDB tracking
    tracking_uri, artifact_root = setup_mongodb_tracking()
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Test experiment management
    experiment_name, experiment_id = test_experiment_management()
    
    if experiment_name:
        # Test model training and logging
        results = test_model_training_and_logging(df, experiment_name)
        
        # Test model loading and inference
        test_model_loading_and_inference(results)
        
        # Test experiment search and comparison
        runs_df = test_experiment_search_and_comparison()
        
        # Test Genesis-Flow specific features
        test_genesis_flow_features()
        
        print("\n🎉 Genesis-Flow MongoDB Integration Test Complete!")
        print("=" * 50)
        print(f"📊 Experiment: {experiment_name}")
        print(f"🔗 Tracking URI: {tracking_uri}")
        print(f"📁 Artifacts: {artifact_root}")
        print(f"🏃 Runs completed: {len(results)}")
        
        if runs_df is not None and len(runs_df) > 0:
            best_run = runs_df.iloc[0]
            best_accuracy = best_run.get('metrics.accuracy', 'N/A')
            best_model = best_run.get('tags.model_type', 'Unknown')
            print(f"🏆 Best model: {best_model} (Accuracy: {best_accuracy})")
    
    else:
        print("❌ Test failed - Could not create experiment")


if __name__ == "__main__":
    main()