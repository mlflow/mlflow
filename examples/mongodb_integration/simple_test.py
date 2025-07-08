#!/usr/bin/env python
"""
Simple Genesis-Flow MongoDB Integration Test

This script demonstrates Genesis-Flow working with MongoDB for tracking and model registry.
"""

import os
import time
import random
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    """Run simple MongoDB integration test."""
    print("🎯 Genesis-Flow MongoDB Integration Test")
    print("=" * 50)
    
    # Configure MongoDB URIs
    tracking_uri = "mongodb://localhost:27017/genesis_flow_test"
    registry_uri = "mongodb://localhost:27017/genesis_flow_test"
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    
    # Set URIs
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    # Create test experiment
    experiment_name = f"genesis_flow_test_{random.randint(1000, 9999)}_{int(time.time())}"
    
    try:
        print(f"\\n🧪 Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Experiment created: {experiment_id}")
        
        # Generate sample data
        print("\\n📊 Generating sample data...")
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Train model with MLflow tracking
        print("\\n🤖 Training model...")
        with mlflow.start_run(experiment_id=experiment_id, run_name="mongodb_test_run") as run:
            # Log parameters
            n_estimators = 100
            max_depth = 10
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("algorithm", "RandomForest")
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions and log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Log model (this will test model registry)
            print("\\n💾 Logging model...")
            model_info = mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                registered_model_name=f"genesis_flow_model_{random.randint(100, 999)}"
            )
            
            print(f"✅ Run completed: {run.info.run_id}")
            print(f"✅ Model logged: {model_info.model_uri}")
            print(f"✅ Accuracy: {accuracy:.4f}")
        
        print("\\n🎉 MongoDB Integration Test SUCCESSFUL!")
        print("\\nKey achievements:")
        print("  ✅ Connected to MongoDB successfully")
        print("  ✅ Created experiment in MongoDB")
        print("  ✅ Logged parameters and metrics to MongoDB")
        print("  ✅ Registered model in MongoDB model registry")
        print("  ✅ Genesis-Flow working end-to-end with MongoDB!")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)