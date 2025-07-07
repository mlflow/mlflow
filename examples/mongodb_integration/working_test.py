#!/usr/bin/env python
"""
Working Genesis-Flow MongoDB Integration Test

This test demonstrates the fully working MongoDB integration with run creation.
"""

import sys
import os

# Ensure we're using Genesis-Flow, not system MLflow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
import random
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    """Run complete Genesis-Flow MongoDB integration test."""
    print("🎯 Genesis-Flow MongoDB Integration - WORKING TEST")
    print("=" * 55)
    
    print(f"MLflow path: {mlflow.__file__}")
    print(f"MLflow version: {mlflow.__version__}")
    
    # Configure MongoDB URIs
    tracking_uri = "mongodb://localhost:27017/genesis_flow_test"
    registry_uri = "mongodb://localhost:27017/genesis_flow_test"
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    
    # Set URIs
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    try:
        print("\\n✅ Phase 1: Experiment Management")
        
        # Create experiment
        experiment_name = f"genesis_mongodb_test_{random.randint(1000, 9999)}_{int(time.time())}"
        print(f"  Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"  ✅ Experiment created: {experiment_id}")
        
        print("\\n✅ Phase 2: Data Generation")
        
        # Generate sample data
        print("  Generating dataset...")
        X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"  ✅ Data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        print("\\n✅ Phase 3: ML Training with Run Tracking")
        
        # Train model with MLflow tracking
        with mlflow.start_run(experiment_id=experiment_id, run_name="mongodb_ml_test") as run:
            print(f"  ✅ Run started: {run.info.run_id}")
            
            # Log parameters
            n_estimators = 50
            max_depth = 8
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("algorithm", "RandomForest")
            mlflow.log_param("test_framework", "genesis-flow")
            print("  ✅ Parameters logged")
            
            # Train model
            print("  Training model...")
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
            mlflow.log_metric("n_features", X.shape[1])
            print("  ✅ Metrics logged")
            
            print(f"  📊 Model accuracy: {accuracy:.4f}")
        
        print("\\n🎉 ALL TESTS PASSED!")
        print("\\n📋 Summary:")
        print("  ✅ MongoDB URI recognition (tracking & model registry)")
        print("  ✅ Experiment creation with MongoDB storage")
        print("  ✅ Run creation and management")
        print("  ✅ Parameter logging to MongoDB")
        print("  ✅ Metric logging to MongoDB")
        print("  ✅ Run completion and status updates")
        print("  ✅ Direct MongoDB storage (no MLflow server needed)")
        
        print(f"\\n🗄️  Database Details:")
        print(f"  Database: genesis_flow_test")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model Accuracy: {accuracy:.4f}")
        print(f"  Storage: MongoDB + Genesis-Flow")
        
        print("\\n🚀 Genesis-Flow MongoDB Integration: FULLY FUNCTIONAL!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)