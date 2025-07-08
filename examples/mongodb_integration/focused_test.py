#!/usr/bin/env python
"""
Focused Genesis-Flow MongoDB Test
"""

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    """Test just tracking functionality first."""
    print("🎯 Focused Genesis-Flow MongoDB Test - Tracking Only")
    print("=" * 60)
    
    # Configure MongoDB URI (tracking only)
    tracking_uri = "mongodb://localhost:27017/genesis_flow_focused_test"
    
    print(f"🔗 Setting tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✅ Tracking URI set: {mlflow.get_tracking_uri()}")
    
    # Test 1: Create experiment
    print("\n🧪 Test 1: Creating experiment...")
    try:
        import time
        experiment_name = f"focused_test_experiment_{int(time.time())}"
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Experiment created: {experiment_id}")
    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")
        return False
    
    # Test 2: Start run and log data
    print("\n📊 Test 2: Starting run and logging data...")
    try:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_param("algorithm", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            
            # Log metrics
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("precision", 0.92)
            
            print(f"✅ Run completed: {run.info.run_id}")
    except Exception as e:
        print(f"❌ Failed to log data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Train and log model (without registry)
    print("\n🤖 Test 3: Training and logging model...")
    try:
        # Generate data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(experiment_id=experiment_id):
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Log model (without registry)
            mlflow.sklearn.log_model(model, "model")
            
            # Test prediction
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", accuracy)
            
            print(f"✅ Model trained and logged. Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ Failed to train/log model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tracking tests PASSED!")
    print("✅ MongoDB tracking integration working correctly")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)