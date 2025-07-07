#!/usr/bin/env python
"""
Comprehensive Model Registry Example with MongoDB Backend

This example demonstrates the complete MLflow Model Registry workflow with MongoDB storage:
- Model registration and versioning
- Stage transitions (Staging, Production, Archived)
- Model aliases and annotations
- Model metadata management
- Model deployment lifecycle
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

# For model examples
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def setup_mongodb_registry():
    """Configure Genesis-Flow to use MongoDB for both tracking and registry."""
    tracking_uri = "mongodb://localhost:27017/genesis_flow_registry"
    registry_uri = "mongodb://localhost:27017/genesis_flow_registry"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    # Initialize MLflow client for registry operations
    client = MlflowClient()
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    print(f"✅ MLflow Client initialized")
    
    return client


def create_sample_models():
    """Create and log multiple models for registry demonstration."""
    print("\n🤖 Creating Sample Models for Registry")
    print("=" * 45)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model configurations
    models_config = [
        {
            "name": "RandomForest_v1",
            "model_class": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "description": "Baseline Random Forest model"
        },
        {
            "name": "RandomForest_v2",
            "model_class": RandomForestClassifier,
            "params": {"n_estimators": 200, "max_depth": 15, "random_state": 42},
            "description": "Improved Random Forest with more trees"
        },
        {
            "name": "GradientBoosting_v1",
            "model_class": GradientBoostingClassifier,
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "description": "Gradient Boosting baseline model"
        },
        {
            "name": "LogisticRegression_v1",
            "model_class": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000},
            "description": "Simple logistic regression baseline"
        }
    ]
    
    # Set experiment
    experiment_name = "model_registry_training"
    mlflow.set_experiment(experiment_name)
    
    model_runs = []
    
    for config in models_config:
        with mlflow.start_run(run_name=config["name"]) as run:
            print(f"🔄 Training {config['name']}...")
            
            # Train model
            model = config["model_class"](**config["params"])
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(config["params"])
            mlflow.log_params({
                "model_type": config["model_class"].__name__,
                "description": config["description"]
            })
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=mlflow.models.infer_signature(X_train, y_pred),
                input_example=X_train[:3]
            )
            
            print(f"  ✅ {config['name']}: accuracy = {accuracy:.4f}, f1 = {f1:.4f}")
            
            model_runs.append({
                "run_id": run.info.run_id,
                "name": config["name"],
                "accuracy": accuracy,
                "f1_score": f1,
                "model_type": config["model_class"].__name__
            })
    
    print(f"\n✅ Created {len(model_runs)} models for registry demonstration")
    return model_runs, X_test, y_test


def demonstrate_model_registration(client, model_runs):
    """Example 1: Register models in the model registry."""
    print("\n📝 Example 1: Model Registration")
    print("=" * 35)
    
    registered_models = []
    
    # Register different model types
    for model_run in model_runs:
        model_name = f"BinaryClassifier_{model_run['model_type']}"
        model_uri = f"runs:/{model_run['run_id']}/model"
        
        try:
            # Register the model
            model_version = client.create_registered_model(
                name=model_name,
                description=f"Binary classification model using {model_run['model_type']}. "
                           f"Trained with accuracy: {model_run['accuracy']:.4f}"
            )
            print(f"✅ Registered model: {model_name}")
            
            # Create first version
            version = client.create_model_version(
                name=model_name,
                source=model_uri,
                description=f"Version 1 - {model_run['name']} (accuracy: {model_run['accuracy']:.4f})"
            )
            
            print(f"  📦 Created version {version.version} for {model_name}")
            
            registered_models.append({
                "name": model_name,
                "version": version.version,
                "run_id": model_run["run_id"],
                "accuracy": model_run["accuracy"]
            })
            
        except MlflowException as e:
            if "already exists" in str(e):
                print(f"⚠️  Model {model_name} already exists, creating new version...")
                # Create new version for existing model
                version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    description=f"New version - {model_run['name']} (accuracy: {model_run['accuracy']:.4f})"
                )
                print(f"  📦 Created version {version.version} for existing model {model_name}")
                
                registered_models.append({
                    "name": model_name,
                    "version": version.version,
                    "run_id": model_run["run_id"],
                    "accuracy": model_run["accuracy"]
                })
            else:
                print(f"❌ Error registering {model_name}: {e}")
    
    return registered_models


def demonstrate_stage_transitions(client, registered_models):
    """Example 2: Model stage transitions and lifecycle management."""
    print("\n🔄 Example 2: Model Stage Transitions")
    print("=" * 40)
    
    # Sort models by accuracy to simulate promotion workflow
    sorted_models = sorted(registered_models, key=lambda x: x["accuracy"], reverse=True)
    
    if len(sorted_models) < 2:
        print("❌ Need at least 2 models for stage transition demo")
        return
    
    best_model = sorted_models[0]
    second_model = sorted_models[1]
    
    print(f"🏆 Best model: {best_model['name']} v{best_model['version']} (accuracy: {best_model['accuracy']:.4f})")
    print(f"🥈 Second model: {second_model['name']} v{second_model['version']} (accuracy: {second_model['accuracy']:.4f})")
    
    # Transition best model to Staging
    print(f"\n🔄 Promoting {best_model['name']} to Staging...")
    client.transition_model_version_stage(
        name=best_model["name"],
        version=best_model["version"],
        stage="Staging",
        description="Promoting best performing model to staging for validation"
    )
    print(f"✅ {best_model['name']} v{best_model['version']} -> Staging")
    
    # Add some validation metadata
    client.set_model_version_tag(
        name=best_model["name"],
        version=best_model["version"],
        key="validation_status",
        value="pending"
    )
    
    client.set_model_version_tag(
        name=best_model["name"],
        version=best_model["version"],
        key="promoted_by",
        value="ml_engineer"
    )
    
    client.set_model_version_tag(
        name=best_model["name"],
        version=best_model["version"],
        key="promotion_date",
        value=datetime.now().isoformat()
    )
    
    # Simulate validation period
    print("⏳ Simulating validation period...")
    time.sleep(1)
    
    # After validation, promote to Production
    print(f"🔄 Promoting {best_model['name']} to Production...")
    client.transition_model_version_stage(
        name=best_model["name"],
        version=best_model["version"],
        stage="Production",
        description="Model passed validation, promoting to production"
    )
    print(f"✅ {best_model['name']} v{best_model['version']} -> Production")
    
    # Update validation status
    client.set_model_version_tag(
        name=best_model["name"],
        version=best_model["version"],
        key="validation_status",
        value="passed"
    )
    
    client.set_model_version_tag(
        name=best_model["name"],
        version=best_model["version"],
        key="production_date",
        value=datetime.now().isoformat()
    )
    
    # Move second model to staging
    print(f"\n🔄 Promoting {second_model['name']} to Staging...")
    client.transition_model_version_stage(
        name=second_model["name"],
        version=second_model["version"],
        stage="Staging",
        description="Alternative model for A/B testing"
    )
    print(f"✅ {second_model['name']} v{second_model['version']} -> Staging")
    
    return best_model, second_model


def demonstrate_model_aliases(client, registered_models):
    """Example 3: Model aliases for flexible model management."""
    print("\n🏷️  Example 3: Model Aliases")
    print("=" * 30)
    
    if not registered_models:
        print("❌ No registered models available for alias demo")
        return
    
    # Get the best performing model
    best_model = max(registered_models, key=lambda x: x["accuracy"])
    
    try:
        # Set aliases for the best model
        aliases_to_set = [
            ("champion", "Current best performing model in production"),
            ("latest", "Most recently validated model"),
            ("stable", "Stable model for consistent predictions")
        ]
        
        for alias, description in aliases_to_set:
            try:
                client.set_registered_model_alias(
                    name=best_model["name"],
                    alias=alias,
                    version=best_model["version"]
                )
                print(f"✅ Set alias '{alias}' -> {best_model['name']} v{best_model['version']}")
                
                # Add description as tag
                client.set_model_version_tag(
                    name=best_model["name"],
                    version=best_model["version"],
                    key=f"alias_{alias}_description",
                    value=description
                )
            except Exception as e:
                print(f"⚠️  Could not set alias '{alias}': {e}")
        
        # Demonstrate alias usage
        print(f"\n📊 Model Aliases for {best_model['name']}:")
        try:
            # List all aliases (this might not be available in all MLflow versions)
            model_details = client.get_registered_model(best_model["name"])
            print(f"  Model: {model_details.name}")
            print(f"  Description: {model_details.description}")
            
            # Get model version details
            version_details = client.get_model_version(
                name=best_model["name"],
                version=best_model["version"]
            )
            print(f"  Version: {version_details.version}")
            print(f"  Stage: {version_details.current_stage}")
            
            # Show tags
            if version_details.tags:
                print(f"  Tags:")
                for key, value in version_details.tags.items():
                    print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"⚠️  Could not retrieve alias details: {e}")
            
    except Exception as e:
        print(f"❌ Error with aliases: {e}")
        print("Note: Model aliases might not be fully supported in this MLflow version")


def demonstrate_model_search_and_discovery(client):
    """Example 4: Search and discover models in the registry."""
    print("\n🔍 Example 4: Model Search and Discovery")
    print("=" * 45)
    
    # List all registered models
    print("📋 All Registered Models:")
    try:
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            print(f"\n📦 Model: {model.name}")
            print(f"   Description: {model.description}")
            print(f"   Creation Time: {model.creation_timestamp}")
            print(f"   Last Updated: {model.last_updated_timestamp}")
            
            # Get all versions for this model
            versions = client.search_model_versions(f"name='{model.name}'")
            print(f"   Versions: {len(versions)}")
            
            for version in versions:
                print(f"     v{version.version}: {version.current_stage} "
                      f"(run: {version.run_id[:8]}...)")
                if version.description:
                    print(f"       Description: {version.description}")
    
    except Exception as e:
        print(f"❌ Error searching models: {e}")
    
    # Search models by stage
    print(f"\n🏭 Production Models:")
    try:
        production_models = client.search_model_versions("current_stage='Production'")
        for model in production_models:
            print(f"  📦 {model.name} v{model.version}")
            print(f"     Run ID: {model.run_id}")
            print(f"     Source: {model.source}")
    except Exception as e:
        print(f"❌ Error searching production models: {e}")
    
    # Search models by name pattern
    print(f"\n🤖 RandomForest Models:")
    try:
        rf_models = client.search_model_versions("name LIKE 'BinaryClassifier_RandomForest%'")
        for model in rf_models:
            print(f"  🌲 {model.name} v{model.version} ({model.current_stage})")
    except Exception as e:
        print(f"❌ Error searching RandomForest models: {e}")


def demonstrate_model_metadata_management(client, registered_models):
    """Example 5: Comprehensive metadata management."""
    print("\n📊 Example 5: Model Metadata Management")
    print("=" * 45)
    
    if not registered_models:
        print("❌ No registered models available for metadata demo")
        return
    
    model = registered_models[0]  # Use first model
    
    print(f"🏷️  Adding metadata to {model['name']} v{model['version']}")
    
    # Add comprehensive tags
    metadata_tags = {
        "model_family": "tree_based",
        "data_version": "v2.1",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "data_scientist": "john_doe",
        "business_unit": "recommendation_engine",
        "cost_per_prediction": "0.001",
        "latency_requirement": "< 100ms",
        "accuracy_threshold": "0.85",
        "model_size_mb": "2.5",
        "inference_framework": "sklearn",
        "deployment_target": "kubernetes",
        "monitoring_enabled": "true",
        "feature_drift_detection": "enabled",
        "model_explanation": "available",
        "compliance_status": "gdpr_compliant"
    }
    
    for key, value in metadata_tags.items():
        try:
            client.set_model_version_tag(
                name=model["name"],
                version=model["version"],
                key=key,
                value=value
            )
            print(f"  ✅ {key}: {value}")
        except Exception as e:
            print(f"  ❌ Failed to set {key}: {e}")
    
    # Add model annotations
    try:
        client.update_model_version(
            name=model["name"],
            version=model["version"],
            description=f"Enhanced model with comprehensive metadata. "
                       f"Original accuracy: {model['accuracy']:.4f}. "
                       f"Suitable for production deployment with monitoring."
        )
        print(f"✅ Updated model description")
    except Exception as e:
        print(f"❌ Failed to update description: {e}")
    
    # Retrieve and display all metadata
    print(f"\n📋 Complete Metadata for {model['name']} v{model['version']}:")
    try:
        version_details = client.get_model_version(
            name=model["name"],
            version=model["version"]
        )
        
        print(f"  Name: {version_details.name}")
        print(f"  Version: {version_details.version}")
        print(f"  Stage: {version_details.current_stage}")
        print(f"  Description: {version_details.description}")
        print(f"  Run ID: {version_details.run_id}")
        print(f"  Source: {version_details.source}")
        
        if version_details.tags:
            print(f"  Tags ({len(version_details.tags)}):")
            for key, value in sorted(version_details.tags.items()):
                print(f"    {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error retrieving metadata: {e}")


def demonstrate_model_deployment_workflow(client, registered_models):
    """Example 6: Complete model deployment workflow."""
    print("\n🚀 Example 6: Model Deployment Workflow")
    print("=" * 45)
    
    if not registered_models:
        print("❌ No registered models available for deployment demo")
        return
    
    # Select best model for deployment
    best_model = max(registered_models, key=lambda x: x["accuracy"])
    
    print(f"🎯 Deploying {best_model['name']} v{best_model['version']}")
    
    # Step 1: Pre-deployment validation
    print(f"\n1️⃣  Pre-deployment Validation")
    validation_tags = {
        "pre_deploy_validation": "passed",
        "security_scan": "clean",
        "performance_test": "passed",
        "integration_test": "passed",
        "load_test": "passed"
    }
    
    for key, value in validation_tags.items():
        client.set_model_version_tag(
            name=best_model["name"],
            version=best_model["version"],
            key=key,
            value=value
        )
        print(f"  ✅ {key}: {value}")
    
    # Step 2: Deployment configuration
    print(f"\n2️⃣  Deployment Configuration")
    deployment_config = {
        "deployment_strategy": "blue_green",
        "target_environment": "production",
        "replicas": "3",
        "cpu_request": "500m",
        "memory_request": "1Gi",
        "cpu_limit": "1000m",
        "memory_limit": "2Gi",
        "autoscaling_enabled": "true",
        "min_replicas": "2",
        "max_replicas": "10"
    }
    
    for key, value in deployment_config.items():
        client.set_model_version_tag(
            name=best_model["name"],
            version=best_model["version"],
            key=f"deploy_{key}",
            value=value
        )
        print(f"  🔧 {key}: {value}")
    
    # Step 3: Deployment execution
    print(f"\n3️⃣  Deployment Execution")
    
    # Transition to Production if not already
    try:
        client.transition_model_version_stage(
            name=best_model["name"],
            version=best_model["version"],
            stage="Production",
            description="Automated deployment to production environment"
        )
        print(f"  ✅ Promoted to Production stage")
    except Exception as e:
        print(f"  ℹ️  Already in Production or transition failed: {e}")
    
    # Set deployment timestamp and status
    deployment_metadata = {
        "deployment_timestamp": datetime.now().isoformat(),
        "deployment_status": "active",
        "deployment_version": "1.0.0",
        "deployment_endpoint": "https://api.company.com/models/binary-classifier/v1",
        "health_check_url": "https://api.company.com/models/binary-classifier/v1/health",
        "monitoring_dashboard": "https://grafana.company.com/dashboard/model-monitoring"
    }
    
    for key, value in deployment_metadata.items():
        client.set_model_version_tag(
            name=best_model["name"],
            version=best_model["version"],
            key=key,
            value=value
        )
        print(f"  🚀 {key}: {value}")
    
    # Step 4: Post-deployment monitoring setup
    print(f"\n4️⃣  Post-deployment Monitoring")
    monitoring_config = {
        "monitoring_enabled": "true",
        "alerting_enabled": "true",
        "drift_detection": "enabled",
        "performance_tracking": "enabled",
        "error_rate_threshold": "1%",
        "latency_threshold": "100ms",
        "throughput_threshold": "1000_rps"
    }
    
    for key, value in monitoring_config.items():
        client.set_model_version_tag(
            name=best_model["name"],
            version=best_model["version"],
            key=f"monitor_{key}",
            value=value
        )
        print(f"  📊 {key}: {value}")
    
    print(f"\n✅ Deployment workflow completed for {best_model['name']} v{best_model['version']}")
    print(f"🌐 Model available at: {deployment_metadata['deployment_endpoint']}")


def main():
    """Run all model registry examples."""
    print("🚀 Genesis-Flow MongoDB Model Registry Examples")
    print("=" * 60)
    
    # Setup
    client = setup_mongodb_registry()
    
    # Create sample models
    model_runs, X_test, y_test = create_sample_models()
    
    # Run registry examples
    registered_models = demonstrate_model_registration(client, model_runs)
    
    if registered_models:
        best_model, second_model = demonstrate_stage_transitions(client, registered_models)
        demonstrate_model_aliases(client, registered_models)
        demonstrate_model_search_and_discovery(client)
        demonstrate_model_metadata_management(client, registered_models)
        demonstrate_model_deployment_workflow(client, registered_models)
    
    print(f"\n🎉 ALL MODEL REGISTRY EXAMPLES COMPLETED!")
    print("=" * 60)
    print(f"📋 Summary:")
    print(f"  ✅ Models registered: {len(registered_models)}")
    print(f"  ✅ Stage transitions: Demonstrated")
    print(f"  ✅ Model aliases: Configured")
    print(f"  ✅ Metadata management: Complete")
    print(f"  ✅ Deployment workflow: Simulated")
    print(f"\n📊 Database: genesis_flow_registry")
    print(f"🔗 All registry data stored in MongoDB!")


if __name__ == "__main__":
    main()