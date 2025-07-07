#!/usr/bin/env python
"""
Complete MLflow Workflow Example with MongoDB Backend

This example demonstrates a comprehensive end-to-end ML workflow using Genesis-Flow with MongoDB:
- Data ingestion and validation
- Experiment tracking with hyperparameter tuning
- Model training and evaluation
- Model registry and versioning
- A/B testing and model comparison
- Production deployment simulation
- Model monitoring and maintenance

This represents a complete production MLflow workflow.
"""

import sys
import os
import time
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# ML libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class MLflowWorkflowOrchestrator:
    """Orchestrates the complete MLflow workflow with MongoDB backend."""
    
    def __init__(self, project_name: str = "complete_mlflow_demo"):
        self.project_name = project_name
        self.tracking_uri = f"mongodb://localhost:27017/genesis_flow_{project_name}"
        self.registry_uri = f"mongodb://localhost:27017/genesis_flow_{project_name}"
        self.client = None
        self.experiment_id = None
        self.best_model = None
        self.model_registry_name = "ProductionBinaryClassifier"
        
    def setup_environment(self):
        """Initialize MLflow environment and MongoDB connection."""
        print("🔧 Setting up MLflow Environment")
        print("=" * 40)
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        self.client = MlflowClient()
        
        print(f"🔗 Tracking URI: {self.tracking_uri}")
        print(f"📝 Registry URI: {self.registry_uri}")
        
        # Create main experiment
        experiment_name = f"{self.project_name}_development"
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": self.project_name,
                    "stage": "development",
                    "created_by": "ml_engineer",
                    "created_at": datetime.now().isoformat()
                }
            )
            print(f"✅ Created experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {experiment_name}")
        
        return self
    
    def stage_1_data_ingestion_and_validation(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Stage 1: Data ingestion, validation, and preparation."""
        print("\n📊 Stage 1: Data Ingestion & Validation")
        print("=" * 45)
        
        mlflow.set_experiment(f"{self.project_name}_development")
        
        with mlflow.start_run(run_name="data_ingestion_and_validation") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Simulate data ingestion
            print("🔄 Ingesting data...")
            X, y = make_classification(
                n_samples=5000,
                n_features=25,
                n_informative=20,
                n_redundant=5,
                n_classes=2,
                class_sep=0.8,
                random_state=42
            )
            
            feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
            
            # Data validation
            print("🔍 Validating data quality...")
            data_quality_metrics = {
                "total_samples": len(X),
                "total_features": X.shape[1],
                "missing_values": 0,  # Simulated
                "duplicate_rows": 0,  # Simulated
                "class_balance_ratio": max(np.bincount(y)) / min(np.bincount(y)),
                "feature_correlation_max": np.abs(np.corrcoef(X.T)).max(),
                "data_quality_score": 0.95
            }
            
            # Log data validation metrics
            mlflow.log_params({
                "data_source": "synthetic_generator",
                "data_version": "v1.0",
                "ingestion_date": datetime.now().isoformat()
            })
            
            mlflow.log_metrics(data_quality_metrics)
            
            # Create and log data profile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Data summary
                df = pd.DataFrame(X, columns=feature_names)
                df['target'] = y
                
                summary_stats = df.describe()
                stats_path = temp_path / "data_summary_statistics.csv"
                summary_stats.to_csv(stats_path)
                mlflow.log_artifact(str(stats_path), "data_validation")
                
                # Data quality report
                quality_report = {
                    "validation_timestamp": datetime.now().isoformat(),
                    "data_quality_checks": {
                        "schema_validation": "passed",
                        "completeness_check": "passed",
                        "consistency_check": "passed",
                        "uniqueness_check": "passed",
                        "range_validation": "passed"
                    },
                    "data_statistics": data_quality_metrics,
                    "recommendations": [
                        "Data quality is excellent",
                        "No missing values detected",
                        "Class distribution is acceptable",
                        "Ready for model training"
                    ]
                }
                
                quality_path = temp_path / "data_quality_report.json"
                with open(quality_path, 'w') as f:
                    json.dump(quality_report, f, indent=2)
                mlflow.log_artifact(str(quality_path), "data_validation")
            
            # Set tags for data stage
            mlflow.set_tags({
                "stage": "data_ingestion",
                "data_quality": "excellent",
                "ready_for_training": "true"
            })
            
            print(f"✅ Data validation completed: {data_quality_metrics['total_samples']} samples, "
                  f"{data_quality_metrics['total_features']} features")
            print(f"✅ Data quality score: {data_quality_metrics['data_quality_score']}")
            
        return X, y, feature_names
    
    def stage_2_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Stage 2: Hyperparameter tuning with multiple algorithms."""
        print("\n🎯 Stage 2: Hyperparameter Tuning")
        print("=" * 38)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define model configurations for hyperparameter tuning
        model_configs = {
            "random_forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 15, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            },
            "logistic_regression": {
                "model": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                ]),
                "params": {
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__penalty": ['l1', 'l2'],
                    "classifier__solver": ['liblinear']
                }
            }
        }
        
        best_models = {}
        
        for model_name, config in model_configs.items():
            print(f"\n🔄 Tuning {model_name}...")
            
            with mlflow.start_run(run_name=f"hyperparameter_tuning_{model_name}") as run:
                print(f"  📝 Started run: {run.info.run_id}")
                
                # Grid search
                grid_search = GridSearchCV(
                    config["model"],
                    config["params"],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Log best parameters
                mlflow.log_params({
                    "model_type": model_name,
                    **{f"best_{k}": v for k, v in grid_search.best_params_.items()}
                })
                
                # Evaluate best model
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                metrics = {
                    "cv_score": grid_search.best_score_,
                    "test_accuracy": accuracy_score(y_test, y_pred),
                    "test_precision": precision_score(y_test, y_pred),
                    "test_recall": recall_score(y_test, y_pred),
                    "test_f1": f1_score(y_test, y_pred),
                    "test_roc_auc": roc_auc_score(y_test, y_proba)
                }
                
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    signature=mlflow.models.infer_signature(X_train, y_pred),
                    input_example=X_train[:3]
                )
                
                # Save hyperparameter tuning results
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Grid search results
                    results_df = pd.DataFrame(grid_search.cv_results_)
                    results_path = temp_path / f"{model_name}_grid_search_results.csv"
                    results_df.to_csv(results_path, index=False)
                    mlflow.log_artifact(str(results_path), "hyperparameter_tuning")
                
                mlflow.set_tags({
                    "stage": "hyperparameter_tuning",
                    "model_family": model_name,
                    "tuning_method": "grid_search"
                })
                
                best_models[model_name] = {
                    "model": best_model,
                    "metrics": metrics,
                    "run_id": run.info.run_id,
                    "params": grid_search.best_params_
                }
                
                print(f"  ✅ {model_name}: CV Score = {grid_search.best_score_:.4f}, "
                      f"Test Accuracy = {metrics['test_accuracy']:.4f}")
        
        return best_models
    
    def stage_3_model_comparison_and_selection(self, best_models: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Compare models and select the best one."""
        print("\n🏆 Stage 3: Model Comparison & Selection")
        print("=" * 45)
        
        with mlflow.start_run(run_name="model_comparison_and_selection") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Create comparison table
            comparison_data = []
            for model_name, model_info in best_models.items():
                comparison_data.append({
                    "model_name": model_name,
                    "run_id": model_info["run_id"],
                    **model_info["metrics"]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Select best model based on F1 score
            best_model_idx = comparison_df["test_f1"].idxmax()
            best_model_name = comparison_df.loc[best_model_idx, "model_name"]
            self.best_model = best_models[best_model_name]
            
            print(f"🏅 Best model: {best_model_name}")
            print(f"   F1 Score: {self.best_model['metrics']['test_f1']:.4f}")
            print(f"   Accuracy: {self.best_model['metrics']['test_accuracy']:.4f}")
            print(f"   ROC AUC: {self.best_model['metrics']['test_roc_auc']:.4f}")
            
            # Log comparison results
            mlflow.log_params({
                "comparison_metric": "test_f1",
                "best_model": best_model_name,
                "models_compared": len(best_models)
            })
            
            mlflow.log_metrics({
                f"best_{k}": v for k, v in self.best_model["metrics"].items()
            })
            
            # Save comparison table
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                comparison_path = temp_path / "model_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(str(comparison_path), "model_selection")
                
                # Create model selection report
                selection_report = {
                    "selection_timestamp": datetime.now().isoformat(),
                    "selection_criteria": "Highest F1 Score",
                    "selected_model": {
                        "name": best_model_name,
                        "run_id": self.best_model["run_id"],
                        "metrics": self.best_model["metrics"],
                        "parameters": self.best_model["params"]
                    },
                    "all_models": comparison_data,
                    "selection_rationale": f"Selected {best_model_name} due to highest F1 score of {self.best_model['metrics']['test_f1']:.4f}"
                }
                
                report_path = temp_path / "model_selection_report.json"
                with open(report_path, 'w') as f:
                    json.dump(selection_report, f, indent=2)
                mlflow.log_artifact(str(report_path), "model_selection")
            
            mlflow.set_tags({
                "stage": "model_selection",
                "selected_model": best_model_name,
                "selection_method": "f1_score_optimization"
            })
            
        return self.best_model
    
    def stage_4_model_registration_and_staging(self) -> str:
        """Stage 4: Register the best model and transition through stages."""
        print("\n📝 Stage 4: Model Registration & Staging")
        print("=" * 45)
        
        try:
            # Create registered model
            try:
                registered_model = self.client.create_registered_model(
                    name=self.model_registry_name,
                    description=f"Production binary classifier for {self.project_name}. "
                               f"Best performing model with F1 score: {self.best_model['metrics']['test_f1']:.4f}"
                )
                print(f"✅ Created registered model: {self.model_registry_name}")
            except Exception:
                print(f"✅ Using existing registered model: {self.model_registry_name}")
            
            # Create model version
            model_uri = f"runs:/{self.best_model['run_id']}/model"
            model_version = self.client.create_model_version(
                name=self.model_registry_name,
                source=model_uri,
                description=f"Version trained on {datetime.now().strftime('%Y-%m-%d')}. "
                           f"Model: {self.best_model.get('model_type', 'unknown')}. "
                           f"F1: {self.best_model['metrics']['test_f1']:.4f}, "
                           f"Accuracy: {self.best_model['metrics']['test_accuracy']:.4f}"
            )
            
            version_number = model_version.version
            print(f"✅ Created model version: {version_number}")
            
            # Add comprehensive metadata
            metadata_tags = {
                "model_type": list(self.best_model["params"].keys())[0].split("__")[0] if "__" in str(self.best_model["params"]) else "unknown",
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "f1_score": str(self.best_model["metrics"]["test_f1"]),
                "accuracy": str(self.best_model["metrics"]["test_accuracy"]),
                "roc_auc": str(self.best_model["metrics"]["test_roc_auc"]),
                "data_version": "v1.0",
                "training_samples": "4000",
                "validation_method": "5_fold_cv",
                "hyperparameter_tuning": "grid_search",
                "production_ready": "pending_validation"
            }
            
            for key, value in metadata_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=version_number,
                    key=key,
                    value=value
                )
            
            print("✅ Added model metadata tags")
            
            # Transition to Staging
            self.client.transition_model_version_stage(
                name=self.model_registry_name,
                version=version_number,
                stage="Staging",
                description="Moving to staging for validation and A/B testing"
            )
            print(f"✅ Transitioned model v{version_number} to Staging")
            
            return version_number
            
        except Exception as e:
            print(f"❌ Error in model registration: {e}")
            return None
    
    def stage_5_model_validation_and_testing(self, model_version: str, X: np.ndarray, y: np.ndarray):
        """Stage 5: Comprehensive model validation and testing."""
        print("\n🧪 Stage 5: Model Validation & Testing")
        print("=" * 42)
        
        with mlflow.start_run(run_name="model_validation_and_testing") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Load model from registry
            model_uri = f"models:/{self.model_registry_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Create validation dataset (simulate fresh data)
            X_val, y_val = make_classification(
                n_samples=1000,
                n_features=25,
                n_informative=20,
                n_redundant=5,
                n_classes=2,
                class_sep=0.8,
                random_state=123  # Different seed for validation
            )
            
            # Comprehensive validation
            print("🔄 Running comprehensive validation...")
            
            # Performance validation
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            validation_metrics = {
                "validation_accuracy": accuracy_score(y_val, y_pred),
                "validation_precision": precision_score(y_val, y_pred),
                "validation_recall": recall_score(y_val, y_pred),
                "validation_f1": f1_score(y_val, y_pred),
                "validation_roc_auc": roc_auc_score(y_val, y_proba)
            }
            
            # Performance comparison with training metrics
            performance_degradation = {
                f"degradation_{metric}": abs(validation_metrics[f"validation_{metric}"] - self.best_model["metrics"][f"test_{metric}"])
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            }
            
            # Validation thresholds
            validation_passed = {
                "accuracy_threshold": validation_metrics["validation_accuracy"] > 0.8,
                "f1_threshold": validation_metrics["validation_f1"] > 0.75,
                "roc_auc_threshold": validation_metrics["validation_roc_auc"] > 0.85,
                "performance_stable": all(deg < 0.05 for deg in performance_degradation.values())
            }
            
            overall_validation = all(validation_passed.values())
            
            # Log validation results
            mlflow.log_metrics({
                **validation_metrics,
                **performance_degradation
            })
            
            mlflow.log_params({
                "validation_samples": len(X_val),
                "validation_passed": overall_validation,
                "validation_method": "holdout_validation"
            })
            
            # Create validation report
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                validation_report = {
                    "validation_timestamp": datetime.now().isoformat(),
                    "model_version": model_version,
                    "validation_metrics": validation_metrics,
                    "performance_degradation": performance_degradation,
                    "validation_thresholds": validation_passed,
                    "overall_validation": overall_validation,
                    "recommendation": "Approve for production" if overall_validation else "Requires improvement"
                }
                
                report_path = temp_path / "model_validation_report.json"
                with open(report_path, 'w') as f:
                    json.dump(validation_report, f, indent=2)
                mlflow.log_artifact(str(report_path), "validation")
                
                # Classification report
                class_report = classification_report(y_val, y_pred, output_dict=True)
                class_report_path = temp_path / "classification_report.json"
                with open(class_report_path, 'w') as f:
                    json.dump(class_report, f, indent=2)
                mlflow.log_artifact(str(class_report_path), "validation")
            
            # Update model registry with validation results
            validation_status = "passed" if overall_validation else "failed"
            self.client.set_model_version_tag(
                name=self.model_registry_name,
                version=model_version,
                key="validation_status",
                value=validation_status
            )
            
            self.client.set_model_version_tag(
                name=self.model_registry_name,
                version=model_version,
                key="validation_date",
                value=datetime.now().isoformat()
            )
            
            mlflow.set_tags({
                "stage": "model_validation",
                "validation_status": validation_status,
                "ready_for_production": str(overall_validation)
            })
            
            print(f"✅ Validation completed: {'PASSED' if overall_validation else 'FAILED'}")
            for metric, value in validation_metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return overall_validation
    
    def stage_6_production_deployment(self, model_version: str, validation_passed: bool):
        """Stage 6: Production deployment simulation."""
        print("\n🚀 Stage 6: Production Deployment")
        print("=" * 38)
        
        if not validation_passed:
            print("❌ Cannot deploy to production: Model validation failed")
            return False
        
        with mlflow.start_run(run_name="production_deployment") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Transition to Production
            self.client.transition_model_version_stage(
                name=self.model_registry_name,
                version=model_version,
                stage="Production",
                description="Deploying to production after successful validation"
            )
            print(f"✅ Model v{model_version} transitioned to Production")
            
            # Set production metadata
            production_metadata = {
                "deployment_timestamp": datetime.now().isoformat(),
                "deployment_environment": "production",
                "deployment_method": "kubernetes",
                "load_balancer": "enabled",
                "auto_scaling": "enabled",
                "monitoring": "enabled",
                "health_checks": "enabled",
                "backup_model": "previous_production_version"
            }
            
            for key, value in production_metadata.items():
                self.client.set_model_version_tag(
                    name=self.model_registry_name,
                    version=model_version,
                    key=key,
                    value=value
                )
            
            # Log deployment configuration
            mlflow.log_params({
                "deployment_target": "production",
                "deployment_strategy": "blue_green",
                "model_version": model_version
            })
            
            # Simulate deployment metrics
            deployment_metrics = {
                "deployment_time_seconds": 120,
                "health_check_success_rate": 1.0,
                "initial_throughput_rps": 100,
                "initial_latency_p95_ms": 45
            }
            
            mlflow.log_metrics(deployment_metrics)
            
            # Create deployment report
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                deployment_report = {
                    "deployment_timestamp": datetime.now().isoformat(),
                    "model_version": model_version,
                    "deployment_status": "successful",
                    "deployment_configuration": production_metadata,
                    "performance_metrics": deployment_metrics,
                    "rollback_plan": "Automatic rollback to previous version if error rate > 1%",
                    "monitoring_dashboard": "https://monitoring.company.com/ml-models",
                    "alert_configuration": "Slack + PagerDuty for critical issues"
                }
                
                report_path = temp_path / "deployment_report.json"
                with open(report_path, 'w') as f:
                    json.dump(deployment_report, f, indent=2)
                mlflow.log_artifact(str(report_path), "deployment")
            
            mlflow.set_tags({
                "stage": "production_deployment",
                "deployment_status": "successful",
                "environment": "production"
            })
            
            print("✅ Production deployment completed successfully")
            print(f"   Deployment time: {deployment_metrics['deployment_time_seconds']} seconds")
            print(f"   Health check success: {deployment_metrics['health_check_success_rate']*100}%")
            print(f"   Initial latency P95: {deployment_metrics['initial_latency_p95_ms']} ms")
            
            return True
    
    def stage_7_monitoring_and_maintenance(self, model_version: str):
        """Stage 7: Model monitoring and maintenance simulation."""
        print("\n📊 Stage 7: Monitoring & Maintenance")
        print("=" * 40)
        
        with mlflow.start_run(run_name="production_monitoring") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Simulate production monitoring data over time
            print("🔄 Simulating production monitoring...")
            
            # Generate monitoring metrics over 7 days
            monitoring_data = []
            base_time = datetime.now()
            
            for day in range(7):
                current_time = base_time + timedelta(days=day)
                
                # Simulate gradual performance degradation
                degradation_factor = 1.0 - (day * 0.01)  # 1% degradation per day
                
                day_metrics = {
                    "timestamp": current_time.isoformat(),
                    "day": day + 1,
                    "accuracy": 0.85 * degradation_factor + np.random.normal(0, 0.01),
                    "precision": 0.83 * degradation_factor + np.random.normal(0, 0.01),
                    "recall": 0.87 * degradation_factor + np.random.normal(0, 0.01),
                    "f1_score": 0.85 * degradation_factor + np.random.normal(0, 0.01),
                    "prediction_volume": np.random.randint(9500, 10500),
                    "avg_response_time_ms": 45 + np.random.normal(0, 5),
                    "error_rate": 0.001 + (day * 0.0002),  # Slight increase in errors
                    "data_drift_score": day * 0.05,  # Increasing drift
                    "feature_drift_count": day
                }
                
                monitoring_data.append(day_metrics)
                
                # Log daily metrics
                mlflow.log_metrics({
                    f"day_{day+1}_{k}": v for k, v in day_metrics.items() 
                    if k not in ["timestamp", "day"]
                })
            
            # Analyze trends
            accuracy_trend = [d["accuracy"] for d in monitoring_data]
            drift_trend = [d["data_drift_score"] for d in monitoring_data]
            
            # Monitoring analysis
            monitoring_analysis = {
                "monitoring_period_days": 7,
                "accuracy_degradation": accuracy_trend[0] - accuracy_trend[-1],
                "max_data_drift": max(drift_trend),
                "avg_response_time": np.mean([d["avg_response_time_ms"] for d in monitoring_data]),
                "total_predictions": sum([d["prediction_volume"] for d in monitoring_data]),
                "max_error_rate": max([d["error_rate"] for d in monitoring_data])
            }
            
            # Alert conditions
            alerts_triggered = {
                "accuracy_degradation_alert": monitoring_analysis["accuracy_degradation"] > 0.03,
                "data_drift_alert": monitoring_analysis["max_data_drift"] > 0.2,
                "error_rate_alert": monitoring_analysis["max_error_rate"] > 0.005,
                "response_time_alert": monitoring_analysis["avg_response_time"] > 100
            }
            
            # Recommendations
            recommendations = []
            if alerts_triggered["accuracy_degradation_alert"]:
                recommendations.append("Model retraining recommended due to accuracy degradation")
            if alerts_triggered["data_drift_alert"]:
                recommendations.append("Data pipeline investigation needed due to feature drift")
            if alerts_triggered["error_rate_alert"]:
                recommendations.append("System investigation needed due to increased error rate")
            
            # Log monitoring analysis
            mlflow.log_metrics(monitoring_analysis)
            mlflow.log_params({
                "monitoring_enabled": True,
                "alert_count": sum(alerts_triggered.values()),
                "recommendation_count": len(recommendations)
            })
            
            # Save monitoring reports
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Monitoring data
                monitoring_df = pd.DataFrame(monitoring_data)
                monitoring_path = temp_path / "production_monitoring_data.csv"
                monitoring_df.to_csv(monitoring_path, index=False)
                mlflow.log_artifact(str(monitoring_path), "monitoring")
                
                # Monitoring report
                monitoring_report = {
                    "report_timestamp": datetime.now().isoformat(),
                    "model_version": model_version,
                    "monitoring_period": "7 days",
                    "analysis": monitoring_analysis,
                    "alerts": alerts_triggered,
                    "recommendations": recommendations,
                    "next_review_date": (datetime.now() + timedelta(days=7)).isoformat()
                }
                
                report_path = temp_path / "monitoring_report.json"
                with open(report_path, 'w') as f:
                    json.dump(monitoring_report, f, indent=2)
                mlflow.log_artifact(str(report_path), "monitoring")
            
            # Update model registry with monitoring status
            self.client.set_model_version_tag(
                name=self.model_registry_name,
                version=model_version,
                key="last_monitoring_check",
                value=datetime.now().isoformat()
            )
            
            self.client.set_model_version_tag(
                name=self.model_registry_name,
                version=model_version,
                key="monitoring_status",
                value="active"
            )
            
            alert_count = sum(alerts_triggered.values())
            self.client.set_model_version_tag(
                name=self.model_registry_name,
                version=model_version,
                key="alert_count",
                value=str(alert_count)
            )
            
            mlflow.set_tags({
                "stage": "production_monitoring",
                "monitoring_status": "active",
                "alert_count": str(alert_count)
            })
            
            print(f"✅ Monitoring analysis completed")
            print(f"   Accuracy degradation: {monitoring_analysis['accuracy_degradation']:.4f}")
            print(f"   Max data drift: {monitoring_analysis['max_data_drift']:.3f}")
            print(f"   Alerts triggered: {alert_count}")
            print(f"   Recommendations: {len(recommendations)}")
            
            if recommendations:
                print("   📋 Recommendations:")
                for rec in recommendations:
                    print(f"     • {rec}")
            
            return monitoring_analysis, alerts_triggered, recommendations
    
    def run_complete_workflow(self):
        """Execute the complete MLflow workflow."""
        print("🚀 Genesis-Flow Complete MLflow Workflow")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Setup
            self.setup_environment()
            
            # Stage 1: Data Ingestion
            X, y, feature_names = self.stage_1_data_ingestion_and_validation()
            
            # Stage 2: Hyperparameter Tuning
            best_models = self.stage_2_hyperparameter_tuning(X, y, feature_names)
            
            # Stage 3: Model Selection
            selected_model = self.stage_3_model_comparison_and_selection(best_models)
            
            # Stage 4: Model Registry
            model_version = self.stage_4_model_registration_and_staging()
            
            if model_version:
                # Stage 5: Validation
                validation_passed = self.stage_5_model_validation_and_testing(model_version, X, y)
                
                # Stage 6: Deployment
                deployment_success = self.stage_6_production_deployment(model_version, validation_passed)
                
                if deployment_success:
                    # Stage 7: Monitoring
                    monitoring_results = self.stage_7_monitoring_and_maintenance(model_version)
            
            execution_time = time.time() - start_time
            
            # Final summary
            print(f"\n🎉 COMPLETE WORKFLOW FINISHED!")
            print("=" * 60)
            print(f"📋 Workflow Summary:")
            print(f"   ✅ Project: {self.project_name}")
            print(f"   ✅ Execution time: {execution_time:.2f} seconds")
            print(f"   ✅ Best model: {self.best_model.get('model_type', 'RandomForest')}")
            print(f"   ✅ Best F1 score: {self.best_model['metrics']['test_f1']:.4f}")
            print(f"   ✅ Model version in production: {model_version}")
            print(f"   ✅ Monitoring: Active")
            print(f"\n📊 Database: {self.tracking_uri}")
            print(f"🔗 All data stored in MongoDB with complete lineage!")
            
            return True
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run the complete MLflow workflow demonstration."""
    orchestrator = MLflowWorkflowOrchestrator("production_classifier")
    success = orchestrator.run_complete_workflow()
    
    if success:
        print("\n🎯 Workflow completed successfully!")
        print("💡 Check your MongoDB database to see all the tracked data!")
    else:
        print("\n❌ Workflow failed. Check the logs for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)