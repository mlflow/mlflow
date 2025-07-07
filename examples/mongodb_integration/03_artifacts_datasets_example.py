#!/usr/bin/env python
"""
Comprehensive Artifacts and Datasets Example with MongoDB Backend

This example demonstrates logging various types of artifacts and datasets:
- Data files (CSV, JSON, Parquet)
- Images and plots (PNG, SVG)
- Model artifacts and configurations
- Code and notebooks
- Custom artifacts and metadata
- Dataset versioning and lineage
"""

import sys
import os
import tempfile
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# For examples
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def setup_mongodb_tracking():
    """Configure Genesis-Flow to use MongoDB backend."""
    tracking_uri = "mongodb://localhost:27017/genesis_flow_artifacts"
    registry_uri = "mongodb://localhost:27017/genesis_flow_artifacts"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    
    return MlflowClient()


def log_data_artifacts_example():
    """Example 1: Log various data formats as artifacts."""
    print("\n📊 Example 1: Data Artifacts")
    print("=" * 35)
    
    experiment_name = "data_artifacts_demo"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="data_logging_demo") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['created_at'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
        
        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Log CSV data
            csv_path = temp_path / "training_data.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), "data")
            print("  ✅ Logged training_data.csv")
            
            # 2. Log JSON metadata
            metadata = {
                "dataset_info": {
                    "name": "synthetic_classification_data",
                    "version": "1.0",
                    "samples": len(df),
                    "features": len(feature_names),
                    "classes": len(np.unique(y)),
                    "created_by": "data_pipeline_v2",
                    "data_quality_score": 0.95
                },
                "feature_info": {
                    name: {
                        "type": "numerical",
                        "min": float(df[name].min()),
                        "max": float(df[name].max()),
                        "mean": float(df[name].mean()),
                        "std": float(df[name].std())
                    } for name in feature_names
                },
                "target_distribution": {
                    str(class_): int(count) 
                    for class_, count in zip(*np.unique(y, return_counts=True))
                }
            }
            
            json_path = temp_path / "dataset_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(json_path), "metadata")
            print("  ✅ Logged dataset_metadata.json")
            
            # 3. Log data summary statistics
            summary_stats = df.describe()
            stats_path = temp_path / "summary_statistics.csv"
            summary_stats.to_csv(stats_path)
            mlflow.log_artifact(str(stats_path), "analysis")
            print("  ✅ Logged summary_statistics.csv")
            
            # 4. Log correlation matrix
            correlation_matrix = df[feature_names].corr()
            corr_path = temp_path / "correlation_matrix.csv"
            correlation_matrix.to_csv(corr_path)
            mlflow.log_artifact(str(corr_path), "analysis")
            print("  ✅ Logged correlation_matrix.csv")
            
            # 5. Log data quality report
            quality_report = {
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "quality_checks": {
                    "no_missing_values": df.isnull().sum().sum() == 0,
                    "no_duplicates": df.duplicated().sum() == 0,
                    "balanced_classes": max(np.bincount(y)) / min(np.bincount(y)) < 2.0
                }
            }
            
            quality_path = temp_path / "data_quality_report.json"
            with open(quality_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            mlflow.log_artifact(str(quality_path), "quality")
            print("  ✅ Logged data_quality_report.json")
        
        # Log metadata as parameters
        mlflow.log_params({
            "dataset_samples": len(df),
            "dataset_features": len(feature_names),
            "dataset_classes": len(np.unique(y)),
            "data_quality_score": 0.95
        })
        
        print(f"✅ Logged {5} data artifacts with comprehensive metadata")
        return run.info.run_id


def log_visualization_artifacts_example():
    """Example 2: Log various visualizations and plots."""
    print("\n📈 Example 2: Visualization Artifacts")
    print("=" * 40)
    
    experiment_name = "visualization_artifacts_demo"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="visualization_demo") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Generate data
        X, y = make_classification(n_samples=500, n_features=8, n_classes=3, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Distribution plots
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(feature_names[:6]):
                plt.subplot(2, 3, i + 1)
                for class_label in np.unique(y):
                    plt.hist(df[df['target'] == class_label][feature], 
                            alpha=0.7, label=f'Class {class_label}', bins=20)
                plt.title(f'{feature} Distribution')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.legend()
            
            plt.tight_layout()
            dist_path = temp_path / "feature_distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(dist_path), "visualizations")
            plt.close()
            print("  ✅ Logged feature_distributions.png")
            
            # 2. Correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[feature_names].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            
            heatmap_path = temp_path / "correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(heatmap_path), "visualizations")
            plt.close()
            print("  ✅ Logged correlation_heatmap.png")
            
            # 3. Pairplot for key features
            plt.figure(figsize=(12, 10))
            key_features = feature_names[:4] + ['target']
            plot_df = df[key_features].copy()
            
            # Create pairplot manually (simplified)
            n_features = len(key_features) - 1
            fig, axes = plt.subplots(n_features, n_features, figsize=(12, 10))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        # Diagonal: histograms
                        for class_label in np.unique(y):
                            axes[i, j].hist(df[df['target'] == class_label][key_features[i]], 
                                          alpha=0.7, label=f'Class {class_label}')
                        axes[i, j].set_title(f'{key_features[i]}')
                    else:
                        # Off-diagonal: scatter plots
                        for class_label in np.unique(y):
                            class_data = df[df['target'] == class_label]
                            axes[i, j].scatter(class_data[key_features[j]], 
                                             class_data[key_features[i]], 
                                             alpha=0.6, label=f'Class {class_label}')
                        axes[i, j].set_xlabel(key_features[j])
                        axes[i, j].set_ylabel(key_features[i])
            
            plt.tight_layout()
            pairplot_path = temp_path / "feature_pairplot.png"
            plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(pairplot_path), "visualizations")
            plt.close()
            print("  ✅ Logged feature_pairplot.png")
            
            # 4. Target distribution
            plt.figure(figsize=(8, 6))
            class_counts = np.bincount(y)
            plt.bar(range(len(class_counts)), class_counts, alpha=0.7)
            plt.title('Target Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            for i, count in enumerate(class_counts):
                plt.text(i, count + 5, str(count), ha='center')
            
            target_dist_path = temp_path / "target_distribution.png"
            plt.savefig(target_dist_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(target_dist_path), "visualizations")
            plt.close()
            print("  ✅ Logged target_distribution.png")
            
            # 5. Save plot configurations
            plot_config = {
                "visualization_settings": {
                    "dpi": 300,
                    "format": "png",
                    "style": "seaborn",
                    "figure_size": [12, 10],
                    "color_palette": "coolwarm"
                },
                "plots_generated": [
                    "feature_distributions.png",
                    "correlation_heatmap.png", 
                    "feature_pairplot.png",
                    "target_distribution.png"
                ],
                "plot_descriptions": {
                    "feature_distributions": "Histograms showing distribution of features by class",
                    "correlation_heatmap": "Correlation matrix heatmap for all features",
                    "feature_pairplot": "Pairwise relationships between key features",
                    "target_distribution": "Bar chart of target class frequencies"
                }
            }
            
            config_path = temp_path / "visualization_config.json"
            with open(config_path, 'w') as f:
                json.dump(plot_config, f, indent=2)
            mlflow.log_artifact(str(config_path), "config")
            print("  ✅ Logged visualization_config.json")
        
        # Log visualization metadata
        mlflow.log_params({
            "visualizations_created": 4,
            "plot_format": "png",
            "plot_dpi": 300
        })
        
        print(f"✅ Logged {4} visualization artifacts")
        return run.info.run_id


def log_model_artifacts_example():
    """Example 3: Log model-specific artifacts and configurations."""
    print("\n🤖 Example 3: Model Artifacts")
    print("=" * 35)
    
    experiment_name = "model_artifacts_demo"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="model_artifacts_demo") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        # Generate data and train model
        X, y = make_classification(n_samples=800, n_features=15, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Model configuration
            model_config = {
                "model_type": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "min_samples_split": model.min_samples_split,
                    "min_samples_leaf": model.min_samples_leaf,
                    "random_state": model.random_state
                },
                "training_config": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "classes": len(np.unique(y))
                },
                "performance_metrics": {
                    "accuracy": float(model.score(X_test, y_test)),
                    "n_trees": model.n_estimators,
                    "max_features": model.max_features_
                }
            }
            
            config_path = temp_path / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            mlflow.log_artifact(str(config_path), "model_config")
            print("  ✅ Logged model_config.json")
            
            # 2. Feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = temp_path / "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path), "model_analysis")
            print("  ✅ Logged feature_importance.csv")
            
            # 3. Cross-validation results
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_results = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "cv_min": float(cv_scores.min()),
                "cv_max": float(cv_scores.max())
            }
            
            cv_path = temp_path / "cross_validation_results.json"
            with open(cv_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
            mlflow.log_artifact(str(cv_path), "model_analysis")
            print("  ✅ Logged cross_validation_results.json")
            
            # 4. Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            report_path = temp_path / "classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            mlflow.log_artifact(str(report_path), "model_analysis")
            print("  ✅ Logged classification_report.json")
            
            # 5. Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            conf_df = pd.DataFrame(conf_matrix, 
                                 columns=[f'Predicted_{i}' for i in range(len(np.unique(y)))],
                                 index=[f'Actual_{i}' for i in range(len(np.unique(y)))])
            
            conf_path = temp_path / "confusion_matrix.csv"
            conf_df.to_csv(conf_path)
            mlflow.log_artifact(str(conf_path), "model_analysis")
            print("  ✅ Logged confusion_matrix.csv")
            
            # 6. Prediction probabilities sample
            prob_sample = pd.DataFrame(y_proba[:20], 
                                     columns=[f'prob_class_{i}' for i in range(y_proba.shape[1])])
            prob_sample['actual'] = y_test[:20]
            prob_sample['predicted'] = y_pred[:20]
            
            prob_path = temp_path / "prediction_probabilities_sample.csv"
            prob_sample.to_csv(prob_path, index=False)
            mlflow.log_artifact(str(prob_path), "model_analysis")
            print("  ✅ Logged prediction_probabilities_sample.csv")
            
            # 7. Model serialization (alternative format)
            model_pickle_path = temp_path / "model_backup.pkl"
            with open(model_pickle_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_pickle_path), "model_backup")
            print("  ✅ Logged model_backup.pkl")
        
        # Log the main model using MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train[:3]
        )
        print("  ✅ Logged main model via MLflow")
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": model.score(X_test, y_test),
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std()
        })
        
        print(f"✅ Logged {7} model artifacts plus main model")
        return run.info.run_id


def log_code_and_configuration_artifacts():
    """Example 4: Log code, notebooks, and configuration files."""
    print("\n💻 Example 4: Code and Configuration Artifacts")
    print("=" * 50)
    
    experiment_name = "code_artifacts_demo"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="code_and_config_demo") as run:
        print(f"📝 Started run: {run.info.run_id}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Training script
            training_script = '''#!/usr/bin/env python
"""
Model Training Script
Generated automatically for MLflow artifact logging demo.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    """Load and prepare data for training."""
    # Data loading logic here
    pass

def train_model(X_train, y_train):
    """Train the machine learning model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    print("Starting model training...")
    # Training pipeline would go here
    print("Training completed!")
'''
            
            script_path = temp_path / "train_model.py"
            with open(script_path, 'w') as f:
                f.write(training_script)
            mlflow.log_artifact(str(script_path), "code")
            print("  ✅ Logged train_model.py")
            
            # 2. Configuration file
            config_yaml = '''# Model Configuration
model:
  type: RandomForestClassifier
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42

data:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  random_state: 42

training:
  cross_validation:
    enabled: true
    folds: 5
    scoring: accuracy
  
  early_stopping:
    enabled: false
    patience: 10
    monitor: val_accuracy

logging:
  level: INFO
  artifacts:
    - model_config
    - feature_importance
    - confusion_matrix
    - classification_report

deployment:
  target: production
  monitoring: true
  auto_scaling: true
'''
            
            config_path = temp_path / "model_config.yaml"
            with open(config_path, 'w') as f:
                f.write(config_yaml)
            mlflow.log_artifact(str(config_path), "config")
            print("  ✅ Logged model_config.yaml")
            
            # 3. Requirements file
            requirements = '''# Python package requirements
scikit-learn==1.3.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
mlflow>=2.0.0
pyyaml>=6.0
joblib>=1.1.0
scipy>=1.9.0
'''
            
            req_path = temp_path / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write(requirements)
            mlflow.log_artifact(str(req_path), "environment")
            print("  ✅ Logged requirements.txt")
            
            # 4. Jupyter notebook (as text)
            notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Training Notebook\\n",
    "This notebook demonstrates the complete ML pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import mlflow\\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load and preprocess data\\n",
    "data = pd.read_csv('training_data.csv')\\n",
    "X = data.drop('target', axis=1)\\n",
    "y = data['target']"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
            
            notebook_path = temp_path / "model_training.ipynb"
            with open(notebook_path, 'w') as f:
                f.write(notebook_content)
            mlflow.log_artifact(str(notebook_path), "notebooks")
            print("  ✅ Logged model_training.ipynb")
            
            # 5. Environment information
            env_info = {
                "python_version": "3.8.10",
                "platform": "linux-x86_64",
                "packages": {
                    "scikit-learn": "1.3.0",
                    "pandas": "1.5.3",
                    "numpy": "1.21.6",
                    "mlflow": "2.0.1"
                },
                "environment_variables": {
                    "MLFLOW_TRACKING_URI": "mongodb://localhost:27017/genesis_flow_artifacts",
                    "PYTHONPATH": "/opt/ml/code"
                },
                "hardware": {
                    "cpu_cores": 4,
                    "memory_gb": 16,
                    "gpu": "none"
                }
            }
            
            env_path = temp_path / "environment_info.json"
            with open(env_path, 'w') as f:
                json.dump(env_info, f, indent=2)
            mlflow.log_artifact(str(env_path), "environment")
            print("  ✅ Logged environment_info.json")
        
        # Log metadata
        mlflow.log_params({
            "code_artifacts": 5,
            "python_version": "3.8.10",
            "environment": "production"
        })
        
        print(f"✅ Logged {5} code and configuration artifacts")
        return run.info.run_id


def demonstrate_artifact_retrieval(client):
    """Example 5: Retrieve and work with logged artifacts."""
    print("\n📦 Example 5: Artifact Retrieval")
    print("=" * 35)
    
    # Get recent runs
    experiments = client.search_experiments()
    if not experiments:
        print("❌ No experiments found")
        return
    
    # Get runs from the first experiment
    experiment_id = experiments[0].experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=1)
    
    if not runs:
        print("❌ No runs found")
        return
    
    run_id = runs[0].info.run_id
    print(f"🔍 Retrieving artifacts from run: {run_id}")
    
    try:
        # List all artifacts
        artifacts = client.list_artifacts(run_id)
        print(f"\n📋 Available artifacts ({len(artifacts)}):")
        
        for artifact in artifacts:
            if artifact.is_dir:
                print(f"  📁 {artifact.path}/")
                # List contents of directory
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts[:3]:  # Show first 3
                    print(f"    📄 {sub_artifact.path}")
                if len(sub_artifacts) > 3:
                    print(f"    ... and {len(sub_artifacts) - 3} more files")
            else:
                print(f"  📄 {artifact.path}")
        
        # Download specific artifacts
        print(f"\n⬇️  Downloading artifacts...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download an artifact directory
            if artifacts:
                for artifact in artifacts[:2]:  # Download first 2 artifacts/directories
                    try:
                        downloaded_path = client.download_artifacts(run_id, artifact.path, temp_dir)
                        print(f"  ✅ Downloaded: {artifact.path} -> {downloaded_path}")
                        
                        # If it's a file, show info
                        if not artifact.is_dir:
                            file_path = Path(downloaded_path)
                            if file_path.exists():
                                size_kb = file_path.stat().st_size / 1024
                                print(f"    📊 Size: {size_kb:.2f} KB")
                    
                    except Exception as e:
                        print(f"  ❌ Failed to download {artifact.path}: {e}")
    
    except Exception as e:
        print(f"❌ Error retrieving artifacts: {e}")


def main():
    """Run all artifact and dataset examples."""
    print("🚀 Genesis-Flow MongoDB Artifacts & Datasets Examples")
    print("=" * 65)
    
    # Setup
    client = setup_mongodb_tracking()
    
    # Run examples
    data_run_id = log_data_artifacts_example()
    viz_run_id = log_visualization_artifacts_example()
    model_run_id = log_model_artifacts_example()
    code_run_id = log_code_and_configuration_artifacts()
    
    # Demonstrate retrieval
    demonstrate_artifact_retrieval(client)
    
    print(f"\n🎉 ALL ARTIFACT EXAMPLES COMPLETED!")
    print("=" * 65)
    print(f"📋 Summary:")
    print(f"  ✅ Data artifacts run: {data_run_id}")
    print(f"  ✅ Visualization artifacts run: {viz_run_id}")
    print(f"  ✅ Model artifacts run: {model_run_id}")
    print(f"  ✅ Code artifacts run: {code_run_id}")
    print(f"  ✅ Artifact retrieval: Demonstrated")
    print(f"\n📊 Database: genesis_flow_artifacts")
    print(f"🔗 All artifacts metadata stored in MongoDB!")
    print(f"💡 Note: Actual artifact files would be stored in configured artifact store (Azure Blob, S3, etc.)")


if __name__ == "__main__":
    main()