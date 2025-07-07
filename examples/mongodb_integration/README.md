# Genesis-Flow MongoDB Integration Examples

This directory contains comprehensive examples demonstrating all MLflow functionality with MongoDB backend integration. These examples showcase the complete compatibility between Genesis-Flow's MongoDB implementation and standard MLflow operations.

## Overview

Genesis-Flow is a secure MLflow fork that provides direct MongoDB integration, eliminating the need for a separate MLflow tracking server. All examples store metadata in MongoDB while maintaining 100% API compatibility with standard MLflow.

Key benefits:
- **Direct MongoDB Integration**: Store experiment metadata directly in MongoDB/Azure Cosmos DB
- **Eliminate MLflow Server**: No need for separate tracking server infrastructure
- **100% API Compatibility**: All MLflow functions work exactly the same
- **Enhanced Performance**: Direct database connections for faster operations
- **Secure by Design**: Built-in security validation and model verification

## Examples Overview

### 1. Model Logging Examples (`01_model_logging_example.py`)

**Demonstrates**: Comprehensive model logging with different frameworks
- **Scikit-learn models** with signatures and metadata
- **Custom PyFunc models** with preprocessing pipelines  
- **Dataset tracking** with model training
- **Model versioning** for comparison
- **Model loading** from MongoDB storage

**Key Features**:
- Model signatures and input examples
- Feature importance logging
- Custom preprocessing pipelines
- Dataset artifact logging
- Multiple model versions for A/B testing

### 2. Model Registry Examples (`02_model_registry_example.py`)

**Demonstrates**: Complete MLflow Model Registry workflow
- **Model registration** with descriptions and metadata
- **Version management** across multiple model types
- **Stage transitions**: None → Staging → Production → Archived
- **Model aliases** for flexible deployment
- **Search and discovery** of registered models
- **Deployment workflows** with comprehensive metadata

**Key Features**:
- Multi-algorithm model comparison
- Stage-based promotion workflow
- Model alias management ("champion", "challenger", "stable")
- Comprehensive metadata tagging
- Production deployment simulation

### 3. Artifacts & Datasets Examples (`03_artifacts_datasets_example.py`)

**Demonstrates**: Comprehensive artifact management
- **Data artifacts**: CSV, JSON, Parquet files
- **Visualizations**: Plots, charts, analysis graphs
- **Model artifacts**: Configurations, analysis results
- **Code artifacts**: Scripts, notebooks, requirements
- **Artifact retrieval** and management

**Key Features**:
- Multiple data format support
- Rich visualization logging
- Model analysis artifacts
- Code and configuration versioning
- Artifact download and management

### 4. Complete MLflow Workflow (`04_complete_mlflow_workflow.py`)

**Demonstrates**: End-to-end production ML workflow
- **7-stage workflow**:
  1. Data ingestion and validation
  2. Hyperparameter tuning (GridSearch)
  3. Model comparison and selection
  4. Model registration and staging
  5. Model validation and testing
  6. Production deployment
  7. Monitoring and maintenance

**Key Features**:
- Multi-algorithm comparison (RandomForest, GradientBoosting, LogisticRegression)
- Automated model selection based on F1 score
- Complete validation pipeline
- Production deployment simulation
- Monitoring and alerting simulation
- Comprehensive reporting at each stage

## Prerequisites

### 1. MongoDB Setup

Choose one of the following options:

#### Option A: Local MongoDB
```bash
# Install MongoDB locally
brew install mongodb/brew/mongodb-community
brew services start mongodb/brew/mongodb-community

# Verify MongoDB is running
mongosh --eval "db.adminCommand('ismaster')"
```

#### Option B: MongoDB Atlas (Cloud)
1. Create a free MongoDB Atlas account
2. Create a cluster
3. Get the connection string
4. Set the environment variable

#### Option C: Azure Cosmos DB
1. Create an Azure Cosmos DB account with MongoDB API
2. Get the connection string from Azure portal
3. Set the environment variable

### 2. Python Dependencies

```bash
# Install required packages
pip install scikit-learn pandas numpy matplotlib seaborn pymongo motor

# Install Genesis-Flow (from local development)
cd /path/to/genesis-flow
pip install -e .
```

## Running the Examples

### Individual Examples
```bash
# Run each example independently
python 01_model_logging_example.py
python 02_model_registry_example.py
python 03_artifacts_datasets_example.py
python 04_complete_mlflow_workflow.py
```

### All Examples Sequentially
```bash
# Run all examples in sequence
for script in 01_model_logging_example.py 02_model_registry_example.py 03_artifacts_datasets_example.py 04_complete_mlflow_workflow.py; do
    echo "Running $script..."
    python $script
    echo "Completed $script"
    echo "========================="
done
```

## Configuration Examples

### Basic MongoDB Configuration
```python
import mlflow

# Local MongoDB
tracking_uri = "mongodb://localhost:27017/genesis_flow"
registry_uri = "mongodb://localhost:27017/genesis_flow"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(registry_uri)
```

### Azure Cosmos DB Configuration
```python
# Azure Cosmos DB with connection string
cosmos_uri = "mongodb://account:key@account.mongo.cosmos.azure.com:10255/database?ssl=true&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@account@"

mlflow.set_tracking_uri(cosmos_uri)
mlflow.set_registry_uri(cosmos_uri)
```

### Environment Variables (Optional)
```bash
# MongoDB Connection (choose one)

# Local MongoDB
export MLFLOW_TRACKING_URI="mongodb://localhost:27017/genesis_flow_test"

# MongoDB Atlas
export MLFLOW_TRACKING_URI="mongodb+srv://username:password@cluster.mongodb.net/genesis_flow_test"

# Azure Cosmos DB
export MLFLOW_TRACKING_URI="mongodb://username:password@cluster.mongo.cosmos.azure.com:10255/genesis_flow_test?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@cluster@"

# Artifact Storage (optional)
export MLFLOW_DEFAULT_ARTIFACT_ROOT="file:///tmp/genesis_flow_artifacts"
# or for Azure Blob Storage
export MLFLOW_DEFAULT_ARTIFACT_ROOT="azure://container/path"
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
```

## MongoDB Storage Details

### Database Structure

Each example uses a dedicated MongoDB database:
- `genesis_flow_models`: Model logging examples
- `genesis_flow_registry`: Model registry examples  
- `genesis_flow_artifacts`: Artifacts and datasets
- `genesis_flow_production_classifier`: Complete workflow

### Collections Schema

**Experiments Collection**:
```json
{
  "_id": "experiment_id",
  "name": "experiment_name", 
  "artifact_location": "azure://artifacts/exp_id",
  "lifecycle_stage": "active",
  "tags": {},
  "creation_time": 1234567890,
  "last_update_time": 1234567890
}
```

**Runs Collection**:
```json
{
  "_id": "run_id",
  "experiment_id": "exp_id",
  "user_id": "user",
  "status": "FINISHED",
  "start_time": 1234567890,
  "end_time": 1234567890,
  "artifact_uri": "azure://artifacts/exp_id/run_id",
  "lifecycle_stage": "active",
  "name": "run_name"
}
```

**Parameters/Metrics/Tags Collections**:
```json
{
  "_id": ObjectId(),
  "run_id": "run_id",
  "key": "parameter_name",
  "value": "parameter_value",
  "timestamp": 1234567890
}
```

## Verification and Testing

### Check MongoDB Data
```python
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")

# List databases created by examples
print("Databases:", [db for db in client.list_database_names() if 'genesis_flow' in db])

# Check collections in a database
db = client['genesis_flow_models']
print("Collections:", db.list_collection_names())

# Count documents
print("Experiments:", db.experiments.count_documents({}))
print("Runs:", db.runs.count_documents({}))
print("Parameters:", db.params.count_documents({}))
print("Metrics:", db.metrics.count_documents({}))
```

### Verify Model Loading
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("mongodb://localhost:27017/genesis_flow_models")

# Load latest model
experiment = mlflow.get_experiment_by_name("sklearn_model_logging_demo")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
latest_run = runs.iloc[0]['run_id']

# Load and test model
model = mlflow.sklearn.load_model(f"runs:/{latest_run}/model")
print("Model loaded successfully:", type(model).__name__)
```

## Integration with Azure

### Azure Cosmos DB
- Provides global distribution and automatic scaling
- Compatible with MongoDB API
- Built-in backup and disaster recovery

### Azure Blob Storage
- Scalable artifact storage
- Integration with Azure Machine Learning
- Lifecycle management policies

### Example Azure Configuration
```python
# Azure Cosmos DB connection
cosmos_connection = "mongodb://genesis-flow:key@genesis-flow.mongo.cosmos.azure.com:10255/mlflow?ssl=true&replicaSet=globaldb"

# Azure Blob Storage for artifacts
artifact_root = "azure://mlflow-artifacts@genesisflow.blob.core.windows.net/"

mlflow.set_tracking_uri(cosmos_connection)
mlflow.set_registry_uri(cosmos_connection)
```

## API Compatibility

Genesis-Flow maintains 100% API compatibility with MLflow:

- ✅ All `mlflow.log_*()` functions
- ✅ All `mlflow.sklearn.*`, `mlflow.pytorch.*`, etc. functions  
- ✅ Model Registry operations
- ✅ Experiment and run management
- ✅ Artifact logging and retrieval
- ✅ Search and filtering operations

## Troubleshooting

### Common Issues

1. **Connection Timeout**:
   ```python
   # Increase timeout
   client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=30000)
   ```

2. **Missing Collections**:
   ```python
   # Collections are created automatically on first write
   # Ensure MongoDB is running and accessible
   ```

3. **Artifact Storage**:
   ```python
   # For local testing, use file:// URI
   mlflow.set_tracking_uri("mongodb://localhost:27017/test")
   # Artifacts will be stored locally by default
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable MLflow debug logging
import mlflow
mlflow.set_tracking_uri("mongodb://localhost:27017/debug")
```

## Performance Considerations

### MongoDB Optimization
- **Indexes**: Automatically created for run_id, experiment_id queries
- **Connection Pooling**: Configured for optimal concurrent access
- **Write Concerns**: Set to 'majority' for data consistency

### Scaling Recommendations
- **Azure Cosmos DB**: Use for production workloads requiring global distribution
- **MongoDB Atlas**: Managed MongoDB service with automatic scaling
- **Sharding**: Consider for very large deployments (>1M runs)

## Architecture Benefits

### 1. No MLflow Server Required
- Direct MongoDB connection eliminates server dependency
- Reduced infrastructure complexity
- Lower operational overhead

### 2. Performance Improvements
- Direct database access reduces latency
- No HTTP REST API overhead
- Faster experiment logging and retrieval

### 3. Scalability
- MongoDB/Cosmos DB provides horizontal scaling
- Azure Blob Storage for artifact scaling
- Independent scaling of compute and storage

### 4. Security
- Direct database authentication
- No exposed MLflow server endpoints
- Secure model loading and validation

## Next Steps

1. **Production Deployment**: Configure Azure Cosmos DB and Blob Storage
2. **CI/CD Integration**: Add MongoDB tracking to your ML pipelines
3. **Monitoring**: Set up alerts and dashboards for model performance
4. **Scaling**: Consider sharding for large-scale deployments

## Support

- **Documentation**: See MLflow documentation for API reference
- **Issues**: Report Genesis-Flow specific issues to the project repository
- **Community**: Join the MLflow community for general questions

---

**Note**: These examples demonstrate the complete feature parity between Genesis-Flow's MongoDB backend and standard MLflow operations. All metadata is stored in MongoDB while maintaining full compatibility with the MLflow ecosystem.