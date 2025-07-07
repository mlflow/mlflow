# Genesis-Flow MongoDB Integration Example

This example demonstrates how to use Genesis-Flow with MongoDB as the tracking store, eliminating the need for a separate MLflow server.

## Overview

Genesis-Flow provides direct MongoDB integration that allows you to:
- Store experiment metadata directly in MongoDB/Azure Cosmos DB
- Eliminate the MLflow server dependency
- Maintain 100% MLflow API compatibility
- Achieve better performance with direct database connections

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
# Install Genesis-Flow and dependencies
pip install scikit-learn pandas numpy pymongo motor

# Install Genesis-Flow (from local development)
cd /path/to/genesis-flow
pip install -e .
```

## Configuration

### Environment Variables

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

## Running the Example

### Basic Test

```bash
cd examples/mongodb_integration
python test_genesis_flow_mongodb.py
```

### With Custom Configuration

```bash
# Test with local MongoDB
MLFLOW_TRACKING_URI="mongodb://localhost:27017/my_test_db" \
python test_genesis_flow_mongodb.py

# Test with Azure Cosmos DB
MLFLOW_TRACKING_URI="mongodb://your-cosmos-account.mongo.cosmos.azure.com:10255/genesis_flow?ssl=true" \
MLFLOW_DEFAULT_ARTIFACT_ROOT="azure://mlflow-artifacts/" \
python test_genesis_flow_mongodb.py
```

## What the Example Tests

### 1. Experiment Management
- ✅ Create experiments with metadata
- ✅ Set active experiments
- ✅ Add experiment tags

### 2. Model Training and Logging
- ✅ Train multiple ML models (Random Forest, Logistic Regression)
- ✅ Log parameters, metrics, and tags
- ✅ Log model artifacts with signatures
- ✅ Log additional files and reports

### 3. Model Loading and Inference
- ✅ Load models from MongoDB storage
- ✅ Perform inference with loaded models
- ✅ Test model prediction capabilities

### 4. Experiment Search and Comparison
- ✅ Search runs across experiments
- ✅ Compare model performance
- ✅ Retrieve run metadata and metrics

### 5. Genesis-Flow Specific Features
- ✅ Direct MongoDB connection testing
- ✅ Artifact storage validation
- ✅ Performance measurement

## Expected Output

```
🎯 Genesis-Flow MongoDB Integration Test
==================================================
🔗 Setting up Genesis-Flow with MongoDB tracking URI: mongodb://localhost:27017/genesis_flow_test
📁 Artifact storage: file:///tmp/genesis_flow_artifacts
📊 Creating sample dataset...
✅ Dataset created: 1000 samples, 20 features

🧪 Testing Experiment Management...
✅ Experiment created: genesis_flow_test_20241207_143022 (ID: 1)

🤖 Testing Model Training and Logging...
🔄 Training random_forest...
✅ random_forest logged - Accuracy: 0.9200, Time: 0.15s
🔄 Training logistic_regression...
✅ logistic_regression logged - Accuracy: 0.8800, Time: 0.08s

🔮 Testing Model Loading and Inference...
🔄 Loading random_forest from Genesis-Flow...
✅ random_forest loaded successfully
   Predictions: [1 0 1 0 1]
   Probabilities shape: (5, 2)
🔄 Loading logistic_regression from Genesis-Flow...
✅ logistic_regression loaded successfully
   Predictions: [0 1 1 0 1]
   Probabilities shape: (5, 2)

🔍 Testing Experiment Search and Comparison...
✅ Found 2 runs in experiment
📊 Top runs by accuracy:
  1. random_forest: Accuracy=0.92, Time=0.15s
  2. logistic_regression: Accuracy=0.88, Time=0.08s

🚀 Testing Genesis-Flow Specific Features...
✅ Tracking URI: mongodb://localhost:27017/genesis_flow_test
✅ Using MongoDB tracking store (Genesis-Flow)
✅ Genesis-Flow features test completed - Run ID: abc123...

🎉 Genesis-Flow MongoDB Integration Test Complete!
==================================================
📊 Experiment: genesis_flow_test_20241207_143022
🔗 Tracking URI: mongodb://localhost:27017/genesis_flow_test
📁 Artifacts: file:///tmp/genesis_flow_artifacts
🏃 Runs completed: 2
🏆 Best model: random_forest (Accuracy: 0.92)
```

## Architecture Benefits Demonstrated

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

## Troubleshooting

### Common Issues

#### MongoDB Connection Failed
```bash
# Check MongoDB is running
mongosh --eval "db.adminCommand('ismaster')"

# Check connection string format
echo $MLFLOW_TRACKING_URI
```

#### Permission Errors
```bash
# Ensure MongoDB user has read/write permissions
# Check Azure Cosmos DB firewall settings
```

#### Module Import Errors
```bash
# Ensure Genesis-Flow is installed
pip install -e /path/to/genesis-flow

# Check Python path
python -c "import mlflow; print(mlflow.__file__)"
```

### Performance Tips

1. **Use Connection Pooling**: MongoDB connections are pooled automatically
2. **Index Strategy**: Genesis-Flow creates appropriate indexes for queries
3. **Artifact Storage**: Use Azure Blob Storage for large artifacts
4. **Batch Operations**: Use MLflow's batch logging for better performance

## Next Steps

After running this example successfully:

1. **Integration Testing**: Test with your existing ML workflows
2. **Performance Benchmarking**: Compare with current MLflow server setup
3. **Production Setup**: Configure Azure Cosmos DB and Blob Storage
4. **Migration Planning**: Use the migration tools to move existing data

## Related Documentation

- [Genesis-Flow Architecture](../../docs/architecture/genesis-flow-architecture.md)
- [MongoDB Store Implementation](../../mlflow/store/tracking/mongodb_store.py)
- [Migration Guide](../../tools/migration/mlflow_to_genesis_flow.py)
- [Security Features](../../mlflow/utils/security_validation.py)