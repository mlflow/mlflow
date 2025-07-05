# Genesis-Flow MongoDB Integration

This document explains how to configure and use MongoDB/Azure Cosmos DB as the tracking store backend for Genesis-Flow.

## Architecture Overview

Genesis-Flow uses a **hybrid storage architecture**:

- **MongoDB/Azure Cosmos DB**: Experiment metadata, runs, parameters, metrics, tags
- **Azure Blob Storage**: Model artifacts, files, logs, notebooks  
- **Motor**: Async MongoDB driver for high performance

This design provides:
- ✅ Scalable metadata storage in cloud-native database
- ✅ Efficient artifact storage in object storage
- ✅ High performance with async operations
- ✅ Azure integration for enterprise deployments

## Configuration

### Environment Variables

Set one of these environment variables to configure MongoDB:

```bash
# Option 1: Azure Cosmos DB (Recommended for production)
export GENESIS_FLOW_COSMOS_DB_URI="mongodb://account:password@account.mongo.cosmos.azure.com:10255/genesis_flow?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000"

# Option 2: MongoDB Atlas
export GENESIS_FLOW_MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/genesis_flow?retryWrites=true&w=majority"

# Option 3: Local MongoDB (Development)
export GENESIS_FLOW_MONGODB_URI="mongodb://localhost:27017/genesis_flow"

# Artifact storage (Azure Blob Storage)
export GENESIS_FLOW_ARTIFACT_ROOT="azure://storageaccount.blob.core.windows.net/genesis-flow-artifacts"
```

### Azure Cosmos DB Setup

1. **Create Cosmos DB Account** with MongoDB API:
   ```bash
   az cosmosdb create \
     --name genesis-flow-cosmos \
     --resource-group genesis-flow-rg \
     --kind MongoDB \
     --server-version 4.2 \
     --locations regionName=eastus
   ```

2. **Get Connection String**:
   ```bash
   az cosmosdb keys list \
     --name genesis-flow-cosmos \
     --resource-group genesis-flow-rg \
     --type connection-strings
   ```

3. **Create Database and Collections**:
   Collections are created automatically with proper indexing.

### Azure Blob Storage Setup

1. **Create Storage Account**:
   ```bash
   az storage account create \
     --name genesisflowstorage \
     --resource-group genesis-flow-rg \
     --location eastus \
     --sku Standard_LRS
   ```

2. **Create Container**:
   ```bash
   az storage container create \
     --name genesis-flow-artifacts \
     --account-name genesisflowstorage
   ```

## Database Schema

### Collections

Genesis-Flow creates the following MongoDB collections:

#### `experiments`
```javascript
{
  "experiment_id": "123456789",
  "name": "fraud_detection_v1",
  "artifact_location": "azure://storage/artifacts/123456789",
  "lifecycle_stage": "active",
  "creation_time": 1609459200000,
  "last_update_time": 1609459200000,
  "tags": [
    {"key": "model_type", "value": "classification"},
    {"key": "team", "value": "ml-platform"}
  ]
}
```

#### `runs`
```javascript
{
  "run_uuid": "a1b2c3d4e5f6",
  "experiment_id": "123456789",
  "user_id": "data-scientist@company.com",
  "status": "FINISHED",
  "start_time": 1609459200000,
  "end_time": 1609459800000,
  "artifact_uri": "azure://storage/artifacts/123456789/a1b2c3d4e5f6",
  "lifecycle_stage": "active"
}
```

#### `params`
```javascript
{
  "run_uuid": "a1b2c3d4e5f6",
  "key": "learning_rate",
  "value": "0.01"
}
```

#### `metrics`
```javascript
{
  "run_uuid": "a1b2c3d4e5f6",
  "key": "accuracy",
  "value": 0.95,
  "timestamp": 1609459500000,
  "step": 100
}
```

#### `tags`
```javascript
{
  "run_uuid": "a1b2c3d4e5f6",
  "key": "framework",
  "value": "scikit-learn"
}
```

### Indexes

Automatic indexes are created for optimal query performance:

- **experiments**: `experiment_id` (unique), `name` (unique), `lifecycle_stage`, `creation_time`
- **runs**: `run_uuid` (unique), `experiment_id`, `status`, `start_time`, compound indexes
- **params**: `(run_uuid, key)` (unique), `run_uuid`, `key`
- **metrics**: `(run_uuid, key, timestamp)`, `(run_uuid, key, step)`, individual fields
- **tags**: `(run_uuid, key)` (unique), `run_uuid`, `key`

## Usage

### Starting Genesis-Flow Server

```bash
# With MongoDB tracking store
export GENESIS_FLOW_MONGODB_URI="mongodb://localhost:27017/genesis_flow"
export GENESIS_FLOW_ARTIFACT_ROOT="azure://artifacts"

genesis-flow server \
  --backend-store-uri mongodb://localhost:27017/genesis_flow \
  --default-artifact-root azure://artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### Python Client Usage

```python
import mlflow
from mlflow.store.tracking.mongodb_store import MongoDBStore

# Configure tracking URI
mlflow.set_tracking_uri("mongodb://localhost:27017/genesis_flow")

# Create experiment
experiment_id = mlflow.create_experiment(
    name="fraud_detection",
    artifact_location="azure://artifacts/fraud_detection"
)

# Start run and log data
with mlflow.start_run(experiment_id=experiment_id):
    # Log parameters
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("precision", 0.93)
    
    # Log artifacts to Azure Blob Storage
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("feature_importance.png")
```

### Direct Store Usage

```python
from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.entities import ExperimentTag

# Initialize store
store = MongoDBStore(
    db_uri="mongodb://localhost:27017/genesis_flow",
    default_artifact_root="azure://artifacts"
)

# Create experiment
experiment_id = store.create_experiment(
    name="model_comparison",
    artifact_location="azure://artifacts/model_comparison",
    tags=[
        ExperimentTag("team", "ml-platform"),
        ExperimentTag("project", "fraud-detection")
    ]
)

# Search experiments
experiments = store.search_experiments(
    view_type=ViewType.ACTIVE_ONLY,
    max_results=10
)
```

## Performance Considerations

### MongoDB Optimization

1. **Connection Pooling**: Configured automatically with optimal pool sizes
2. **Indexes**: All collections have performance-optimized indexes
3. **Async Operations**: Uses Motor async driver for high throughput
4. **Write Concerns**: Configured for data consistency

### Azure Cosmos DB Optimization

1. **Request Units (RUs)**: Scale based on workload
2. **Partition Strategy**: Experiments partitioned by `experiment_id`
3. **Consistency Level**: Session consistency for optimal performance
4. **Multi-region**: Replicate across regions for availability

### Monitoring

```python
# Enable MongoDB logging
import logging
logging.getLogger("mlflow.store.tracking.mongodb_store").setLevel(logging.INFO)

# Monitor connection health
store = MongoDBStore("mongodb://localhost:27017/genesis_flow")
# Check store.client.admin.command("ping") for connectivity
```

## Security

### Authentication

```bash
# MongoDB with authentication
export GENESIS_FLOW_MONGODB_URI="mongodb://username:password@host:27017/genesis_flow?authSource=admin"

# Azure Cosmos DB with managed identity
export GENESIS_FLOW_COSMOS_DB_URI="mongodb://cosmos-account:connection-string@cosmos-account.mongo.cosmos.azure.com:10255/genesis_flow?ssl=true"
```

### Network Security

- Use SSL/TLS for all connections
- Configure VNet integration for Azure
- Implement IP allowlisting
- Use Azure Private Endpoints

### Data Encryption

- **At Rest**: Enabled by default in Azure Cosmos DB
- **In Transit**: SSL/TLS required for all connections
- **Application Level**: Sensitive data can be encrypted before storage

## Troubleshooting

### Common Issues

1. **Connection Timeout**:
   ```bash
   # Increase timeout values
   export GENESIS_FLOW_MONGODB_URI="mongodb://host:27017/genesis_flow?serverSelectionTimeoutMS=10000"
   ```

2. **Authentication Failed**:
   ```bash
   # Verify credentials and auth source
   export GENESIS_FLOW_MONGODB_URI="mongodb://user:pass@host:27017/genesis_flow?authSource=admin"
   ```

3. **SSL Certificate Issues**:
   ```bash
   # For development only (not production)
   export GENESIS_FLOW_MONGODB_URI="mongodb://host:27017/genesis_flow?ssl=true&ssl_cert_reqs=CERT_NONE"
   ```

### Testing Connection

```python
from mlflow.store.tracking.mongodb_config import MongoDBConfig

# Test configuration
config = MongoDBConfig.get_connection_config()
print(f"Database: {config['database_name']}")
print(f"URI: {config['uri']}")

# Validate setup
is_valid = MongoDBConfig.validate_configuration()
print(f"Configuration valid: {is_valid}")
```

### Logs

```bash
# Enable debug logging
export PYTHONPATH=/path/to/genesis-flow
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from mlflow.store.tracking.mongodb_store import MongoDBStore
store = MongoDBStore('mongodb://localhost:27017/test')
"
```

## Migration

### From File Store

```python
# Script to migrate from file store to MongoDB
import mlflow
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.mongodb_store import MongoDBStore

# Source and target stores
file_store = FileStore("/path/to/mlruns")
mongo_store = MongoDBStore("mongodb://localhost:27017/genesis_flow")

# Migrate experiments
for exp in file_store.search_experiments():
    mongo_store.create_experiment(
        name=exp.name,
        artifact_location=exp.artifact_location,
        tags=exp.tags
    )
```

### From SQLAlchemy Store

```python
# Migrate from SQLite/PostgreSQL to MongoDB
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.mongodb_store import MongoDBStore

sql_store = SqlAlchemyStore("sqlite:///mlflow.db")
mongo_store = MongoDBStore("mongodb://localhost:27017/genesis_flow")

# Migration logic here...
```

## Production Deployment

### High Availability

```yaml
# Azure Cosmos DB with multi-region
regions:
  - East US 2 (Primary)
  - West US 2 (Secondary)
  - Europe West (Read)

consistency: Session
automatic_failover: true
multi_master: false
```

### Backup Strategy

```bash
# Automated backups in Azure Cosmos DB
az cosmosdb sql backup policy update \
  --account-name genesis-flow-cosmos \
  --resource-group genesis-flow-rg \
  --backup-interval 60 \
  --backup-retention 720
```

### Monitoring

```yaml
# Azure Monitor alerts
alerts:
  - name: "High RU Consumption"
    metric: "TotalRequestUnits"
    threshold: 10000
    
  - name: "Connection Failures"
    metric: "ConnectionErrors"
    threshold: 10
```

This completes the MongoDB integration for Genesis-Flow, providing a scalable, cloud-native backend for ML experiment tracking with Azure integration.