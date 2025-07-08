# MLflow PostgreSQL to MongoDB Migration Tool

This tool provides comprehensive migration capabilities for transferring all MLflow data from PostgreSQL to MongoDB/Azure Cosmos DB with complete data integrity preservation.

## 🎯 Overview

The migration tool handles complete MLflow data migration including:
- ✅ Experiments, runs, metrics, parameters, tags
- ✅ Model registry data (registered models, model versions)
- ✅ Data validation and integrity checks
- ✅ Batch processing for large datasets
- ✅ Progress tracking and detailed reporting
- ✅ Dry-run mode for safe testing

## 🚀 Quick Start

### Option 1: Quick Migration (Autonomize PostgreSQL → Local MongoDB)

```bash
# Dry run to analyze data (recommended first step)
python tools/migration/run_migration.py --quick-migrate-autonomize --dry-run

# Actual migration
python tools/migration/run_migration.py --quick-migrate-autonomize
```

### Option 2: Custom Database URIs

```bash
# Migrate to local MongoDB
python tools/migration/run_migration.py \
  --postgres-uri "postgresql://postgres:password@host:5432/mlflow" \
  --mongodb-uri "mongodb://localhost:27017/mlflow_migrated"

# Migrate to Azure Cosmos DB
python tools/migration/run_migration.py \
  --postgres-uri "postgresql://postgres:password@host:5432/mlflow" \
  --mongodb-uri "mongodb://account:key@account.mongo.cosmos.azure.com:10255/mlflow?ssl=true"
```

### Option 3: Configuration File

```bash
# Create configuration file
cp tools/migration/migration_config.example.json my_migration_config.json
# Edit configuration as needed

# Run migration with config
python tools/migration/run_migration.py --config my_migration_config.json
```

## 📋 Migration Process

### 1. Pre-Migration Analysis

The tool first analyzes your PostgreSQL database:

```
Analyzing source data in PostgreSQL...
  experiments: 284 records
  runs: 6,013 records
  metrics: 57,350 records
  params: 71,003 records
  tags: 34,791 records
  registered_models: 212 records
  model_versions: 1,164 records
Total records to migrate: 170,817
```

### 2. Data Migration

Data is migrated in the following order:
1. **Experiments** → MongoDB `experiments` collection
2. **Runs** → MongoDB `runs` collection
3. **Metrics** → MongoDB `metrics` collection  
4. **Parameters** → MongoDB `params` collection
5. **Tags** → MongoDB `tags` collection
6. **Registered Models** → MongoDB `registered_models` collection
7. **Model Versions** → MongoDB `model_versions` collection

### 3. Data Validation

Post-migration validation ensures:
- ✅ Record counts match between PostgreSQL and MongoDB
- ✅ Data types are correctly converted
- ✅ Indexes are properly created
- ✅ Referential integrity is maintained

## 🔧 Configuration Options

### Command Line Arguments

```bash
python tools/migration/postgres_to_mongodb.py \
  --postgres-uri "postgresql://user:pass@host:port/database" \
  --mongodb-uri "mongodb://host:port/database" \
  --batch-size 1000 \
  --dry-run
```

**Parameters:**
- `--postgres-uri`: PostgreSQL connection string
- `--mongodb-uri`: MongoDB connection string  
- `--batch-size`: Records per batch (default: 1000)
- `--dry-run`: Analyze without migration

### Configuration File Format

```json
{
  "source": {
    "connection_uri": "postgresql://user:pass@host:port/database"
  },
  "destination": {
    "recommended": "mongodb://localhost:27017/mlflow_migrated"
  },
  "migration_settings": {
    "batch_size": 1000,
    "dry_run": false,
    "validate_data": true
  }
}
```

## 📊 Example Migration Report

```
============================================================
MLFLOW POSTGRESQL TO MONGODB MIGRATION REPORT
============================================================
Migration Date: 2025-07-09 00:23:03
Source: postgresql://postgres:pass@host:5432/mlflow
Destination: mongodb://localhost:27017/mlflow_migrated
Batch Size: 1000
Dry Run: False

MIGRATION SUMMARY:
----------------------------------------
✅ EXPERIMENTS          Total:      284 | Migrated:      284 | Errors:     0
✅ RUNS                 Total:    6,013 | Migrated:    6,013 | Errors:     0
✅ METRICS              Total:   57,350 | Migrated:   57,350 | Errors:     0
✅ PARAMS               Total:   71,003 | Migrated:   71,003 | Errors:     0
✅ TAGS                 Total:   34,791 | Migrated:   34,791 | Errors:     0
✅ REGISTERED_MODELS    Total:      212 | Migrated:      212 | Errors:     0
✅ MODEL_VERSIONS       Total:    1,164 | Migrated:    1,164 | Errors:     0
----------------------------------------
TOTAL RECORDS:                Total:  170,817 | Migrated:  170,817 | Errors:     0

🎉 MIGRATION COMPLETED SUCCESSFULLY!
```

## 🗄️ Database Schema Mapping

### PostgreSQL → MongoDB Collections

| PostgreSQL Table | MongoDB Collection | Notes |
|---|---|---|
| `experiments` | `experiments` | Indexed on name, creation_time |
| `runs` | `runs` | Indexed on experiment_id, start_time, status |
| `metrics` | `metrics` | Indexed on run_uuid+key+step, timestamp |
| `params` | `params` | Indexed on run_uuid+key (unique) |
| `tags` | `tags` | Indexed on run_uuid+key (unique) |
| `registered_models` | `registered_models` | Indexed on name (unique), creation_timestamp |
| `model_versions` | `model_versions` | Indexed on name+version (unique), current_stage |

### Data Type Conversions

- **Timestamps**: PostgreSQL bigint → MongoDB int64
- **Status Values**: PostgreSQL enum/int → MongoDB string
- **UUIDs**: Preserved as strings for compatibility
- **JSON Fields**: Parsed and stored as MongoDB documents
- **Nullable Fields**: Properly handled with None values

## 🔍 Troubleshooting

### Common Issues

1. **Connection Timeout**
   ```bash
   # Increase connection timeout
   export PGCONNECT_TIMEOUT=30
   ```

2. **Large Dataset Migration**
   ```bash
   # Reduce batch size for memory optimization
   python tools/migration/run_migration.py --batch-size 500
   ```

3. **Network Issues**
   ```bash
   # Test connections separately
   psql -h host -U user -d database -c "SELECT COUNT(*) FROM experiments;"
   mongo "mongodb://host:port/database" --eval "db.stats()"
   ```

4. **Permission Issues**
   ```bash
   # Ensure read access on PostgreSQL
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO migration_user;
   
   # Ensure write access on MongoDB
   # Create user with readWrite role on target database
   ```

### Validation Commands

```bash
# Verify PostgreSQL data
psql -h host -U user -d database -c "
  SELECT 'experiments' as table_name, COUNT(*) as count FROM experiments
  UNION ALL SELECT 'runs', COUNT(*) FROM runs
  UNION ALL SELECT 'metrics', COUNT(*) FROM metrics;"

# Verify MongoDB data  
mongo "mongodb://localhost:27017/mlflow_migrated" --eval "
  db.experiments.count() + ' experiments, ' + 
  db.runs.count() + ' runs, ' + 
  db.metrics.count() + ' metrics'"
```

## 📦 Post-Migration Steps

### 1. Update Application Configuration

```python
import mlflow

# Update MLflow to use MongoDB backend
mlflow.set_tracking_uri("mongodb://localhost:27017/mlflow_migrated")
mlflow.set_registry_uri("mongodb://localhost:27017/mlflow_migrated")

# Test basic operations
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")
```

### 2. Verify Data Integrity

```bash
# Run compatibility tests
python run_compatibility_tests.py

# Test with migrated data
python -c "
import mlflow
mlflow.set_tracking_uri('mongodb://localhost:27017/mlflow_migrated')
runs = mlflow.search_runs()
print(f'Successfully accessed {len(runs)} runs')
"
```

### 3. Performance Optimization

```python
# Create additional indexes for better performance
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['mlflow_migrated']

# Create composite indexes for common queries
db.runs.create_index([("experiment_id", 1), ("start_time", -1)])
db.metrics.create_index([("run_uuid", 1), ("timestamp", -1)])
```

## 🚀 Migration Performance

### Typical Performance

- **Small Dataset** (< 10K records): 1-5 minutes
- **Medium Dataset** (10K-100K records): 5-30 minutes  
- **Large Dataset** (100K-1M records): 30 minutes - 2 hours
- **Very Large Dataset** (> 1M records): 2+ hours

### Optimization Tips

1. **Batch Size**: Adjust based on available memory
   - Small systems: 500-1000 records
   - Large systems: 2000-5000 records

2. **Network**: Run migration from same network as databases

3. **Resources**: Ensure adequate disk space and memory

4. **Parallel Processing**: For very large datasets, consider table-specific migrations

## 📞 Support

For migration issues:
1. Check the generated migration log files
2. Review the migration report
3. Test with `--dry-run` first
4. Validate connections before migration
5. Monitor system resources during migration

## 🔗 Related Documentation

- [Genesis-Flow MongoDB Compatibility](../../MONGODB_COMPATIBILITY_VERIFICATION.md)
- [MLflow Compatibility Tests](../../tests/integration/test_mlflow_compatibility.py)
- [MongoDB Integration Examples](../../examples/mongodb_integration/)