// MongoDB initialization script for Genesis-Flow testing

// Switch to the genesis_flow_test database
db = db.getSiblingDB('genesis_flow_test');

// Create a test user with read/write permissions
db.createUser({
  user: 'genesis_flow_user',
  pwd: 'genesis_flow_password',
  roles: [
    {
      role: 'readWrite',
      db: 'genesis_flow_test'
    }
  ]
});

// Create indexes for better performance (Genesis-Flow will also create these)
db.experiments.createIndex({ "name": 1 }, { unique: true });
db.experiments.createIndex({ "lifecycle_stage": 1 });
db.experiments.createIndex({ "creation_time": 1 });

db.runs.createIndex({ "experiment_id": 1 });
db.runs.createIndex({ "status": 1 });
db.runs.createIndex({ "start_time": 1 });
db.runs.createIndex({ "end_time": 1 });
db.runs.createIndex({ "user_id": 1 });

db.metrics.createIndex({ "run_uuid": 1, "key": 1 });
db.metrics.createIndex({ "timestamp": 1 });

db.params.createIndex({ "run_uuid": 1, "key": 1 });

db.tags.createIndex({ "run_uuid": 1, "key": 1 });

// Insert a welcome document
db.genesis_flow_info.insertOne({
  message: "Genesis-Flow MongoDB Integration Test Database",
  created_at: new Date(),
  version: "1.0.0",
  features: [
    "Direct MongoDB integration",
    "No MLflow server required",
    "Enhanced performance",
    "100% API compatibility"
  ]
});

print("Genesis-Flow MongoDB test database initialized successfully!");
print("Database: genesis_flow_test");
print("Test user: genesis_flow_user");
print("Collections initialized with indexes for optimal performance");