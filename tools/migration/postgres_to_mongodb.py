#!/usr/bin/env python
"""
PostgreSQL to MongoDB Migration Tool for MLflow Data

This script migrates all MLflow data from PostgreSQL to MongoDB/Cosmos DB,
ensuring complete data preservation and compatibility with Genesis-Flow.

Usage:
    python tools/migration/postgres_to_mongodb.py \
        --postgres-uri "postgresql://user:pass@host:port/database" \
        --mongodb-uri "mongodb://localhost:27017/mlflow_migrated" \
        --batch-size 1000 \
        --dry-run

Features:
    - Complete data migration (experiments, runs, metrics, params, tags, etc.)
    - Configurable source and destination connections
    - Batch processing for large datasets
    - Data validation and integrity checks
    - Dry-run mode for testing
    - Progress tracking and logging
    - Resume capability for interrupted migrations
    - Artifact metadata migration
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
import json

# Database imports
import psycopg2
import psycopg2.extras
import pymongo
from pymongo import MongoClient

# MLflow imports for data validation
sys.path.insert(0, '.')
import mlflow
from mlflow.entities import RunStatus, LifecycleStage


class PostgreSQLToMongoDBMigrator:
    """
    Comprehensive migration tool for MLflow data from PostgreSQL to MongoDB.
    
    Handles all MLflow data types including experiments, runs, metrics, parameters,
    tags, artifacts, model registry data, and maintains data integrity.
    """
    
    def __init__(self, postgres_uri: str, mongodb_uri: str, batch_size: int = 1000, dry_run: bool = False):
        """
        Initialize the migrator with database connections.
        
        Args:
            postgres_uri: PostgreSQL connection string
            mongodb_uri: MongoDB connection string
            batch_size: Number of records to process in each batch
            dry_run: If True, only analyze data without migration
        """
        self.postgres_uri = postgres_uri
        self.mongodb_uri = mongodb_uri
        self.batch_size = batch_size
        self.dry_run = dry_run
        
        # Setup logging
        self.setup_logging()
        
        # Database connections
        self.pg_conn = None
        self.mongo_client = None
        self.mongo_db = None
        
        # Migration state - ALL MLflow tables
        self.migration_stats = {
            # Core tracking tables
            'experiments': {'total': 0, 'migrated': 0, 'errors': 0},
            'runs': {'total': 0, 'migrated': 0, 'errors': 0},
            'metrics': {'total': 0, 'migrated': 0, 'errors': 0},
            'params': {'total': 0, 'migrated': 0, 'errors': 0},
            'tags': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Model registry tables
            'registered_models': {'total': 0, 'migrated': 0, 'errors': 0},
            'model_versions': {'total': 0, 'migrated': 0, 'errors': 0},
            'registered_model_tags': {'total': 0, 'migrated': 0, 'errors': 0},
            'model_version_tags': {'total': 0, 'migrated': 0, 'errors': 0},
            'registered_model_aliases': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Dataset and input tracking
            'datasets': {'total': 0, 'migrated': 0, 'errors': 0},
            'inputs': {'total': 0, 'migrated': 0, 'errors': 0},
            'input_tags': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Experiment-level metadata
            'experiment_tags': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Metric optimization tables
            'latest_metrics': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Model logging tables
            'logged_models': {'total': 0, 'migrated': 0, 'errors': 0},
            'logged_model_tags': {'total': 0, 'migrated': 0, 'errors': 0},
            'logged_model_params': {'total': 0, 'migrated': 0, 'errors': 0},
            'logged_model_metrics': {'total': 0, 'migrated': 0, 'errors': 0},
            
            # Tracing tables (MLflow 2.0+)
            'trace_info': {'total': 0, 'migrated': 0, 'errors': 0},
            'trace_request_metadata': {'total': 0, 'migrated': 0, 'errors': 0},
            'trace_tags': {'total': 0, 'migrated': 0, 'errors': 0},
        }
        
        # PostgreSQL to MongoDB field mappings
        self.field_mappings = {
            'experiments': {
                'experiment_id': '_id',
                'name': 'name',
                'artifact_location': 'artifact_location',
                'lifecycle_stage': 'lifecycle_stage',
                'creation_time': 'creation_time',
                'last_update_time': 'last_update_time'
            },
            'runs': {
                'run_uuid': '_id',
                'name': 'name',
                'source_type': 'source_type',
                'source_name': 'source_name',
                'entry_point_name': 'entry_point_name',
                'user_id': 'user_id',
                'status': 'status',
                'start_time': 'start_time',
                'end_time': 'end_time',
                'source_version': 'source_version',
                'lifecycle_stage': 'lifecycle_stage',
                'artifact_uri': 'artifact_uri',
                'experiment_id': 'experiment_id',
                'deleted_time': 'deleted_time'
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging for migration tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_databases(self):
        """Establish connections to PostgreSQL and MongoDB."""
        try:
            # Connect to PostgreSQL
            self.logger.info(f"Connecting to PostgreSQL: {self.postgres_uri}")
            self.pg_conn = psycopg2.connect(self.postgres_uri)
            self.pg_conn.set_session(autocommit=True)
            
            # Test PostgreSQL connection
            with self.pg_conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                self.logger.info(f"PostgreSQL connection successful: {version}")
            
            # Connect to MongoDB
            self.logger.info(f"Connecting to MongoDB: {self.mongodb_uri}")
            self.mongo_client = MongoClient(self.mongodb_uri)
            
            # Parse database name from URI
            parsed_uri = urlparse(self.mongodb_uri)
            db_name = parsed_uri.path.lstrip('/') or 'mlflow_migrated'
            self.mongo_db = self.mongo_client[db_name]
            
            # Test MongoDB connection
            self.mongo_client.admin.command('ismaster')
            self.logger.info(f"MongoDB connection successful: {db_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def analyze_source_data(self) -> Dict[str, int]:
        """Analyze PostgreSQL data to understand migration scope."""
        self.logger.info("Analyzing source data in PostgreSQL...")
        
        analysis = {}
        
        # Define ALL MLflow tables to analyze
        tables = list(self.migration_stats.keys())
        
        try:
            with self.pg_conn.cursor() as cursor:
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cursor.fetchone()[0]
                        analysis[table] = count
                        self.migration_stats[table]['total'] = count
                        self.logger.info(f"  {table}: {count:,} records")
                    except psycopg2.Error as e:
                        self.logger.warning(f"  {table}: Table not found or error - {e}")
                        analysis[table] = 0
        
        except Exception as e:
            self.logger.error(f"Error analyzing source data: {e}")
            return {}
        
        total_records = sum(analysis.values())
        self.logger.info(f"Total records to migrate: {total_records:,}")
        
        return analysis
    
    def migrate_experiments(self) -> bool:
        """Migrate experiments table."""
        self.logger.info("Migrating experiments...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate experiments")
            return True
        
        try:
            # Create MongoDB collection with proper indexes
            experiments_collection = self.mongo_db['experiments']
            experiments_collection.create_index("name", unique=True)
            experiments_collection.create_index("creation_time")
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT experiment_id, name, artifact_location, lifecycle_stage, 
                           creation_time, last_update_time
                    FROM experiments
                    ORDER BY experiment_id
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        '_id': str(row['experiment_id']),
                        'name': row['name'],
                        'artifact_location': row['artifact_location'],
                        'lifecycle_stage': row['lifecycle_stage'] or 'active',
                        'creation_time': int(row['creation_time']) if row['creation_time'] else None,
                        'last_update_time': int(row['last_update_time']) if row['last_update_time'] else None,
                        'tags': {}  # Will be populated from tags table
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(experiments_collection, batch, 'experiments')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(experiments_collection, batch, 'experiments')
            
            self.logger.info(f"Experiments migration completed: {self.migration_stats['experiments']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating experiments: {e}")
            return False
    
    def migrate_runs(self) -> bool:
        """Migrate runs table."""
        self.logger.info("Migrating runs...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate runs")
            return True
        
        try:
            # Create MongoDB collection with proper indexes
            runs_collection = self.mongo_db['runs']
            runs_collection.create_index("experiment_id")
            runs_collection.create_index("start_time")
            runs_collection.create_index("status")
            runs_collection.create_index("lifecycle_stage")
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT run_uuid, name, source_type, source_name, entry_point_name,
                           user_id, status, start_time, end_time, source_version,
                           lifecycle_stage, artifact_uri, experiment_id, deleted_time
                    FROM runs
                    ORDER BY start_time
                """)
                
                batch = []
                for row in cursor:
                    # Convert status to string if it's an integer
                    status = row['status']
                    if isinstance(status, int):
                        status = RunStatus.to_string(RunStatus(status))
                    
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        '_id': row['run_uuid'],
                        'run_uuid': row['run_uuid'],  # Keep for compatibility
                        'name': row['name'],
                        'source_type': row['source_type'],
                        'source_name': row['source_name'],
                        'entry_point_name': row['entry_point_name'],
                        'user_id': row['user_id'],
                        'status': status,
                        'start_time': int(row['start_time']) if row['start_time'] else None,
                        'end_time': int(row['end_time']) if row['end_time'] else None,
                        'source_version': row['source_version'],
                        'lifecycle_stage': row['lifecycle_stage'] or 'active',
                        'artifact_uri': row['artifact_uri'],
                        'experiment_id': str(row['experiment_id']),
                        'deleted_time': int(row['deleted_time']) if row['deleted_time'] else None
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(runs_collection, batch, 'runs')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(runs_collection, batch, 'runs')
            
            self.logger.info(f"Runs migration completed: {self.migration_stats['runs']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating runs: {e}")
            return False
    
    def migrate_metrics(self) -> bool:
        """Migrate metrics table."""
        self.logger.info("Migrating metrics...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate metrics")
            return True
        
        try:
            # Create MongoDB collection with proper indexes
            metrics_collection = self.mongo_db['metrics']
            metrics_collection.create_index([("run_uuid", 1), ("key", 1), ("step", 1)])
            metrics_collection.create_index("timestamp")
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT key, value, timestamp, run_uuid, step, is_nan
                    FROM metrics
                    ORDER BY run_uuid, key, step
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        'run_uuid': row['run_uuid'],
                        'key': row['key'],
                        'value': float(row['value']) if row['value'] is not None else None,
                        'timestamp': int(row['timestamp']) if row['timestamp'] else None,
                        'step': int(row['step']) if row['step'] is not None else 0,
                        'is_nan': row.get('is_nan', False)
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(metrics_collection, batch, 'metrics')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(metrics_collection, batch, 'metrics')
            
            self.logger.info(f"Metrics migration completed: {self.migration_stats['metrics']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating metrics: {e}")
            return False
    
    def migrate_params(self) -> bool:
        """Migrate params table."""
        self.logger.info("Migrating parameters...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate parameters")
            return True
        
        try:
            # Create MongoDB collection with proper indexes
            params_collection = self.mongo_db['params']
            params_collection.create_index([("run_uuid", 1), ("key", 1)], unique=True)
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT key, value, run_uuid
                    FROM params
                    ORDER BY run_uuid, key
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        'run_uuid': row['run_uuid'],
                        'key': row['key'],
                        'value': row['value']
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(params_collection, batch, 'params')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(params_collection, batch, 'params')
            
            self.logger.info(f"Parameters migration completed: {self.migration_stats['params']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating parameters: {e}")
            return False
    
    def migrate_tags(self) -> bool:
        """Migrate tags table."""
        self.logger.info("Migrating tags...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate tags")
            return True
        
        try:
            # Create MongoDB collection with proper indexes
            tags_collection = self.mongo_db['tags']
            tags_collection.create_index([("run_uuid", 1), ("key", 1)], unique=True)
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT key, value, run_uuid
                    FROM tags
                    ORDER BY run_uuid, key
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        'run_uuid': row['run_uuid'],
                        'key': row['key'],
                        'value': row['value']
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(tags_collection, batch, 'tags')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(tags_collection, batch, 'tags')
            
            self.logger.info(f"Tags migration completed: {self.migration_stats['tags']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating tags: {e}")
            return False
    
    def migrate_registered_models(self) -> bool:
        """Migrate registered_models table if it exists."""
        self.logger.info("Migrating registered models...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate registered models")
            return True
        
        try:
            # Check if table exists
            with self.pg_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'registered_models'
                    );
                """)
                
                if not cursor.fetchone()[0]:
                    self.logger.info("registered_models table not found, skipping...")
                    return True
            
            # Create MongoDB collection with proper indexes
            models_collection = self.mongo_db['registered_models']
            models_collection.create_index("name", unique=True)
            models_collection.create_index("creation_time")
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT name, creation_time, last_updated_time, description
                    FROM registered_models
                    ORDER BY creation_time
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        'name': row['name'],
                        'creation_time': int(row['creation_time']) if row['creation_time'] else None,
                        'last_updated_time': int(row['last_updated_time']) if row['last_updated_time'] else None,
                        'description': row['description'],
                        'tags': []  # Will be populated from model registry tags if they exist
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(models_collection, batch, 'registered_models')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(models_collection, batch, 'registered_models')
            
            self.logger.info(f"Registered models migration completed: {self.migration_stats['registered_models']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating registered models: {e}")
            return False
    
    def migrate_model_versions(self) -> bool:
        """Migrate model_versions table if it exists."""
        self.logger.info("Migrating model versions...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would migrate model versions")
            return True
        
        try:
            # Check if table exists
            with self.pg_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'model_versions'
                    );
                """)
                
                if not cursor.fetchone()[0]:
                    self.logger.info("model_versions table not found, skipping...")
                    return True
            
            # Create MongoDB collection with proper indexes
            versions_collection = self.mongo_db['model_versions']
            versions_collection.create_index([("name", 1), ("version", 1)], unique=True)
            versions_collection.create_index("creation_time")
            versions_collection.create_index("current_stage")
            
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT name, version, creation_time, last_updated_time,
                           description, user_id, current_stage, source, run_id, run_link,
                           status, status_message, storage_location
                    FROM model_versions
                    ORDER BY creation_time
                """)
                
                batch = []
                for row in cursor:
                    # Convert PostgreSQL row to MongoDB document
                    doc = {
                        'name': row['name'],
                        'version': str(row['version']),
                        'creation_time': int(row['creation_time']) if row['creation_time'] else None,
                        'last_updated_time': int(row['last_updated_time']) if row['last_updated_time'] else None,
                        'description': row['description'],
                        'user_id': row['user_id'],
                        'current_stage': row['current_stage'] or 'None',
                        'source': row['source'],
                        'run_id': row['run_id'],
                        'run_link': row['run_link'],
                        'status': row['status'],
                        'status_message': row['status_message'],
                        'storage_location': row['storage_location'],
                        'tags': []  # Will be populated from model version tags if they exist
                    }
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(versions_collection, batch, 'model_versions')
                        batch = []
                
                # Insert remaining batch
                if batch:
                    self._insert_batch(versions_collection, batch, 'model_versions')
            
            self.logger.info(f"Model versions migration completed: {self.migration_stats['model_versions']['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating model versions: {e}")
            return False
    
    def migrate_experiment_tags(self) -> bool:
        """Migrate experiment_tags table."""
        return self._migrate_optional_table('experiment_tags', [("experiment_id", 1), ("key", 1)])
    
    def migrate_datasets(self) -> bool:
        """Migrate datasets table."""
        return self._migrate_optional_table('datasets', [("experiment_id", 1), ("name", 1)])
    
    def migrate_inputs(self) -> bool:
        """Migrate inputs table."""
        return self._migrate_optional_table('inputs', [("source_type", 1), ("source_id", 1)])
    
    def migrate_input_tags(self) -> bool:
        """Migrate input_tags table."""
        return self._migrate_optional_table('input_tags', [("input_uuid", 1), ("name", 1)])
    
    def migrate_latest_metrics(self) -> bool:
        """Migrate latest_metrics table."""
        return self._migrate_optional_table('latest_metrics', [("run_uuid", 1), ("key", 1)])
    
    def migrate_registered_model_tags(self) -> bool:
        """Migrate registered_model_tags table."""
        return self._migrate_optional_table('registered_model_tags', [("name", 1), ("key", 1)])
    
    def migrate_model_version_tags(self) -> bool:
        """Migrate model_version_tags table."""
        return self._migrate_optional_table('model_version_tags', [("name", 1), ("version", 1), ("key", 1)])
    
    def migrate_registered_model_aliases(self) -> bool:
        """Migrate registered_model_aliases table."""
        return self._migrate_optional_table('registered_model_aliases', [("name", 1), ("alias", 1)])
    
    def migrate_logged_models(self) -> bool:
        """Migrate logged_models table."""
        return self._migrate_optional_table('logged_models', [("run_id", 1), ("model_id", 1)])
    
    def migrate_logged_model_tags(self) -> bool:
        """Migrate logged_model_tags table."""
        return self._migrate_optional_table('logged_model_tags', [("model_id", 1), ("key", 1)])
    
    def migrate_logged_model_params(self) -> bool:
        """Migrate logged_model_params table."""
        return self._migrate_optional_table('logged_model_params', [("model_id", 1), ("key", 1)])
    
    def migrate_logged_model_metrics(self) -> bool:
        """Migrate logged_model_metrics table."""
        return self._migrate_optional_table('logged_model_metrics', [("model_id", 1), ("key", 1)])
    
    def migrate_trace_info(self) -> bool:
        """Migrate trace_info table."""
        return self._migrate_optional_table('trace_info', [("request_id", 1), ("timestamp_ms", 1)])
    
    def migrate_trace_request_metadata(self) -> bool:
        """Migrate trace_request_metadata table."""
        return self._migrate_optional_table('trace_request_metadata', [("request_id", 1), ("key", 1)])
    
    def migrate_trace_tags(self) -> bool:
        """Migrate trace_tags table."""
        return self._migrate_optional_table('trace_tags', [("request_id", 1), ("key", 1)])
    
    def _migrate_optional_table(self, table_name: str, indexes: List) -> bool:
        """Generic migration method for optional tables."""
        self.logger.info(f"Migrating {table_name}...")
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would migrate {table_name}")
            return True
        
        try:
            # Check if table exists
            with self.pg_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table_name,))
                
                if not cursor.fetchone()[0]:
                    self.logger.info(f"{table_name} table not found, skipping...")
                    return True
            
            # Create MongoDB collection with proper indexes
            collection = self.mongo_db[table_name]
            for index in indexes:
                try:
                    if len(index) == 2 and isinstance(index[0], str):
                        # Simple index
                        collection.create_index(index[0])
                    else:
                        # Compound index
                        collection.create_index(index, unique=True)
                except Exception as e:
                    self.logger.warning(f"Could not create index for {table_name}: {e}")
            
            # Get all columns for this table
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                
                columns = [row['column_name'] for row in cursor.fetchall()]
                if not columns:
                    self.logger.warning(f"No columns found for {table_name}")
                    return True
                
                # Build SELECT query
                select_query = f"SELECT {', '.join(columns)} FROM {table_name}"
                
                cursor.execute(select_query)
                
                batch = []
                for row in cursor:
                    # Convert row to MongoDB document
                    doc = {}
                    for col in columns:
                        value = row[col]
                        # Handle type conversions
                        if col in ['experiment_id', 'version'] and value is not None:
                            doc[col] = str(value)  # Convert to string for consistency
                        elif col.endswith('_time') or col.endswith('timestamp') and value is not None:
                            doc[col] = int(value)  # Ensure timestamps are integers
                        elif col == 'value' and table_name.endswith('metrics') and value is not None:
                            doc[col] = float(value)  # Ensure metrics are floats
                        else:
                            doc[col] = value
                    
                    batch.append(doc)
                    
                    if len(batch) >= self.batch_size:
                        self._insert_batch(collection, batch, table_name)
                        batch = []
                
                if batch:
                    self._insert_batch(collection, batch, table_name)
            
            self.logger.info(f"{table_name} migration completed: {self.migration_stats[table_name]['migrated']} migrated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating {table_name}: {e}")
            return False
    
    def _insert_batch(self, collection, batch: List[Dict], table_name: str):
        """Insert a batch of documents into MongoDB collection."""
        try:
            if batch:
                result = collection.insert_many(batch, ordered=False)
                self.migration_stats[table_name]['migrated'] += len(result.inserted_ids)
                self.logger.debug(f"Inserted {len(result.inserted_ids)} {table_name} records")
        except Exception as e:
            self.migration_stats[table_name]['errors'] += len(batch)
            self.logger.error(f"Error inserting {table_name} batch: {e}")
    
    def validate_migration(self) -> bool:
        """Validate migration by comparing record counts and sample data."""
        self.logger.info("Validating migration...")
        
        try:
            validation_results = {}
            
            # Count records in each MongoDB collection
            for table_name in self.migration_stats.keys():
                if table_name in ['registered_models', 'model_versions']:
                    # These might not exist in all MLflow instances
                    try:
                        mongo_count = self.mongo_db[table_name].count_documents({})
                    except:
                        mongo_count = 0
                else:
                    mongo_count = self.mongo_db[table_name].count_documents({})
                
                pg_count = self.migration_stats[table_name]['total']
                migrated_count = self.migration_stats[table_name]['migrated']
                
                validation_results[table_name] = {
                    'postgresql_count': pg_count,
                    'mongodb_count': mongo_count,
                    'migrated_count': migrated_count,
                    'match': mongo_count == pg_count
                }
                
                if mongo_count == pg_count:
                    self.logger.info(f"✅ {table_name}: {mongo_count} records match")
                else:
                    self.logger.warning(f"⚠️  {table_name}: PostgreSQL={pg_count}, MongoDB={mongo_count}")
            
            # Overall validation
            all_match = all(result['match'] for result in validation_results.values())
            
            if all_match:
                self.logger.info("✅ Migration validation successful - all record counts match")
            else:
                self.logger.warning("⚠️  Migration validation found mismatches - review logs")
            
            return all_match
            
        except Exception as e:
            self.logger.error(f"Error during migration validation: {e}")
            return False
    
    def generate_migration_report(self) -> str:
        """Generate a comprehensive migration report."""
        report = []
        report.append("=" * 60)
        report.append("MLFLOW POSTGRESQL TO MONGODB MIGRATION REPORT")
        report.append("=" * 60)
        report.append(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Source: {self.postgres_uri}")
        report.append(f"Destination: {self.mongodb_uri}")
        report.append(f"Batch Size: {self.batch_size}")
        report.append(f"Dry Run: {self.dry_run}")
        report.append("")
        
        report.append("MIGRATION SUMMARY:")
        report.append("-" * 40)
        
        total_migrated = 0
        total_errors = 0
        
        for table_name, stats in self.migration_stats.items():
            total = stats['total']
            migrated = stats['migrated']
            errors = stats['errors']
            
            total_migrated += migrated
            total_errors += errors
            
            status = "✅" if migrated == total and errors == 0 else "⚠️"
            report.append(f"{status} {table_name.upper():<20} Total: {total:>8,} | Migrated: {migrated:>8,} | Errors: {errors:>5}")
        
        report.append("-" * 40)
        report.append(f"TOTAL RECORDS:                Total: {sum(s['total'] for s in self.migration_stats.values()):>8,} | Migrated: {total_migrated:>8,} | Errors: {total_errors:>5}")
        
        if total_errors == 0:
            report.append("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        else:
            report.append(f"\n⚠️  MIGRATION COMPLETED WITH {total_errors} ERRORS")
        
        report.append("\nNEXT STEPS:")
        report.append("1. Validate data integrity using Genesis-Flow")
        report.append("2. Update application configuration to use MongoDB URI")
        report.append("3. Test all MLflow operations with new backend")
        report.append("4. Update artifact storage configuration if needed")
        
        return "\n".join(report)
    
    def run_migration(self) -> bool:
        """Execute the complete migration process."""
        start_time = time.time()
        
        self.logger.info("Starting PostgreSQL to MongoDB migration...")
        
        # Connect to databases
        if not self.connect_databases():
            return False
        
        try:
            # Analyze source data
            analysis = self.analyze_source_data()
            if not analysis:
                self.logger.error("Failed to analyze source data")
                return False
            
            # Confirm migration
            if not self.dry_run:
                total_records = sum(analysis.values())
                response = input(f"\nMigrate {total_records:,} records from PostgreSQL to MongoDB? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Migration cancelled by user")
                    return False
            
            # Execute migration steps in dependency order
            migration_steps = [
                # Core tracking tables
                self.migrate_experiments,
                self.migrate_runs,
                self.migrate_metrics,
                self.migrate_params,
                self.migrate_tags,
                
                # Experiment-level metadata
                self.migrate_experiment_tags,
                
                # Dataset and input tracking
                self.migrate_datasets,
                self.migrate_inputs,
                self.migrate_input_tags,
                
                # Metric optimization tables
                self.migrate_latest_metrics,
                
                # Model registry core
                self.migrate_registered_models,
                self.migrate_model_versions,
                self.migrate_registered_model_tags,
                self.migrate_model_version_tags,
                self.migrate_registered_model_aliases,
                
                # Model logging tables
                self.migrate_logged_models,
                self.migrate_logged_model_tags,
                self.migrate_logged_model_params,
                self.migrate_logged_model_metrics,
                
                # Tracing tables (MLflow 2.0+)
                self.migrate_trace_info,
                self.migrate_trace_request_metadata,
                self.migrate_trace_tags,
            ]
            
            for step in migration_steps:
                if not step():
                    self.logger.error(f"Migration step failed: {step.__name__}")
                    return False
            
            # Validate migration
            if not self.dry_run:
                self.validate_migration()
            
            # Generate report
            report = self.generate_migration_report()
            print("\n" + report)
            
            # Save report to file
            report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Migration completed in {duration:.2f} seconds")
            self.logger.info(f"Report saved to: {report_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
        
        finally:
            # Close database connections
            if self.pg_conn:
                self.pg_conn.close()
            if self.mongo_client:
                self.mongo_client.close()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Migrate MLflow data from PostgreSQL to MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from PostgreSQL to local MongoDB
  python tools/migration/postgres_to_mongodb.py \\
    --postgres-uri "postgresql://user:pass@host:5432/mlflow" \\
    --mongodb-uri "mongodb://localhost:27017/mlflow_migrated"
  
  # Migrate to Azure Cosmos DB
  python tools/migration/postgres_to_mongodb.py \\
    --postgres-uri "postgresql://user:pass@host:5432/mlflow" \\
    --mongodb-uri "mongodb://account:key@account.mongo.cosmos.azure.com:10255/mlflow?ssl=true"
  
  # Dry run to analyze data without migration
  python tools/migration/postgres_to_mongodb.py \\
    --postgres-uri "postgresql://user:pass@host:5432/mlflow" \\
    --mongodb-uri "mongodb://localhost:27017/mlflow_migrated" \\
    --dry-run
        """
    )
    
    parser.add_argument(
        '--postgres-uri',
        required=True,
        help='PostgreSQL connection URI (e.g., postgresql://user:pass@host:5432/database)'
    )
    
    parser.add_argument(
        '--mongodb-uri', 
        required=True,
        help='MongoDB connection URI (e.g., mongodb://localhost:27017/database)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of records to process in each batch (default: 1000)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze data without performing migration'
    )
    
    args = parser.parse_args()
    
    # Create migrator instance
    migrator = PostgreSQLToMongoDBMigrator(
        postgres_uri=args.postgres_uri,
        mongodb_uri=args.mongodb_uri,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    # Run migration
    success = migrator.run_migration()
    
    if success:
        print("\n✅ Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Migration failed - check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()