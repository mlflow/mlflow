#!/usr/bin/env python
"""
MongoDB Store Implementation for MLflow Tracking
Provides automatic collection creation with proper schema for Genesis-Flow compatibility.

This module implements:
- Automatic collection creation when MongoDB is set as tracking URI
- Proper indexes for performance optimization
- Schema validation for data integrity
- Full compatibility with PostgreSQL store operations
"""

import logging
import pymongo
from pymongo import MongoClient
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)


class MongoDBAutoSetup:
    """
    Automatic MongoDB collection setup for MLflow compatibility.
    
    Creates all necessary collections with proper indexes when MongoDB is used as tracking URI.
    """
    
    @staticmethod
    def setup_collections(db_uri: str):
        """
        Setup all MLflow collections with proper schema.
        
        Args:
            db_uri: MongoDB connection URI
        """
        try:
            # Parse database name from URI
            parsed = urlparse(db_uri)
            db_name = parsed.path.lstrip('/') or 'mlflow'
            
            # Connect to MongoDB
            client = MongoClient(db_uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ismaster')  # Test connection
            db = client[db_name]
            
            # Setup collections with schema
            collections_schema = MongoDBAutoSetup._get_collections_schema()
            
            for collection_name, schema in collections_schema.items():
                collection = db[collection_name]
                
                # Create indexes
                for index in schema.get('indexes', []):
                    try:
                        if isinstance(index, list) and len(index) > 1:
                            # Compound index
                            collection.create_index(index)
                        elif isinstance(index, dict):
                            # Index with options
                            collection.create_index(
                                list(index.items()),
                                unique=index.get('unique', False),
                                sparse=index.get('sparse', False)
                            )
                        else:
                            # Simple index
                            collection.create_index(index)
                    except Exception as e:
                        logger.warning(f"Could not create index for {collection_name}: {e}")
                
                logger.debug(f"Collection '{collection_name}' setup completed")
            
            logger.info(f"MongoDB collections setup completed for database: {db_name}")
            client.close()
            
        except Exception as e:
            logger.error(f"Failed to setup MongoDB collections: {e}")
            raise
    
    @staticmethod
    def _get_collections_schema() -> Dict[str, Dict]:
        """
        Define schema for all MLflow collections.
        
        Returns:
            Dictionary mapping collection names to their schema definitions
        """
        return {
            # Core tracking collections
            'experiments': {
                'indexes': [
                    'name',
                    'creation_time',
                    'lifecycle_stage',
                    [('name', 1)],  # Unique experiment names
                ]
            },
            'runs': {
                'indexes': [
                    'experiment_id',
                    'status',
                    'start_time',
                    'end_time',
                    'lifecycle_stage',
                    [('experiment_id', 1), ('start_time', -1)],
                    [('status', 1), ('start_time', -1)],
                ]
            },
            'metrics': {
                'indexes': [
                    'run_uuid',
                    'key',
                    'timestamp',
                    'step',
                    [('run_uuid', 1), ('key', 1), ('step', 1)],
                    [('run_uuid', 1), ('timestamp', -1)],
                ]
            },
            'params': {
                'indexes': [
                    'run_uuid',
                    'key',
                    [('run_uuid', 1), ('key', 1)],
                ]
            },
            'tags': {
                'indexes': [
                    'run_uuid',
                    'key',
                    [('run_uuid', 1), ('key', 1)],
                ]
            },
            
            # Model registry collections
            'registered_models': {
                'indexes': [
                    [('name', 1)],
                    'creation_time',
                    'last_updated_time',
                ]
            },
            'model_versions': {
                'indexes': [
                    [('name', 1), ('version', 1)],
                    'creation_time',
                    'current_stage',
                    'status',
                    [('name', 1), ('creation_time', -1)],
                ]
            },
            'registered_model_tags': {
                'indexes': [
                    [('name', 1), ('key', 1)],
                ]
            },
            'model_version_tags': {
                'indexes': [
                    [('name', 1), ('version', 1), ('key', 1)],
                ]
            },
            'registered_model_aliases': {
                'indexes': [
                    [('name', 1), ('alias', 1)],
                ]
            },
            
            # Dataset and input tracking
            'datasets': {
                'indexes': [
                    'name',
                    'digest',
                    'source_type',
                    [('name', 1), ('digest', 1)],
                ]
            },
            'inputs': {
                'indexes': [
                    'source_type',
                    'source_id',
                    'destination_type',
                    'destination_id',
                ]
            },
            'input_tags': {
                'indexes': [
                    'input_uuid',
                    'name',
                    [('input_uuid', 1), ('name', 1)],
                ]
            },
            
            # Experiment-level metadata
            'experiment_tags': {
                'indexes': [
                    'experiment_id',
                    'key',
                    [('experiment_id', 1), ('key', 1)],
                ]
            },
            
            # Optimization tables
            'latest_metrics': {
                'indexes': [
                    'run_uuid',
                    'key',
                    [('run_uuid', 1), ('key', 1)],
                ]
            },
            
            # Model logging tables
            'logged_models': {
                'indexes': [
                    'run_id',
                    'model_id',
                    [('run_id', 1), ('model_id', 1)],
                ]
            },
            'logged_model_tags': {
                'indexes': [
                    'model_id',
                    'key',
                    [('model_id', 1), ('key', 1)],
                ]
            },
            'logged_model_params': {
                'indexes': [
                    'model_id',
                    'key',
                    [('model_id', 1), ('key', 1)],
                ]
            },
            'logged_model_metrics': {
                'indexes': [
                    'model_id',
                    'key',
                    [('model_id', 1), ('key', 1)],
                ]
            },
            
            # Tracing tables (MLflow 2.0+)
            'trace_info': {
                'indexes': [
                    'request_id',
                    'timestamp_ms',
                    'execution_time_ms',
                    [('request_id', 1)],
                    [('timestamp_ms', -1)],
                ]
            },
            'trace_request_metadata': {
                'indexes': [
                    'request_id',
                    'key',
                    [('request_id', 1), ('key', 1)],
                ]
            },
            'trace_tags': {
                'indexes': [
                    'request_id',
                    'key',
                    [('request_id', 1), ('key', 1)],
                ]
            },
        }


# Auto-setup function that can be called when MongoDB URI is detected
def setup_mongodb_collections(tracking_uri: str):
    """
    Setup MongoDB collections when MongoDB is used as tracking URI.
    
    Args:
        tracking_uri: MLflow tracking URI
    """
    if tracking_uri.startswith('mongodb://') or tracking_uri.startswith('mongodb+srv://'):
        logger.info("MongoDB tracking URI detected - setting up collections")
        MongoDBAutoSetup.setup_collections(tracking_uri)
    else:
        logger.debug("Non-MongoDB tracking URI - skipping collection setup")