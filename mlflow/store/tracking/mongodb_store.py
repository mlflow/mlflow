"""
Genesis-Flow MongoDB Tracking Store

This module implements a MongoDB-based tracking store for Genesis-Flow,
designed to integrate with Azure Cosmos DB and Azure Blob Storage for artifacts.

Architecture:
- MongoDB/Cosmos DB: Experiments, runs, parameters, metrics, tags metadata
- Azure Blob Storage: Model artifacts, files, logs, notebooks
- Motor: Async MongoDB driver for performance
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import motor.motor_asyncio
import pymongo
from pymongo import IndexModel, ASCENDING, DESCENDING

from mlflow.entities import (
    Experiment,
    ExperimentTag,
    FileInfo,
    LifecycleStage,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.uri import extract_and_normalize_path
from mlflow.utils.time import get_current_time_millis

logger = logging.getLogger(__name__)

class MongoDBStore(AbstractStore):
    """
    MongoDB/Cosmos DB implementation of MLflow tracking store.
    
    Stores experiment metadata in MongoDB while delegating artifact storage
    to cloud storage backends (Azure Blob Storage, S3, etc.).
    """
    
    # Collection names
    EXPERIMENTS_COLLECTION = "experiments"
    RUNS_COLLECTION = "runs"
    PARAMS_COLLECTION = "params"
    METRICS_COLLECTION = "metrics"
    TAGS_COLLECTION = "tags"
    
    def __init__(self, db_uri: str, default_artifact_root: Optional[str] = None):
        """
        Initialize MongoDB tracking store.
        
        Args:
            db_uri: MongoDB connection string (mongodb:// or Azure Cosmos DB connection string)
            default_artifact_root: Default location for artifacts (Azure Blob Storage URI)
        """
        super().__init__()
        
        self.db_uri = db_uri
        self.default_artifact_root = default_artifact_root or "azure://artifacts"
        
        # Parse connection URI
        parsed_uri = urlparse(db_uri)
        self.database_name = parsed_uri.path.lstrip('/') or "genesis_flow"
        
        # Initialize MongoDB client
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                db_uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                retryWrites=True,
                w='majority',  # Write concern for data consistency
                readPreference='primaryPreferred'
            )
            self.db = self.client[self.database_name]
            
            # Initialize collections with proper indexing
            self._initialize_collections()
            
            logger.info(f"MongoDB store initialized with database: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB store: {e}")
            raise MlflowException(f"MongoDB connection failed: {e}")
    
    def _initialize_collections(self):
        """Initialize MongoDB collections with proper indexes for performance."""
        try:
            # Experiments collection indexes
            experiments_indexes = [
                IndexModel([("experiment_id", ASCENDING)], unique=True),
                IndexModel([("name", ASCENDING)], unique=True),
                IndexModel([("lifecycle_stage", ASCENDING)]),
                IndexModel([("creation_time", DESCENDING)]),
                IndexModel([("last_update_time", DESCENDING)]),
            ]
            
            # Runs collection indexes  
            runs_indexes = [
                IndexModel([("run_uuid", ASCENDING)], unique=True),
                IndexModel([("experiment_id", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
                IndexModel([("start_time", DESCENDING)]),
                IndexModel([("end_time", DESCENDING)]),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("lifecycle_stage", ASCENDING)]),
                # Compound indexes for common queries
                IndexModel([("experiment_id", ASCENDING), ("status", ASCENDING)]),
                IndexModel([("experiment_id", ASCENDING), ("start_time", DESCENDING)]),
            ]
            
            # Parameters collection indexes
            params_indexes = [
                IndexModel([("run_uuid", ASCENDING), ("key", ASCENDING)], unique=True),
                IndexModel([("run_uuid", ASCENDING)]),
                IndexModel([("key", ASCENDING)]),
            ]
            
            # Metrics collection indexes
            metrics_indexes = [
                IndexModel([("run_uuid", ASCENDING), ("key", ASCENDING), ("timestamp", ASCENDING)]),
                IndexModel([("run_uuid", ASCENDING)]),
                IndexModel([("key", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                # Compound index for metric history queries
                IndexModel([("run_uuid", ASCENDING), ("key", ASCENDING), ("step", ASCENDING)]),
            ]
            
            # Tags collection indexes
            tags_indexes = [
                IndexModel([("run_uuid", ASCENDING), ("key", ASCENDING)], unique=True),
                IndexModel([("run_uuid", ASCENDING)]),
                IndexModel([("key", ASCENDING)]),
            ]
            
            # Create indexes (motor will handle this asynchronously in background)
            self.db[self.EXPERIMENTS_COLLECTION].create_indexes(experiments_indexes)
            self.db[self.RUNS_COLLECTION].create_indexes(runs_indexes)
            self.db[self.PARAMS_COLLECTION].create_indexes(params_indexes)
            self.db[self.METRICS_COLLECTION].create_indexes(metrics_indexes)
            self.db[self.TAGS_COLLECTION].create_indexes(tags_indexes)
            
            logger.info("MongoDB collections and indexes initialized")
            
        except Exception as e:
            logger.warning(f"Failed to create MongoDB indexes: {e}")
            # Continue without indexes - they're for performance, not functionality
    
    async def _get_experiment_by_id(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment document by ID."""
        return await self.db[self.EXPERIMENTS_COLLECTION].find_one(
            {"experiment_id": experiment_id}
        )
    
    async def _get_experiment_by_name(self, name: str) -> Optional[Dict]:
        """Get experiment document by name."""
        return await self.db[self.EXPERIMENTS_COLLECTION].find_one(
            {"name": name, "lifecycle_stage": {"$ne": LifecycleStage.DELETED}}
        )
    
    async def _get_run_by_uuid(self, run_uuid: str) -> Optional[Dict]:
        """Get run document by UUID."""
        return await self.db[self.RUNS_COLLECTION].find_one(
            {"run_uuid": run_uuid}
        )
    
    def _experiment_doc_to_entity(self, doc: Dict) -> Experiment:
        """Convert MongoDB document to Experiment entity."""
        tags = [ExperimentTag(tag["key"], tag["value"]) for tag in doc.get("tags", [])]
        
        return Experiment(
            experiment_id=doc["experiment_id"],
            name=doc["name"],
            artifact_location=doc["artifact_location"],
            lifecycle_stage=doc["lifecycle_stage"],
            tags=tags,
            creation_time=doc.get("creation_time"),
            last_update_time=doc.get("last_update_time"),
        )
    
    def _run_doc_to_entity(self, doc: Dict) -> Run:
        """Convert MongoDB document to Run entity."""
        run_info = RunInfo(
            run_uuid=doc["run_uuid"],
            run_id=doc["run_uuid"],  # MLflow compatibility
            experiment_id=doc["experiment_id"],
            user_id=doc["user_id"],
            status=RunStatus.from_string(doc["status"]),
            start_time=doc["start_time"],
            end_time=doc.get("end_time"),
            artifact_uri=doc["artifact_uri"],
            lifecycle_stage=doc["lifecycle_stage"],
        )
        
        run_data = RunData(
            metrics=[],  # Loaded separately for performance
            params=[],   # Loaded separately for performance
            tags=[],     # Loaded separately for performance
        )
        
        return Run(run_info=run_info, run_data=run_data)
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        return str(int(time.time() * 1000000))
    
    def _generate_run_uuid(self) -> str:
        """Generate unique run UUID."""
        import uuid
        return uuid.uuid4().hex
    
    # Abstract method implementations
    
    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=1000,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        """Search experiments with filtering and pagination."""
        # Note: This is a sync method that wraps async implementation
        import asyncio
        try:
            return asyncio.run(self._search_experiments_async(
                view_type, max_results, filter_string, order_by, page_token
            ))
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            raise MlflowException(f"MongoDB search experiments failed: {e}")
    
    async def _search_experiments_async(
        self, view_type, max_results, filter_string, order_by, page_token
    ):
        """Async implementation of search experiments."""
        # Build MongoDB query
        query = {}
        
        # Handle view type filter
        if view_type == ViewType.ACTIVE_ONLY:
            query["lifecycle_stage"] = LifecycleStage.ACTIVE
        elif view_type == ViewType.DELETED_ONLY:
            query["lifecycle_stage"] = LifecycleStage.DELETED
        # ALL includes both active and deleted
        
        # Handle pagination
        skip = 0
        if page_token:
            try:
                skip = int(page_token)
            except ValueError:
                raise MlflowException(
                    f"Invalid page token: {page_token}",
                    error_code=INVALID_PARAMETER_VALUE
                )
        
        # Handle ordering
        sort_spec = [("creation_time", DESCENDING)]  # Default sort
        if order_by:
            # Parse order_by string (e.g., "name ASC", "creation_time DESC")
            sort_spec = []
            for order_clause in order_by:
                parts = order_clause.strip().split()
                if len(parts) >= 2:
                    field, direction = parts[0], parts[1].upper()
                    mongo_direction = ASCENDING if direction == "ASC" else DESCENDING
                    sort_spec.append((field, mongo_direction))
        
        # Execute query
        cursor = self.db[self.EXPERIMENTS_COLLECTION].find(query).sort(sort_spec)
        
        # Get total count for pagination
        total_count = await self.db[self.EXPERIMENTS_COLLECTION].count_documents(query)
        
        # Apply pagination
        experiments_docs = await cursor.skip(skip).limit(max_results).to_list(max_results)
        
        # Convert to entities
        experiments = [self._experiment_doc_to_entity(doc) for doc in experiments_docs]
        
        # Determine next page token
        next_page_token = None
        if skip + len(experiments) < total_count:
            next_page_token = str(skip + max_results)
        
        return PagedList(experiments, next_page_token)
    
    def create_experiment(self, name: str, artifact_location: Optional[str] = None, tags: Optional[List[ExperimentTag]] = None) -> str:
        """Create a new experiment."""
        import asyncio
        try:
            return asyncio.run(self._create_experiment_async(name, artifact_location, tags))
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise MlflowException(f"MongoDB create experiment failed: {e}")
    
    async def _create_experiment_async(self, name: str, artifact_location: Optional[str], tags: Optional[List[ExperimentTag]]) -> str:
        """Async implementation of create experiment."""
        # Check if experiment already exists
        existing = await self._get_experiment_by_name(name)
        if existing:
            raise MlflowException(
                f"Experiment '{name}' already exists.",
                error_code=RESOURCE_ALREADY_EXISTS
            )
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id()
        
        # Set artifact location (default to Azure Blob Storage)
        if not artifact_location:
            artifact_location = f"{self.default_artifact_root}/{experiment_id}"
        
        # Create experiment document
        current_time = get_current_time_millis()
        experiment_doc = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "creation_time": current_time,
            "last_update_time": current_time,
            "tags": [{"key": tag.key, "value": tag.value} for tag in (tags or [])],
        }
        
        # Insert into MongoDB
        await self.db[self.EXPERIMENTS_COLLECTION].insert_one(experiment_doc)
        
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID."""
        import asyncio
        try:
            return asyncio.run(self._get_experiment_async(experiment_id))
        except Exception as e:
            logger.error(f"Failed to get experiment: {e}")
            raise MlflowException(f"MongoDB get experiment failed: {e}")
    
    async def _get_experiment_async(self, experiment_id: str) -> Experiment:
        """Async implementation of get experiment."""
        doc = await self._get_experiment_by_id(experiment_id)
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        return self._experiment_doc_to_entity(doc)
    
    def get_experiment_by_name(self, experiment_name: str) -> Experiment:
        """Get experiment by name."""
        import asyncio
        try:
            return asyncio.run(self._get_experiment_by_name_async(experiment_name))
        except Exception as e:
            logger.error(f"Failed to get experiment by name: {e}")
            raise MlflowException(f"MongoDB get experiment by name failed: {e}")
    
    async def _get_experiment_by_name_async(self, experiment_name: str) -> Experiment:
        """Async implementation of get experiment by name."""
        doc = await self._get_experiment_by_name(experiment_name)
        if not doc:
            raise MlflowException(
                f"Experiment with name '{experiment_name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        return self._experiment_doc_to_entity(doc)
    
    def delete_experiment(self, experiment_id: str):
        """Delete (mark as deleted) an experiment."""
        import asyncio
        try:
            return asyncio.run(self._delete_experiment_async(experiment_id))
        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}")
            raise MlflowException(f"MongoDB delete experiment failed: {e}")
    
    async def _delete_experiment_async(self, experiment_id: str):
        """Async implementation of delete experiment."""
        # Check if experiment exists
        doc = await self._get_experiment_by_id(experiment_id)
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Mark as deleted
        current_time = get_current_time_millis()
        await self.db[self.EXPERIMENTS_COLLECTION].update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "lifecycle_stage": LifecycleStage.DELETED,
                    "last_update_time": current_time
                }
            }
        )
        
        logger.info(f"Deleted experiment: {experiment_id}")
    
    def restore_experiment(self, experiment_id: str):
        """Restore a deleted experiment."""
        import asyncio
        try:
            return asyncio.run(self._restore_experiment_async(experiment_id))
        except Exception as e:
            logger.error(f"Failed to restore experiment: {e}")
            raise MlflowException(f"MongoDB restore experiment failed: {e}")
    
    async def _restore_experiment_async(self, experiment_id: str):
        """Async implementation of restore experiment."""
        # Check if experiment exists
        doc = await self._get_experiment_by_id(experiment_id)
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Restore from deleted state
        current_time = get_current_time_millis()
        await self.db[self.EXPERIMENTS_COLLECTION].update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "lifecycle_stage": LifecycleStage.ACTIVE,
                    "last_update_time": current_time
                }
            }
        )
        
        logger.info(f"Restored experiment: {experiment_id}")
    
    def rename_experiment(self, experiment_id: str, new_name: str):
        """Rename an experiment."""
        import asyncio
        try:
            return asyncio.run(self._rename_experiment_async(experiment_id, new_name))
        except Exception as e:
            logger.error(f"Failed to rename experiment: {e}")
            raise MlflowException(f"MongoDB rename experiment failed: {e}")
    
    async def _rename_experiment_async(self, experiment_id: str, new_name: str):
        """Async implementation of rename experiment."""
        # Check if experiment exists
        doc = await self._get_experiment_by_id(experiment_id)
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Check if new name already exists
        existing = await self._get_experiment_by_name(new_name)
        if existing and existing["experiment_id"] != experiment_id:
            raise MlflowException(
                f"Experiment with name '{new_name}' already exists.",
                error_code=RESOURCE_ALREADY_EXISTS
            )
        
        # Update name
        current_time = get_current_time_millis()
        await self.db[self.EXPERIMENTS_COLLECTION].update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "name": new_name,
                    "last_update_time": current_time
                }
            }
        )
        
        logger.info(f"Renamed experiment {experiment_id} to: {new_name}")
    
    # Placeholder implementations for other abstract methods
    # These will be implemented in subsequent iterations
    
    def get_run(self, run_id: str) -> Run:
        """Get run by ID - placeholder implementation."""
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def update_run_info(self, run_id: str, run_status: RunStatus, end_time: int, run_name: str) -> RunInfo:
        """Update run info - placeholder implementation."""
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def create_run(self, experiment_id: str, user_id: str, start_time: int, tags: List[RunTag], run_name: str) -> Run:
        """Create run - placeholder implementation.""" 
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def delete_run(self, run_id: str):
        """Delete run - placeholder implementation."""
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def restore_run(self, run_id: str):
        """Restore run - placeholder implementation."""
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def search_runs(self, experiment_ids: List[str], filter_string: str = "", run_view_type: ViewType = ViewType.ACTIVE_ONLY, max_results: int = 1000, order_by: List[str] = None, page_token: str = None) -> PagedList[Run]:
        """Search runs - placeholder implementation."""
        raise NotImplementedError("Run operations will be implemented in next iteration")
    
    def log_metric(self, run_id: str, metric: Metric):
        """Log metric - placeholder implementation."""
        raise NotImplementedError("Metric operations will be implemented in next iteration")
    
    def log_param(self, run_id: str, param: Param):
        """Log parameter - placeholder implementation."""
        raise NotImplementedError("Parameter operations will be implemented in next iteration")
    
    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag):
        """Set experiment tag - placeholder implementation."""
        raise NotImplementedError("Tag operations will be implemented in next iteration")
    
    def set_tag(self, run_id: str, tag: RunTag):
        """Set run tag - placeholder implementation."""
        raise NotImplementedError("Tag operations will be implemented in next iteration")
    
    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        """Get metric history - placeholder implementation."""
        raise NotImplementedError("Metric operations will be implemented in next iteration")
    
    def list_artifacts(self, run_id: str, path: str = None) -> List[FileInfo]:
        """List artifacts - delegates to artifact repository."""
        # This will delegate to the artifact repository (Azure Blob Storage)
        # For now, return empty list as placeholder
        return []
    
    def log_batch(self, run_id: str, metrics: List[Metric] = None, params: List[Param] = None, tags: List[RunTag] = None):
        """Log batch - placeholder implementation."""
        raise NotImplementedError("Batch operations will be implemented in next iteration")