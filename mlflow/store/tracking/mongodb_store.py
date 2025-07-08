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
    LoggedModel,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunInputs,
    RunOutputs,
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
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, _get_run_name_from_tags
from mlflow.utils.name_utils import _generate_random_name

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
        
        # Initialize MongoDB client (both sync and async)
        try:
            # Synchronous client for blocking operations
            import pymongo
            self.sync_client = pymongo.MongoClient(
                db_uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                retryWrites=True,
                w='majority',  # Write concern for data consistency
                readPreference='primaryPreferred'
            )
            self.sync_db = self.sync_client[self.database_name]
            
            # Asynchronous client for non-blocking operations
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
            # Use synchronous implementation to avoid asyncio issues
            return self._create_experiment_sync(name, artifact_location, tags)
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
    
    def _create_experiment_sync(self, name: str, artifact_location: Optional[str], tags: Optional[List[ExperimentTag]]) -> str:
        """Synchronous implementation of create experiment."""
        # Check if experiment already exists
        logger.debug(f"Checking if experiment '{name}' exists in collection: {self.EXPERIMENTS_COLLECTION}")
        existing = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"name": name})
        logger.debug(f"Query result for experiment '{name}': {existing}")
        if existing:
            logger.error(f"Experiment '{name}' already exists: {existing}")
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
        self.sync_db[self.EXPERIMENTS_COLLECTION].insert_one(experiment_doc)
        
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID."""
        doc = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"experiment_id": experiment_id})
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        return self._experiment_doc_to_entity(doc)
    
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
        doc = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"name": experiment_name})
        if not doc:
            raise MlflowException(
                f"Experiment with name '{experiment_name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        return self._experiment_doc_to_entity(doc)
    
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
        # Check if experiment exists
        doc = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"experiment_id": experiment_id})
        if not doc:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Mark as deleted
        current_time = get_current_time_millis()
        self.sync_db[self.EXPERIMENTS_COLLECTION].update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "lifecycle_stage": LifecycleStage.DELETED,
                    "last_update_time": current_time
                }
            }
        )
        
        logger.info(f"Deleted experiment: {experiment_id}")
    
    def search_experiments(self, view_type: ViewType = ViewType.ACTIVE_ONLY, max_results: int = 1000, filter_string: str = "", order_by: List[str] = None, page_token: str = None) -> PagedList[Experiment]:
        """Search experiments in MongoDB."""
        # Build query based on view type
        query = {}
        if view_type == ViewType.ACTIVE_ONLY:
            query["lifecycle_stage"] = LifecycleStage.ACTIVE
        elif view_type == ViewType.DELETED_ONLY:
            query["lifecycle_stage"] = LifecycleStage.DELETED
        # ViewType.ALL includes both active and deleted
        
        # Build sort criteria
        sort_criteria = []
        if order_by:
            for order_item in order_by:
                if "creation_time" in order_item:
                    direction = DESCENDING if "DESC" in order_item else ASCENDING
                    sort_criteria.append(("creation_time", direction))
                elif "last_update_time" in order_item:
                    direction = DESCENDING if "DESC" in order_item else ASCENDING
                    sort_criteria.append(("last_update_time", direction))
        
        if not sort_criteria:
            sort_criteria = [("creation_time", DESCENDING)]  # Default sort
        
        # Execute query
        cursor = self.sync_db[self.EXPERIMENTS_COLLECTION].find(query).sort(sort_criteria).limit(max_results)
        
        # Convert documents to Experiment objects
        experiments = []
        for exp_doc in cursor:
            try:
                experiment = self._experiment_doc_to_entity(exp_doc)
                experiments.append(experiment)
            except Exception as e:
                logger.warning(f"Failed to convert experiment document to Experiment object: {e}")
                continue
        
        return PagedList(experiments, None)  # No pagination token for now
    
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
        """Get a run by ID from MongoDB."""
        # Get run document
        run_doc = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run_doc:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Get parameters
        param_docs = list(self.sync_db[self.PARAMS_COLLECTION].find({"run_uuid": run_id}))
        params = [Param(key=doc["key"], value=doc["value"]) for doc in param_docs]
        
        # Get metrics (latest values only)
        metric_pipeline = [
            {"$match": {"run_uuid": run_id}},
            {"$sort": {"timestamp": -1, "step": -1}},
            {"$group": {
                "_id": "$key",
                "latest": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest"}}
        ]
        
        metric_docs = list(self.sync_db[self.METRICS_COLLECTION].aggregate(metric_pipeline))
        metrics = [
            Metric(
                key=doc["key"],
                value=doc["value"],
                timestamp=doc["timestamp"],
                step=doc["step"]
            )
            for doc in metric_docs
        ]
        
        # Get tags from TAGS collection
        tag_docs = list(self.sync_db[self.TAGS_COLLECTION].find({"run_uuid": run_id}))
        tags_dict = {doc["key"]: doc["value"] for doc in tag_docs}
        
        # Add tags from run document (merge, don't duplicate)
        if run_doc.get("tags"):
            for tag_data in run_doc["tags"]:
                tags_dict[tag_data["key"]] = tag_data["value"]
        
        # Convert to RunTag objects
        tags = [RunTag(key=key, value=value) for key, value in tags_dict.items()]
        
        # Create RunInfo
        run_info = RunInfo(
            run_id=run_doc["run_uuid"],
            run_name=run_doc.get("run_name"),
            experiment_id=run_doc["experiment_id"],
            user_id=run_doc.get("user_id"),
            status=run_doc["status"],
            start_time=run_doc["start_time"],
            end_time=run_doc.get("end_time"),
            artifact_uri=run_doc["artifact_uri"],
            lifecycle_stage=run_doc.get("lifecycle_stage", LifecycleStage.ACTIVE)
        )
        
        # Create RunData
        run_data = RunData(
            metrics=metrics,
            params=params,
            tags=tags
        )
        
        # Create empty RunInputs and RunOutputs
        run_inputs = RunInputs(dataset_inputs=[])
        run_outputs = RunOutputs(model_outputs=[])
        
        return Run(run_info=run_info, run_data=run_data, run_inputs=run_inputs, run_outputs=run_outputs)
    
    def update_run_info(self, run_id: str, run_status: RunStatus, end_time: int, run_name: str) -> RunInfo:
        """Update run information in MongoDB."""
        # Validate run exists
        run_doc = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run_doc:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Prepare update document
        update_doc = {}
        if run_status is not None:
            update_doc["status"] = RunStatus.to_string(run_status)
        if end_time is not None:
            update_doc["end_time"] = end_time
        if run_name is not None:
            update_doc["run_name"] = run_name
        
        # Update run document
        if update_doc:
            result = self.sync_db[self.RUNS_COLLECTION].update_one(
                {"run_uuid": run_id},
                {"$set": update_doc}
            )
            
            if result.matched_count == 0:
                raise MlflowException(f"Failed to update run {run_id}")
        
        # Get updated run document
        updated_run_doc = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        
        # Create RunInfo
        run_info = RunInfo(
            run_id=updated_run_doc["run_uuid"],
            run_name=updated_run_doc.get("run_name"),
            experiment_id=updated_run_doc["experiment_id"],
            user_id=updated_run_doc.get("user_id"),
            status=updated_run_doc["status"],
            start_time=updated_run_doc["start_time"],
            end_time=updated_run_doc.get("end_time"),
            artifact_uri=updated_run_doc["artifact_uri"],
            lifecycle_stage=updated_run_doc.get("lifecycle_stage", LifecycleStage.ACTIVE)
        )
        
        logger.info(f"Updated run {run_id}: status={run_status}, end_time={end_time}")
        return run_info
    
    def create_run(self, experiment_id: str, user_id: str, start_time: int, tags: List[RunTag], run_name: str) -> Run:
        """Create a new run in MongoDB."""
        # Generate unique run UUID
        import uuid
        run_uuid = uuid.uuid4().hex
        
        # Validate experiment exists
        experiment = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"experiment_id": experiment_id})
        if not experiment:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Handle run name and mlflow.runName tag logic (same as SQLAlchemy store)
        tags = tags or []  # Ensure tags is not None
        run_name_tag = _get_run_name_from_tags(tags)
        
        # Validate consistency between run_name parameter and mlflow.runName tag
        if run_name and run_name_tag and (run_name != run_name_tag):
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                error_code=INVALID_PARAMETER_VALUE
            )
        
        # Determine final run name (use provided name, tag name, or generate random)
        run_name = run_name or run_name_tag or _generate_random_name()
        
        # Automatically add mlflow.runName tag if not present
        if not run_name_tag:
            tags.append(RunTag(key=MLFLOW_RUN_NAME, value=run_name))
        
        # Set default artifact URI
        artifact_uri = f"{self.default_artifact_root}/{experiment_id}/{run_uuid}/artifacts"
        
        current_time = start_time or get_current_time_millis()
        
        # Create run document
        run_doc = {
            "run_uuid": run_uuid,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": current_time,
            "end_time": None,
            "artifact_uri": artifact_uri,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "tags": [{"key": tag.key, "value": tag.value} for tag in (tags or [])],
            "creation_time": current_time,
        }
        
        # Insert run into MongoDB
        result = self.sync_db[self.RUNS_COLLECTION].insert_one(run_doc)
        
        if not result.inserted_id:
            raise MlflowException(f"Failed to create run for experiment {experiment_id}")
        
        # Create RunInfo and RunData objects
        run_info = RunInfo(
            run_id=run_uuid,
            run_name=run_name,
            experiment_id=experiment_id,
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=current_time,
            end_time=None,
            artifact_uri=artifact_uri,
            lifecycle_stage=LifecycleStage.ACTIVE
        )
        
        run_data = RunData(
            metrics=[],
            params=[],
            tags=tags  # Use the updated tags list that includes mlflow.runName
        )
        
        # Create empty RunInputs and RunOutputs
        run_inputs = RunInputs(dataset_inputs=[])
        run_outputs = RunOutputs(model_outputs=[])
        
        logger.info(f"Created run: {run_uuid} for experiment: {experiment_id}")
        return Run(run_info=run_info, run_data=run_data, run_inputs=run_inputs, run_outputs=run_outputs)
    
    def delete_run(self, run_id: str):
        """Delete a run and all its associated data from MongoDB."""
        # Validate run exists
        run_doc = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run_doc:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Delete run and all associated data
        self.sync_db[self.RUNS_COLLECTION].delete_one({"run_uuid": run_id})
        self.sync_db[self.PARAMS_COLLECTION].delete_many({"run_uuid": run_id})
        self.sync_db[self.METRICS_COLLECTION].delete_many({"run_uuid": run_id})
        self.sync_db[self.TAGS_COLLECTION].delete_many({"run_uuid": run_id})
        
        logger.info(f"Deleted run: {run_id} and all associated data")
    
    def restore_run(self, run_id: str):
        """Restore a deleted run by changing its lifecycle stage to ACTIVE."""
        # Validate run exists
        run_doc = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run_doc:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Update lifecycle stage to ACTIVE
        result = self.sync_db[self.RUNS_COLLECTION].update_one(
            {"run_uuid": run_id},
            {"$set": {"lifecycle_stage": LifecycleStage.ACTIVE}}
        )
        
        if result.matched_count == 0:
            raise MlflowException(f"Failed to restore run {run_id}")
        
        logger.info(f"Restored run: {run_id}")
    
    def search_runs(self, experiment_ids: List[str], filter_string: str = "", run_view_type: ViewType = ViewType.ACTIVE_ONLY, max_results: int = 1000, order_by: List[str] = None, page_token: str = None) -> PagedList[Run]:
        """Search runs in MongoDB with filtering and sorting."""
        # Build query
        query = {}
        
        # Filter by experiment IDs
        if experiment_ids:
            query["experiment_id"] = {"$in": experiment_ids}
        
        # Filter by lifecycle stage based on view type
        if run_view_type == ViewType.ACTIVE_ONLY:
            query["lifecycle_stage"] = LifecycleStage.ACTIVE
        elif run_view_type == ViewType.DELETED_ONLY:
            query["lifecycle_stage"] = LifecycleStage.DELETED
        # ViewType.ALL includes both active and deleted
        
        # Basic filter string parsing (simplified for now)
        # In a full implementation, this would parse MLflow filter syntax
        if filter_string:
            # Simple status filter example
            if "status" in filter_string.lower():
                if "FINISHED" in filter_string:
                    query["status"] = RunStatus.to_string(RunStatus.FINISHED)
                elif "RUNNING" in filter_string:
                    query["status"] = RunStatus.to_string(RunStatus.RUNNING)
        
        # Handle pagination
        skip = 0
        if page_token:
            try:
                skip = int(page_token)
            except ValueError:
                skip = 0
        
        # Build sort criteria
        sort_criteria = []
        if order_by:
            for order_item in order_by:
                if order_item.startswith("start_time"):
                    direction = DESCENDING if "DESC" in order_item else ASCENDING
                    sort_criteria.append(("start_time", direction))
                elif order_item.startswith("end_time"):
                    direction = DESCENDING if "DESC" in order_item else ASCENDING
                    sort_criteria.append(("end_time", direction))
        
        if not sort_criteria:
            sort_criteria = [("start_time", DESCENDING)]  # Default sort
        
        # Execute query with pagination
        cursor = self.sync_db[self.RUNS_COLLECTION].find(query).sort(sort_criteria)
        
        # Get total count for pagination
        total_count = self.sync_db[self.RUNS_COLLECTION].count_documents(query)
        
        # Apply pagination
        cursor = cursor.skip(skip).limit(max_results)
        
        # Convert documents to Run objects
        runs = []
        for run_doc in cursor:
            try:
                run = self._doc_to_run(run_doc)
                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to convert run document to Run object: {e}")
                continue
        
        # Determine next page token
        next_page_token = None
        if skip + len(runs) < total_count:
            next_page_token = str(skip + max_results)
        
        return PagedList(runs, next_page_token)
    
    def _doc_to_run(self, run_doc: Dict) -> Run:
        """Convert MongoDB document to Run object."""
        run_uuid = run_doc["run_uuid"]
        
        # Get parameters
        param_docs = list(self.sync_db[self.PARAMS_COLLECTION].find({"run_uuid": run_uuid}))
        params = [Param(key=doc["key"], value=doc["value"]) for doc in param_docs]
        
        # Get metrics (latest values only)
        metric_pipeline = [
            {"$match": {"run_uuid": run_uuid}},
            {"$sort": {"timestamp": -1, "step": -1}},
            {"$group": {
                "_id": "$key",
                "latest": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest"}}
        ]
        
        metric_docs = list(self.sync_db[self.METRICS_COLLECTION].aggregate(metric_pipeline))
        metrics = [
            Metric(
                key=doc["key"],
                value=doc["value"],
                timestamp=doc["timestamp"],
                step=doc["step"]
            )
            for doc in metric_docs
        ]
        
        # Get tags from TAGS collection
        tag_docs = list(self.sync_db[self.TAGS_COLLECTION].find({"run_uuid": run_uuid}))
        tags_dict = {doc["key"]: doc["value"] for doc in tag_docs}
        
        # Add tags from run document (merge, don't duplicate)
        if run_doc.get("tags"):
            for tag_data in run_doc["tags"]:
                tags_dict[tag_data["key"]] = tag_data["value"]
        
        # Convert to RunTag objects
        tags = [RunTag(key=key, value=value) for key, value in tags_dict.items()]
        
        # Create RunInfo
        run_info = RunInfo(
            run_id=run_doc["run_uuid"],
            run_name=run_doc.get("run_name"),
            experiment_id=run_doc["experiment_id"],
            user_id=run_doc.get("user_id"),
            status=run_doc["status"],
            start_time=run_doc["start_time"],
            end_time=run_doc.get("end_time"),
            artifact_uri=run_doc["artifact_uri"],
            lifecycle_stage=run_doc.get("lifecycle_stage", LifecycleStage.ACTIVE)
        )
        
        # Create RunData
        run_data = RunData(
            metrics=metrics,
            params=params,
            tags=tags
        )
        
        # Create empty RunInputs and RunOutputs
        run_inputs = RunInputs(dataset_inputs=[])
        run_outputs = RunOutputs(model_outputs=[])
        
        return Run(run_info=run_info, run_data=run_data, run_inputs=run_inputs, run_outputs=run_outputs)
    
    def log_metric(self, run_id: str, metric: Metric):
        """Log a metric to MongoDB."""
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Create metric document
        metric_doc = {
            "run_uuid": run_id,
            "key": metric.key,
            "value": metric.value,
            "timestamp": metric.timestamp,
            "step": metric.step or 0,
            "is_nan": str(metric.value).lower() == 'nan',
            "creation_time": get_current_time_millis(),
        }
        
        # Insert metric (allow duplicates for time series)
        result = self.sync_db[self.METRICS_COLLECTION].insert_one(metric_doc)
        
        if not result.inserted_id:
            raise MlflowException(f"Failed to log metric {metric.key} for run {run_id}")
        
        logger.debug(f"Logged metric {metric.key}={metric.value} for run {run_id}")
    
    def log_param(self, run_id: str, param: Param):
        """Log a parameter to MongoDB."""
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Check if parameter already exists (parameters are immutable)
        existing_param = self.sync_db[self.PARAMS_COLLECTION].find_one({
            "run_uuid": run_id,
            "key": param.key
        })
        
        if existing_param:
            if existing_param["value"] != param.value:
                raise MlflowException(
                    f"Changing param values is not allowed. Param with key='{param.key}' was already "
                    f"logged with value='{existing_param['value']}' for run ID='{run_id}'. "
                    f"Attempted logging new value '{param.value}'.",
                    error_code=INVALID_PARAMETER_VALUE
                )
            # Parameter already exists with same value, no need to log again
            return
        
        # Create parameter document
        param_doc = {
            "run_uuid": run_id,
            "key": param.key,
            "value": param.value,
            "creation_time": get_current_time_millis(),
        }
        
        # Insert parameter
        result = self.sync_db[self.PARAMS_COLLECTION].insert_one(param_doc)
        
        if not result.inserted_id:
            raise MlflowException(f"Failed to log param {param.key} for run {run_id}")
        
        logger.debug(f"Logged param {param.key}={param.value} for run {run_id}")
    
    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag):
        """Set an experiment tag in MongoDB."""
        # Validate experiment exists
        experiment = self.sync_db[self.EXPERIMENTS_COLLECTION].find_one({"experiment_id": experiment_id})
        if not experiment:
            raise MlflowException(
                f"Experiment with id '{experiment_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Update experiment with tag
        result = self.sync_db[self.EXPERIMENTS_COLLECTION].update_one(
            {"experiment_id": experiment_id},
            {
                "$push": {
                    "tags": {"key": tag.key, "value": tag.value}
                },
                "$set": {
                    "last_update_time": get_current_time_millis()
                }
            }
        )
        
        if result.matched_count == 0 and not result.upserted_id:
            raise MlflowException(f"Failed to set tag {tag.key} for experiment {experiment_id}")
        
        logger.debug(f"Set experiment tag {tag.key}={tag.value} for experiment {experiment_id}")
    
    def set_tag(self, run_id: str, tag: RunTag):
        """Set a tag for a run in MongoDB."""
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Create tag document
        tag_doc = {
            "run_uuid": run_id,
            "key": tag.key,
            "value": tag.value,
            "creation_time": get_current_time_millis(),
        }
        
        # Use upsert to replace existing tag with same key
        result = self.sync_db[self.TAGS_COLLECTION].replace_one(
            {"run_uuid": run_id, "key": tag.key},
            tag_doc,
            upsert=True
        )
        
        if result.matched_count == 0 and not result.upserted_id:
            raise MlflowException(f"Failed to set tag {tag.key} for run {run_id}")
        
        logger.debug(f"Set tag {tag.key}={tag.value} for run {run_id}")
    
    def get_metric_history(self, run_id: str, metric_key: str, max_results: int = None, page_token: str = None) -> List[Metric]:
        """Get the history of a metric for a run."""
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Get all metric values for this key, sorted by timestamp and step
        query = {"run_uuid": run_id, "key": metric_key}
        cursor = self.sync_db[self.METRICS_COLLECTION].find(query).sort([("timestamp", ASCENDING), ("step", ASCENDING)])
        
        # Apply max_results limit if specified
        if max_results is not None:
            cursor = cursor.limit(max_results)
            
        metric_docs = list(cursor)
        
        # Convert to Metric objects
        metrics = [
            Metric(
                key=doc["key"],
                value=doc["value"],
                timestamp=doc["timestamp"],
                step=doc["step"]
            )
            for doc in metric_docs
        ]
        
        return metrics
    
    def log_batch(self, run_id, metrics, params, tags):
        """Log a batch of metrics, params, and tags for a run."""
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        current_time = get_current_time_millis()
        
        # Log metrics
        if metrics:
            metric_docs = []
            for metric in metrics:
                metric_doc = {
                    "run_uuid": run_id,
                    "key": metric.key,
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "step": metric.step or 0,
                    "is_nan": str(metric.value).lower() == 'nan',
                    "creation_time": current_time,
                }
                metric_docs.append(metric_doc)
            
            if metric_docs:
                self.sync_db[self.METRICS_COLLECTION].insert_many(metric_docs)
                logger.debug(f"Logged {len(metric_docs)} metrics for run {run_id}")
        
        # Log parameters
        if params:
            param_docs = []
            for param in params:
                # Check if parameter already exists
                existing_param = self.sync_db[self.PARAMS_COLLECTION].find_one({
                    "run_uuid": run_id,
                    "key": param.key
                })
                
                if existing_param:
                    if existing_param["value"] != param.value:
                        raise MlflowException(
                            f"Changing param values is not allowed. Param with key='{param.key}' was already "
                            f"logged with value='{existing_param['value']}' for run ID='{run_id}'. "
                            f"Attempted logging new value '{param.value}'.",
                            error_code=INVALID_PARAMETER_VALUE
                        )
                else:
                    param_doc = {
                        "run_uuid": run_id,
                        "key": param.key,
                        "value": param.value,
                        "creation_time": current_time,
                    }
                    param_docs.append(param_doc)
            
            if param_docs:
                self.sync_db[self.PARAMS_COLLECTION].insert_many(param_docs)
                logger.debug(f"Logged {len(param_docs)} params for run {run_id}")
        
        # Log tags
        if tags:
            for tag in tags:
                tag_doc = {
                    "run_uuid": run_id,
                    "key": tag.key,
                    "value": tag.value,
                    "creation_time": current_time,
                }
                
                # Use upsert to replace existing tag with same key
                self.sync_db[self.TAGS_COLLECTION].replace_one(
                    {"run_uuid": run_id, "key": tag.key},
                    tag_doc,
                    upsert=True
                )
            
            logger.debug(f"Logged {len(tags)} tags for run {run_id}")
    
    def log_outputs(self, run_id: str, models):
        """
        Log outputs, such as models, to the specified run.
        
        Args:
            run_id: String id for the run
            models: List of LoggedModelOutput instances to log as outputs of the run.
        """
        # Validate run exists
        run = self.sync_db[self.RUNS_COLLECTION].find_one({"run_uuid": run_id})
        if not run:
            raise MlflowException(
                f"Run with id '{run_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        current_time = get_current_time_millis()
        
        # Process each model output
        for model in models:
            try:
                # Store model output metadata in a dedicated collection
                model_output_doc = {
                    "run_uuid": run_id,
                    "model_id": model.model_id,
                    "step": model.step,
                    "output_type": "model",
                    "logged_time": current_time
                }
                
                # Insert into run_outputs collection
                self.sync_db["run_outputs"].insert_one(model_output_doc)
                
                logger.debug(f"Logged model output for run {run_id}: {model.model_id}")
                
            except Exception as e:
                logger.warning(f"Failed to log model output {model.model_id} for run {run_id}: {e}")
    
    def record_logged_model(self, run_id: str, mlflow_model):
        """Record a logged model (placeholder implementation)."""
        # This would typically store model metadata in MongoDB
        # For now, we'll just log a debug message
        logger.debug(f"Record logged model for run {run_id}: {mlflow_model}")
        pass
    
    def create_logged_model(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        source_run_id: Optional[str] = None,
        tags: Optional[List] = None,
        params: Optional[List] = None,
        model_type: Optional[str] = None,
    ):
        """
        Create a new logged model in MongoDB.
        
        Args:
            experiment_id: ID of the experiment to which the model belongs.
            name: Name of the model. If not specified, a random name will be generated.
            source_run_id: ID of the run that produced the model.
            tags: Tags to set on the model.
            params: Parameters to set on the model.
            model_type: Type of the model.
            
        Returns:
            The created LoggedModel object.
        """
        from mlflow.entities import LoggedModel
        import uuid
        
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Generate name if not provided
        if not name:
            name = f"model_{model_id[:8]}"
        
        current_time = get_current_time_millis()
        
        # Create logged model document
        logged_model_doc = {
            "_id": model_id,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "name": name,
            "source_run_id": source_run_id,
            "model_type": model_type,
            "creation_time": current_time,
            "last_update_time": current_time,
            "tags": [],
            "params": []
        }
        
        # Add tags if provided
        if tags:
            for tag in tags:
                logged_model_doc["tags"].append({
                    "key": tag.key,
                    "value": tag.value
                })
        
        # Add params if provided  
        if params:
            for param in params:
                logged_model_doc["params"].append({
                    "key": param.key,
                    "value": param.value
                })
        
        # Store in MongoDB
        self.sync_db["logged_models"].insert_one(logged_model_doc)
        
        logger.info(f"Created logged model: {name} (ID: {model_id})")
        
        # Return LoggedModel entity
        artifact_location = f"models:/{model_id}"
        return LoggedModel(
            model_id=model_id,
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            creation_timestamp=current_time,
            last_updated_timestamp=current_time,
            source_run_id=source_run_id,
            model_type=model_type,
            tags=tags or [],
            params=params or []
        )
    
    def search_logged_models(
        self,
        experiment_ids: List[str],
        filter_string: Optional[str] = None,
        datasets: Optional[List[Dict[str, Any]]] = None,
        max_results: Optional[int] = None,
        order_by: Optional[List[Dict[str, Any]]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList:
        """
        Search for logged models in MongoDB.
        
        Args:
            experiment_ids: List of experiment IDs to search within.
            filter_string: Filter string for model search.
            datasets: Dataset filters.
            max_results: Maximum number of results to return.
            order_by: Order by criteria.
            page_token: Page token for pagination.
            
        Returns:
            PagedList of LoggedModel objects.
        """
        from mlflow.entities import LoggedModel
        
        # Build query
        query = {"experiment_id": {"$in": experiment_ids}}
        
        # Apply filter string if provided (basic implementation)
        if filter_string:
            # Simple name-based filtering
            if "name" in filter_string:
                name_value = filter_string.split("=")[-1].strip("'\"")
                query["name"] = {"$regex": name_value, "$options": "i"}
        
        # Apply sorting
        sort_criteria = [("creation_time", DESCENDING)]  # Default sort
        if order_by:
            sort_criteria = []
            for order_item in order_by:
                field = order_item.get("key", "creation_time")
                direction = DESCENDING if order_item.get("descending", True) else ASCENDING
                sort_criteria.append((field, direction))
        
        # Apply limit
        limit = max_results or 1000
        
        # Execute query
        cursor = self.sync_db["logged_models"].find(query).sort(sort_criteria).limit(limit)
        
        # Convert to LoggedModel objects
        logged_models = []
        for doc in cursor:
            # Convert tags and params back to objects
            tags = []
            for tag_doc in doc.get("tags", []):
                from mlflow.entities import LoggedModelTag
                tags.append(LoggedModelTag(key=tag_doc["key"], value=tag_doc["value"]))
            
            params = []
            for param_doc in doc.get("params", []):
                from mlflow.entities import LoggedModelParameter
                params.append(LoggedModelParameter(key=param_doc["key"], value=param_doc["value"]))
            
            logged_model = LoggedModel(
                model_id=doc["model_id"],
                experiment_id=doc["experiment_id"],
                name=doc["name"],
                artifact_location=f"models:/{doc['model_id']}",
                creation_timestamp=doc["creation_time"],
                last_updated_timestamp=doc["last_update_time"],
                source_run_id=doc.get("source_run_id"),
                model_type=doc.get("model_type"),
                tags=tags,
                params=params
            )
            logged_models.append(logged_model)
        
        # Return as PagedList
        return PagedList(logged_models, next_page_token=None)
    
    def finalize_logged_model(self, model_id: str, status) -> LoggedModel:
        """
        Finalize a model by updating its status.
        
        Args:
            model_id: ID of the model to finalize.
            status: Final status to set on the model.
            
        Returns:
            The updated LoggedModel object.
        """
        from mlflow.entities import LoggedModel, LoggedModelStatus
        
        current_time = get_current_time_millis()
        
        # Update model status in MongoDB
        update_doc = {
            "status": status.to_int() if hasattr(status, 'to_int') else int(status),
            "last_update_time": current_time
        }
        
        result = self.sync_db["logged_models"].update_one(
            {"model_id": model_id},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Logged model with id '{model_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Return updated model
        return self.get_logged_model(model_id)
    
    def set_logged_model_tags(self, model_id: str, tags: List) -> None:
        """
        Set tags on the specified logged model.
        
        Args:
            model_id: ID of the model.
            tags: Tags to set on the model.
        """
        # Convert tags to dict format
        tag_docs = []
        for tag in tags:
            tag_docs.append({
                "key": tag.key,
                "value": tag.value
            })
        
        # Update model tags in MongoDB
        result = self.sync_db["logged_models"].update_one(
            {"model_id": model_id},
            {
                "$set": {
                    "tags": tag_docs,
                    "last_update_time": get_current_time_millis()
                }
            }
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Logged model with id '{model_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
    
    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """
        Delete a tag from the specified logged model.
        
        Args:
            model_id: ID of the model.
            key: Key of the tag to delete.
        """
        # Get current model
        model_doc = self.sync_db["logged_models"].find_one({"model_id": model_id})
        if not model_doc:
            raise MlflowException(
                f"Logged model with id '{model_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Remove tag with specified key
        updated_tags = [tag for tag in model_doc.get("tags", []) if tag["key"] != key]
        
        # Update model
        self.sync_db["logged_models"].update_one(
            {"model_id": model_id},
            {
                "$set": {
                    "tags": updated_tags,
                    "last_update_time": get_current_time_millis()
                }
            }
        )
    
    def get_logged_model(self, model_id: str) -> LoggedModel:
        """
        Fetch the logged model with the specified ID.
        
        Args:
            model_id: ID of the model to fetch.
            
        Returns:
            The fetched LoggedModel object.
        """
        from mlflow.entities import LoggedModel, LoggedModelTag, LoggedModelParameter
        
        doc = self.sync_db["logged_models"].find_one({"model_id": model_id})
        if not doc:
            raise MlflowException(
                f"Logged model with id '{model_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        # Convert tags and params back to objects
        tags = []
        for tag_doc in doc.get("tags", []):
            tags.append(LoggedModelTag(key=tag_doc["key"], value=tag_doc["value"]))
        
        params = []
        for param_doc in doc.get("params", []):
            params.append(LoggedModelParameter(key=param_doc["key"], value=param_doc["value"]))
        
        return LoggedModel(
            model_id=doc["model_id"],
            experiment_id=doc["experiment_id"],
            name=doc["name"],
            artifact_location=f"models:/{doc['model_id']}",
            creation_timestamp=doc["creation_time"],
            last_updated_timestamp=doc["last_update_time"],
            source_run_id=doc.get("source_run_id"),
            model_type=doc.get("model_type"),
            tags=tags,
            params=params
        )
    
    def delete_logged_model(self, model_id: str) -> None:
        """
        Delete the logged model with the specified ID.
        
        Args:
            model_id: ID of the model to delete.
        """
        result = self.sync_db["logged_models"].delete_one({"model_id": model_id})
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Logged model with id '{model_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST
            )
        
        logger.info(f"Deleted logged model: {model_id}")
    
    def list_artifacts(self, run_id: str, path: str = None) -> List[FileInfo]:
        """List artifacts for a run (placeholder implementation)."""
        # This would typically interact with artifact storage (Azure Blob, S3, etc.)
        # For now, return empty list
        logger.debug(f"List artifacts for run {run_id}, path: {path}")
        return []
    
    def download_artifacts(self, run_id: str, path: str, dst_path: str = None) -> str:
        """Download artifacts for a run (placeholder implementation)."""
        # This would typically download from artifact storage
        logger.debug(f"Download artifacts for run {run_id}, path: {path}, dst: {dst_path}")
        raise NotImplementedError("Artifact operations require artifact store implementation")
    
    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str = None):
        """Log artifacts for a run (placeholder implementation)."""
        # This would typically upload to artifact storage
        logger.debug(f"Log artifacts for run {run_id}, local_dir: {local_dir}, artifact_path: {artifact_path}")
        raise NotImplementedError("Artifact operations require artifact store implementation")
    
