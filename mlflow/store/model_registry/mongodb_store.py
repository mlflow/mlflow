"""
MongoDB Model Registry Store for Genesis-Flow

This module provides a MongoDB-based implementation of the MLflow Model Registry,
allowing direct integration with MongoDB without requiring a separate MLflow server.
"""

import logging
from typing import List, Optional

from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.utils.validation import _validate_model_name, _validate_model_version

_logger = logging.getLogger(__name__)


class MongoDBModelRegistryStore(AbstractStore):
    """
    MongoDB-based Model Registry Store for Genesis-Flow.
    
    This store provides model registry functionality using MongoDB as the backend,
    enabling direct integration without requiring a separate MLflow server.
    """

    def __init__(self, store_uri: str):
        """
        Initialize MongoDB Model Registry Store.
        
        Args:
            store_uri: MongoDB connection URI
        """
        super().__init__()
        self.store_uri = store_uri
        self._tracking_store = MongoDBStore(store_uri)
        self.db = self._tracking_store.db
        
        # Collections for model registry
        self.registered_models_collection = self.db.registered_models
        self.model_versions_collection = self.db.model_versions
        
        # Create indexes for better performance
        self._create_indexes()
        
        _logger.info(f"Initialized MongoDB Model Registry Store with URI: {store_uri}")

    def _create_indexes(self):
        """Create MongoDB indexes for optimal performance."""
        try:
            # Registered models indexes
            self.registered_models_collection.create_index("name", unique=True)
            self.registered_models_collection.create_index("creation_timestamp")
            self.registered_models_collection.create_index("last_updated_timestamp")
            
            # Model versions indexes
            self.model_versions_collection.create_index([("name", 1), ("version", 1)], unique=True)
            self.model_versions_collection.create_index("name")
            self.model_versions_collection.create_index("version")
            self.model_versions_collection.create_index("current_stage")
            self.model_versions_collection.create_index("creation_timestamp")
            self.model_versions_collection.create_index("last_updated_timestamp")
            self.model_versions_collection.create_index("run_id")
            
            _logger.debug("Created MongoDB indexes for model registry collections")
        except Exception as e:
            _logger.warning(f"Failed to create some indexes: {e}")

    def create_registered_model(self, name: str, tags: Optional[List] = None, description: Optional[str] = None) -> RegisteredModel:
        """
        Create a new registered model in MongoDB.
        
        Args:
            name: Name of the model
            tags: Optional list of tags
            description: Optional description
            
        Returns:
            RegisteredModel: The created registered model
        """
        _validate_model_name(name)
        
        import time
        current_time = int(time.time() * 1000)
        
        # Check if model already exists
        existing = self.registered_models_collection.find_one({"name": name})
        if existing:
            raise MlflowException(
                f"Registered model with name '{name}' already exists",
                RESOURCE_ALREADY_EXISTS
            )
        
        # Prepare model document
        model_doc = {
            "name": name,
            "description": description,
            "creation_timestamp": current_time,
            "last_updated_timestamp": current_time,
            "tags": tags or []
        }
        
        # Insert into MongoDB
        result = self.registered_models_collection.insert_one(model_doc)
        
        if not result.inserted_id:
            raise MlflowException(f"Failed to create registered model '{name}'")
        
        _logger.info(f"Created registered model: {name}")
        
        return RegisteredModel(
            name=name,
            creation_timestamp=current_time,
            last_updated_timestamp=current_time,
            description=description,
            latest_versions=[],
            tags=tags or []
        )

    def update_registered_model(self, name: str, description: Optional[str] = None) -> RegisteredModel:
        """
        Update a registered model's description.
        
        Args:
            name: Name of the model
            description: New description
            
        Returns:
            RegisteredModel: The updated registered model
        """
        _validate_model_name(name)
        
        import time
        current_time = int(time.time() * 1000)
        
        # Check if model exists
        existing = self.registered_models_collection.find_one({"name": name})
        if not existing:
            raise MlflowException(
                f"Registered model with name '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Update document
        update_doc = {
            "last_updated_timestamp": current_time
        }
        if description is not None:
            update_doc["description"] = description
        
        result = self.registered_models_collection.update_one(
            {"name": name},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise MlflowException(f"Failed to update registered model '{name}'")
        
        # Get updated document
        updated_doc = self.registered_models_collection.find_one({"name": name})
        
        _logger.info(f"Updated registered model: {name}")
        
        return RegisteredModel(
            name=name,
            creation_timestamp=updated_doc["creation_timestamp"],
            last_updated_timestamp=current_time,
            description=updated_doc.get("description"),
            latest_versions=[],
            tags=updated_doc.get("tags", [])
        )

    def delete_registered_model(self, name: str) -> None:
        """
        Delete a registered model and all its versions.
        
        Args:
            name: Name of the model to delete
        """
        _validate_model_name(name)
        
        # Check if model exists
        existing = self.registered_models_collection.find_one({"name": name})
        if not existing:
            raise MlflowException(
                f"Registered model with name '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Delete all model versions first
        self.model_versions_collection.delete_many({"name": name})
        
        # Delete the registered model
        result = self.registered_models_collection.delete_one({"name": name})
        
        if result.deleted_count == 0:
            raise MlflowException(f"Failed to delete registered model '{name}'")
        
        _logger.info(f"Deleted registered model: {name}")

    def list_registered_models(self, max_results: Optional[int] = None, page_token: Optional[str] = None) -> List[RegisteredModel]:
        """
        List all registered models.
        
        Args:
            max_results: Maximum number of results to return
            page_token: Token for pagination (not implemented)
            
        Returns:
            List[RegisteredModel]: List of registered models
        """
        cursor = self.registered_models_collection.find({}).sort("name", 1)
        
        if max_results:
            cursor = cursor.limit(max_results)
        
        models = []
        for doc in cursor:
            # Get latest versions for this model
            latest_versions = list(
                self.model_versions_collection.find({"name": doc["name"]})
                .sort("version", -1)
                .limit(5)
            )
            
            model = RegisteredModel(
                name=doc["name"],
                creation_timestamp=doc["creation_timestamp"],
                last_updated_timestamp=doc["last_updated_timestamp"],
                description=doc.get("description"),
                latest_versions=[],  # We'll populate this if needed
                tags=doc.get("tags", [])
            )
            models.append(model)
        
        return models

    def get_registered_model(self, name: str) -> RegisteredModel:
        """
        Get a registered model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            RegisteredModel: The registered model
        """
        _validate_model_name(name)
        
        doc = self.registered_models_collection.find_one({"name": name})
        if not doc:
            raise MlflowException(
                f"Registered model with name '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return RegisteredModel(
            name=doc["name"],
            creation_timestamp=doc["creation_timestamp"],
            last_updated_timestamp=doc["last_updated_timestamp"],
            description=doc.get("description"),
            latest_versions=[],
            tags=doc.get("tags", [])
        )

    def get_latest_versions(self, name: str, stages: Optional[List[str]] = None) -> List[ModelVersion]:
        """
        Get the latest versions of a model, optionally filtered by stage.
        
        Args:
            name: Name of the model
            stages: Optional list of stages to filter by
            
        Returns:
            List[ModelVersion]: List of latest model versions
        """
        _validate_model_name(name)
        
        # Check if model exists
        existing = self.registered_models_collection.find_one({"name": name})
        if not existing:
            raise MlflowException(
                f"Registered model with name '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        query = {"name": name}
        if stages:
            query["current_stage"] = {"$in": stages}
        
        # Get latest version for each stage
        pipeline = [
            {"$match": query},
            {"$sort": {"version": -1}},
            {"$group": {
                "_id": "$current_stage",
                "latest": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest"}}
        ]
        
        versions = []
        for doc in self.model_versions_collection.aggregate(pipeline):
            version = ModelVersion(
                name=doc["name"],
                version=str(doc["version"]),
                creation_timestamp=doc["creation_timestamp"],
                last_updated_timestamp=doc["last_updated_timestamp"],
                description=doc.get("description"),
                user_id=doc.get("user_id"),
                current_stage=doc.get("current_stage", "None"),
                source=doc.get("source"),
                run_id=doc.get("run_id"),
                status=doc.get("status", "READY"),
                status_message=doc.get("status_message"),
                tags=doc.get("tags", [])
            )
            versions.append(version)
        
        return versions

    def create_model_version(self, name: str, source: str, run_id: Optional[str] = None, 
                           tags: Optional[List] = None, run_link: Optional[str] = None,
                           description: Optional[str] = None) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            name: Name of the model
            source: Source path of the model
            run_id: Optional run ID
            tags: Optional list of tags
            run_link: Optional run link
            description: Optional description
            
        Returns:
            ModelVersion: The created model version
        """
        _validate_model_name(name)
        
        # Check if model exists
        existing = self.registered_models_collection.find_one({"name": name})
        if not existing:
            raise MlflowException(
                f"Registered model with name '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        import time
        current_time = int(time.time() * 1000)
        
        # Get next version number
        latest_version = self.model_versions_collection.find_one(
            {"name": name},
            sort=[("version", -1)]
        )
        next_version = 1 if not latest_version else latest_version["version"] + 1
        
        # Prepare version document
        version_doc = {
            "name": name,
            "version": next_version,
            "creation_timestamp": current_time,
            "last_updated_timestamp": current_time,
            "description": description,
            "user_id": None,  # TODO: Get from context
            "current_stage": "None",
            "source": source,
            "run_id": run_id,
            "status": "READY",
            "status_message": None,
            "tags": tags or []
        }
        
        # Insert into MongoDB
        result = self.model_versions_collection.insert_one(version_doc)
        
        if not result.inserted_id:
            raise MlflowException(f"Failed to create model version for '{name}'")
        
        _logger.info(f"Created model version {next_version} for model: {name}")
        
        return ModelVersion(
            name=name,
            version=str(next_version),
            creation_timestamp=current_time,
            last_updated_timestamp=current_time,
            description=description,
            user_id=version_doc["user_id"],
            current_stage="None",
            source=source,
            run_id=run_id,
            status="READY",
            status_message=None,
            tags=tags or []
        )

    def update_model_version(self, name: str, version: str, description: Optional[str] = None) -> ModelVersion:
        """
        Update a model version's description.
        
        Args:
            name: Name of the model
            version: Version of the model
            description: New description
            
        Returns:
            ModelVersion: The updated model version
        """
        _validate_model_name(name)
        _validate_model_version(version)
        
        import time
        current_time = int(time.time() * 1000)
        
        # Check if version exists
        existing = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        if not existing:
            raise MlflowException(
                f"Model version {version} of model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Update document
        update_doc = {
            "last_updated_timestamp": current_time
        }
        if description is not None:
            update_doc["description"] = description
        
        result = self.model_versions_collection.update_one(
            {"name": name, "version": int(version)},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise MlflowException(f"Failed to update model version {version} of '{name}'")
        
        # Get updated document
        updated_doc = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        
        _logger.info(f"Updated model version {version} for model: {name}")
        
        return ModelVersion(
            name=name,
            version=version,
            creation_timestamp=updated_doc["creation_timestamp"],
            last_updated_timestamp=current_time,
            description=updated_doc.get("description"),
            user_id=updated_doc.get("user_id"),
            current_stage=updated_doc.get("current_stage", "None"),
            source=updated_doc.get("source"),
            run_id=updated_doc.get("run_id"),
            status=updated_doc.get("status", "READY"),
            status_message=updated_doc.get("status_message"),
            tags=updated_doc.get("tags", [])
        )

    def transition_model_version_stage(self, name: str, version: str, stage: str, 
                                     archive_existing_versions: bool = False) -> ModelVersion:
        """
        Transition a model version to a new stage.
        
        Args:
            name: Name of the model
            version: Version of the model
            stage: New stage (None, Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions in the target stage
            
        Returns:
            ModelVersion: The updated model version
        """
        _validate_model_name(name)
        _validate_model_version(version)
        
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise MlflowException(
                f"Invalid stage '{stage}'. Must be one of: {valid_stages}",
                INVALID_PARAMETER_VALUE
            )
        
        import time
        current_time = int(time.time() * 1000)
        
        # Check if version exists
        existing = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        if not existing:
            raise MlflowException(
                f"Model version {version} of model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Archive existing versions if requested
        if archive_existing_versions and stage in ["Staging", "Production"]:
            self.model_versions_collection.update_many(
                {"name": name, "current_stage": stage},
                {"$set": {"current_stage": "Archived", "last_updated_timestamp": current_time}}
            )
        
        # Update the version's stage
        result = self.model_versions_collection.update_one(
            {"name": name, "version": int(version)},
            {"$set": {"current_stage": stage, "last_updated_timestamp": current_time}}
        )
        
        if result.matched_count == 0:
            raise MlflowException(f"Failed to transition model version {version} of '{name}'")
        
        # Get updated document
        updated_doc = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        
        _logger.info(f"Transitioned model version {version} of '{name}' to stage: {stage}")
        
        return ModelVersion(
            name=name,
            version=version,
            creation_timestamp=updated_doc["creation_timestamp"],
            last_updated_timestamp=current_time,
            description=updated_doc.get("description"),
            user_id=updated_doc.get("user_id"),
            current_stage=stage,
            source=updated_doc.get("source"),
            run_id=updated_doc.get("run_id"),
            status=updated_doc.get("status", "READY"),
            status_message=updated_doc.get("status_message"),
            tags=updated_doc.get("tags", [])
        )

    def delete_model_version(self, name: str, version: str) -> None:
        """
        Delete a model version.
        
        Args:
            name: Name of the model
            version: Version of the model to delete
        """
        _validate_model_name(name)
        _validate_model_version(version)
        
        # Check if version exists
        existing = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        if not existing:
            raise MlflowException(
                f"Model version {version} of model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Delete the version
        result = self.model_versions_collection.delete_one({
            "name": name,
            "version": int(version)
        })
        
        if result.deleted_count == 0:
            raise MlflowException(f"Failed to delete model version {version} of '{name}'")
        
        _logger.info(f"Deleted model version {version} for model: {name}")

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """
        Get a specific model version.
        
        Args:
            name: Name of the model
            version: Version of the model
            
        Returns:
            ModelVersion: The model version
        """
        _validate_model_name(name)
        _validate_model_version(version)
        
        doc = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        if not doc:
            raise MlflowException(
                f"Model version {version} of model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return ModelVersion(
            name=name,
            version=version,
            creation_timestamp=doc["creation_timestamp"],
            last_updated_timestamp=doc["last_updated_timestamp"],
            description=doc.get("description"),
            user_id=doc.get("user_id"),
            current_stage=doc.get("current_stage", "None"),
            source=doc.get("source"),
            run_id=doc.get("run_id"),
            status=doc.get("status", "READY"),
            status_message=doc.get("status_message"),
            tags=doc.get("tags", [])
        )

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """
        Get the download URI for a model version.
        
        Args:
            name: Name of the model
            version: Version of the model
            
        Returns:
            str: Download URI for the model version
        """
        model_version = self.get_model_version(name, version)
        return model_version.source

    def search_model_versions(self, filter_string: Optional[str] = None, 
                            max_results: Optional[int] = None,
                            order_by: Optional[List[str]] = None,
                            page_token: Optional[str] = None) -> List[ModelVersion]:
        """
        Search for model versions based on filter criteria.
        
        Args:
            filter_string: Search filter (not fully implemented)
            max_results: Maximum number of results
            order_by: Order by criteria (not fully implemented)
            page_token: Pagination token (not implemented)
            
        Returns:
            List[ModelVersion]: List of matching model versions
        """
        # Basic implementation - just return all versions
        cursor = self.model_versions_collection.find({})
        
        if max_results:
            cursor = cursor.limit(max_results)
        
        versions = []
        for doc in cursor:
            version = ModelVersion(
                name=doc["name"],
                version=str(doc["version"]),
                creation_timestamp=doc["creation_timestamp"],
                last_updated_timestamp=doc["last_updated_timestamp"],
                description=doc.get("description"),
                user_id=doc.get("user_id"),
                current_stage=doc.get("current_stage", "None"),
                source=doc.get("source"),
                run_id=doc.get("run_id"),
                status=doc.get("status", "READY"),
                status_message=doc.get("status_message"),
                tags=doc.get("tags", [])
            )
            versions.append(version)
        
        return versions