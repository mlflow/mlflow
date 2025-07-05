"""
Genesis-Flow MongoDB Configuration

Configuration management for MongoDB/Cosmos DB connections with Azure integration.
"""

import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class MongoDBConfig:
    """Configuration manager for MongoDB/Azure Cosmos DB connections."""
    
    # Environment variable names
    MONGODB_URI_ENV = "GENESIS_FLOW_MONGODB_URI"
    COSMOS_DB_URI_ENV = "GENESIS_FLOW_COSMOS_DB_URI"
    ARTIFACT_ROOT_ENV = "GENESIS_FLOW_ARTIFACT_ROOT"
    
    # Default values
    DEFAULT_DATABASE_NAME = "genesis_flow"
    DEFAULT_ARTIFACT_ROOT = "azure://genesis-flow-artifacts"
    
    # Azure Cosmos DB specific settings
    COSMOS_DB_SETTINGS = {
        "retryWrites": False,  # Cosmos DB doesn't support retryable writes
        "w": 1,  # Write concern for Cosmos DB
        "readPreference": "primary",
        "ssl": True,
        "serverSelectionTimeoutMS": 10000,
        "maxPoolSize": 100,
        "minPoolSize": 10,
        "connectTimeoutMS": 20000,
        "socketTimeoutMS": 30000,
    }
    
    # Standard MongoDB settings
    MONGODB_SETTINGS = {
        "retryWrites": True,
        "w": "majority",
        "readPreference": "primaryPreferred", 
        "serverSelectionTimeoutMS": 5000,
        "maxPoolSize": 50,
        "minPoolSize": 5,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 20000,
    }
    
    @classmethod
    def get_connection_config(cls) -> Dict[str, Any]:
        """
        Get MongoDB connection configuration from environment.
        
        Returns:
            Dict containing connection URI and settings
        """
        # Try Cosmos DB first, then MongoDB
        cosmos_uri = os.getenv(cls.COSMOS_DB_URI_ENV)
        mongodb_uri = os.getenv(cls.MONGODB_URI_ENV)
        
        if cosmos_uri:
            logger.info("Using Azure Cosmos DB configuration")
            return {
                "uri": cosmos_uri,
                "settings": cls.COSMOS_DB_SETTINGS,
                "database_name": cls._extract_database_name(cosmos_uri),
                "is_cosmos_db": True,
            }
        elif mongodb_uri:
            logger.info("Using MongoDB configuration")
            return {
                "uri": mongodb_uri,
                "settings": cls.MONGODB_SETTINGS,
                "database_name": cls._extract_database_name(mongodb_uri),
                "is_cosmos_db": False,
            }
        else:
            # Default to local MongoDB for development
            logger.warning("No MongoDB URI configured, using local MongoDB")
            local_uri = f"mongodb://localhost:27017/{cls.DEFAULT_DATABASE_NAME}"
            return {
                "uri": local_uri,
                "settings": cls.MONGODB_SETTINGS,
                "database_name": cls.DEFAULT_DATABASE_NAME,
                "is_cosmos_db": False,
            }
    
    @classmethod
    def _extract_database_name(cls, uri: str) -> str:
        """Extract database name from MongoDB URI."""
        try:
            parsed = urlparse(uri)
            # Database name is the path without leading slash
            db_name = parsed.path.lstrip('/')
            if not db_name:
                db_name = cls.DEFAULT_DATABASE_NAME
            return db_name
        except Exception as e:
            logger.warning(f"Failed to extract database name from URI: {e}")
            return cls.DEFAULT_DATABASE_NAME
    
    @classmethod
    def get_artifact_root(cls) -> str:
        """Get artifact root location from environment."""
        return os.getenv(cls.ARTIFACT_ROOT_ENV, cls.DEFAULT_ARTIFACT_ROOT)
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """
        Validate MongoDB configuration.
        
        Returns:
            True if configuration is valid
        """
        config = cls.get_connection_config()
        
        try:
            # Basic URI validation
            parsed = urlparse(config["uri"])
            if not parsed.scheme.startswith("mongodb"):
                logger.error(f"Invalid MongoDB scheme: {parsed.scheme}")
                return False
            
            if not parsed.hostname:
                logger.error("MongoDB hostname not specified")
                return False
            
            # Cosmos DB specific validation
            if config["is_cosmos_db"]:
                if "documents.azure.com" not in parsed.hostname:
                    logger.warning("Cosmos DB URI should contain 'documents.azure.com'")
                
                # Check for required Cosmos DB parameters
                query_params = parse_qs(parsed.query)
                required_params = ["ssl", "retrywrites", "maxidletime"]
                missing_params = [p for p in required_params if p not in query_params]
                if missing_params:
                    logger.info(f"Cosmos DB URI missing recommended parameters: {missing_params}")
            
            logger.info("MongoDB configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB configuration validation failed: {e}")
            return False
    
    @classmethod
    def get_connection_string_template(cls) -> Dict[str, str]:
        """
        Get connection string templates for different environments.
        
        Returns:
            Dict containing connection string templates
        """
        return {
            "local_mongodb": "mongodb://localhost:27017/genesis_flow",
            "mongodb_atlas": "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/genesis_flow?retryWrites=true&w=majority",
            "azure_cosmos_db": "mongodb://<account-name>:<password>@<account-name>.mongo.cosmos.azure.com:10255/genesis_flow?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@<account-name>@",
        }
    
    @classmethod
    def create_test_config(cls, use_memory: bool = False) -> Dict[str, Any]:
        """
        Create test configuration for unit testing.
        
        Args:
            use_memory: Whether to use in-memory database (requires mongomock)
            
        Returns:
            Test configuration dict
        """
        if use_memory:
            # Use mongomock for testing
            return {
                "uri": "mongomock://localhost/test_genesis_flow",
                "settings": {},
                "database_name": "test_genesis_flow",
                "is_cosmos_db": False,
            }
        else:
            # Use local test MongoDB
            return {
                "uri": "mongodb://localhost:27017/test_genesis_flow",
                "settings": cls.MONGODB_SETTINGS.copy(),
                "database_name": "test_genesis_flow", 
                "is_cosmos_db": False,
            }


class AzureIntegration:
    """Helper class for Azure-specific integrations."""
    
    @staticmethod
    def get_cosmos_db_connection_string() -> Optional[str]:
        """
        Get Cosmos DB connection string from Azure environment.
        
        Checks multiple environment variables in order of preference:
        1. GENESIS_FLOW_COSMOS_DB_URI (explicit override)
        2. AZURE_COSMOS_DB_CONNECTION_STRING (Azure standard)
        3. COSMOSDB_CONNECTION_STRING (alternative naming)
        """
        env_vars = [
            "GENESIS_FLOW_COSMOS_DB_URI",
            "AZURE_COSMOS_DB_CONNECTION_STRING", 
            "COSMOSDB_CONNECTION_STRING",
        ]
        
        for env_var in env_vars:
            connection_string = os.getenv(env_var)
            if connection_string:
                logger.info(f"Found Cosmos DB connection string in {env_var}")
                return connection_string
        
        logger.debug("No Cosmos DB connection string found in environment")
        return None
    
    @staticmethod
    def get_azure_blob_artifact_root() -> Optional[str]:
        """
        Get Azure Blob Storage artifact root from environment.
        
        Returns Azure Blob Storage URI for artifacts.
        """
        # Check for explicit artifact root
        artifact_root = os.getenv("GENESIS_FLOW_ARTIFACT_ROOT")
        if artifact_root:
            return artifact_root
        
        # Try to construct from Azure storage account
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        if storage_account:
            container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "genesis-flow-artifacts")
            return f"azure://{storage_account}.blob.core.windows.net/{container}"
        
        # Default Azure Blob location
        return "azure://genesis-flow-artifacts"
    
    @staticmethod
    def validate_azure_credentials() -> bool:
        """
        Validate Azure credentials for Cosmos DB and Blob Storage.
        
        Returns:
            True if credentials are available
        """
        has_cosmos_db = bool(AzureIntegration.get_cosmos_db_connection_string())
        
        # Check for Azure Blob Storage credentials
        has_blob_storage = any([
            os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
            os.getenv("AZURE_CLIENT_ID"),  # Managed Identity
        ])
        
        if has_cosmos_db and has_blob_storage:
            logger.info("Azure Cosmos DB and Blob Storage credentials found")
            return True
        elif has_cosmos_db:
            logger.warning("Cosmos DB credentials found but missing Blob Storage credentials")
            return True
        elif has_blob_storage:
            logger.warning("Blob Storage credentials found but missing Cosmos DB credentials")
            return False
        else:
            logger.info("No Azure credentials found, using local development setup")
            return False


def get_mongodb_store_config() -> Dict[str, Any]:
    """
    Get complete MongoDB store configuration.
    
    Returns:
        Complete configuration dict for MongoDB store initialization
    """
    # Get base MongoDB configuration
    config = MongoDBConfig.get_connection_config()
    
    # Add artifact root
    config["artifact_root"] = MongoDBConfig.get_artifact_root()
    
    # Add Azure-specific settings if available
    if config.get("is_cosmos_db"):
        config["azure_integration"] = True
        config["blob_artifact_root"] = AzureIntegration.get_azure_blob_artifact_root()
    else:
        config["azure_integration"] = False
    
    # Validate configuration
    if not MongoDBConfig.validate_configuration():
        logger.warning("MongoDB configuration validation failed, proceeding with defaults")
    
    return config