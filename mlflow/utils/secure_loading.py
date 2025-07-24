"""
Genesis-Flow Secure Model Loading

This module provides secure alternatives to pickle-based model loading,
addressing security vulnerabilities in model deserialization.
"""

import io
import pickle
import cloudpickle
import logging
import hashlib
from typing import Any, Set, Optional, Union, Type
from pathlib import Path

logger = logging.getLogger(__name__)

# Allowlist of safe classes that can be unpickled
SAFE_PICKLE_CLASSES = {
    # NumPy types
    'numpy.ndarray',
    'numpy.dtype',
    'numpy.int32', 'numpy.int64', 'numpy.float32', 'numpy.float64',
    'numpy.bool_', 'numpy.str_',

    # Pandas types
    'pandas.core.frame.DataFrame',
    'pandas.core.series.Series',
    'pandas.core.index.Index',
    'pandas.core.dtypes.dtypes.CategoricalDtype',

    # Scikit-learn estimators
    'sklearn.linear_model._base.LinearRegression',
    'sklearn.linear_model._logistic.LogisticRegression',
    'sklearn.ensemble._forest.RandomForestClassifier',
    'sklearn.ensemble._forest.RandomForestRegressor',
    'sklearn.tree._classes.DecisionTreeClassifier',
    'sklearn.tree._classes.DecisionTreeRegressor',
    'sklearn.svm._classes.SVC',
    'sklearn.svm._classes.SVR',

    # Built-in types
    'builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.set',
    'builtins.str', 'builtins.int', 'builtins.float', 'builtins.bool',
    'builtins.type',

    # Collections
    'collections.OrderedDict',
    'collections.defaultdict',

    # MLflow types
    'mlflow.models.signature.ModelSignature',
    'mlflow.models.signature._TypeHints',
    'mlflow.types.schema.Schema',
    'mlflow.pyfunc.model.PythonModel',

    # Cloudpickle internals
    'cloudpickle.cloudpickle._make_skeleton_class',
    'cloudpickle.cloudpickle._class_setstate',
    'cloudpickle.cloudpickle._make_function',
    'cloudpickle.cloudpickle._builtin_type',
    'cloudpickle.cloudpickle._function_setstate',
    'cloudpickle.cloudpickle._make_empty_cell',
    'cloudpickle.cloudpickle._make_cell',

    # Sentence Transformers
    'sentence_transformers.SentenceTransformer.SentenceTransformer',
    'sentence_transformers.model_card.SentenceTransformerModelCardData',
    'sentence_transformers.models.Transformer.Transformer',
    'sentence_transformers.models.Pooling.Pooling',
    'sentence_transformers.models.Normalize.Normalize',

    # Torch
    'torch.torch_version.TorchVersion',
    'torch._utils._rebuild_tensor_v2',
    'torch.storage._load_from_bytes',
    'torch.nn.modules.sparse.Embedding',
    'torch._utils._rebuild_parameter',
    'torch.nn.modules.normalization.LayerNorm',
    'torch.nn.modules.dropout.Dropout',
    'torch.nn.modules.container.ModuleList',
    'torch.nn.modules.linear.Linear',
    'torch.nn.modules.activation.Tanh',
    'torch.float32',
    'torch._C._nn.gelu',

    # Transformers
    'transformers.models.bert.modeling_bert.BertModel',
    'transformers.models.bert.modeling_bert.BertEmbeddings',
    'transformers.models.bert.modeling_bert.BertEncoder',
    'transformers.models.bert.modeling_bert.BertLayer',
    'transformers.models.bert.modeling_bert.BertAttention',
    'transformers.models.bert.modeling_bert.BertSdpaSelfAttention',
    'transformers.models.bert.modeling_bert.BertSelfOutput',
    'transformers.models.bert.modeling_bert.BertIntermediate',
    'transformers.models.bert.modeling_bert.BertOutput',
    'transformers.models.bert.modeling_bert.BertPooler',
    'transformers.models.bert.configuration_bert.BertConfig',
    'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast',
    'transformers.activations.GELUActivation',

    # Tokenizers
    'tokenizers.Tokenizer',
    'tokenizers.models.Model',
    'tokenizers.AddedToken',
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Secure unpickler that only allows safe, whitelisted classes.
    
    This prevents arbitrary code execution during model deserialization
    by restricting which classes can be instantiated.
    """
    
    def __init__(self, file, *, safe_classes: Optional[Set[str]] = None):
        super().__init__(file)
        self.safe_classes = safe_classes or SAFE_PICKLE_CLASSES
        
    def find_class(self, module: str, name: str) -> Type:
        """
        Override to restrict class loading to safe classes only.
        
        Args:
            module: Module name
            name: Class name
            
        Returns:
            Class object if safe
            
        Raises:
            SecurityError: If class is not in allowlist
        """
        full_name = f"{module}.{name}"
        
        # Check if the class is in our safe list
        if full_name in self.safe_classes:
            logger.debug(f"Loading safe class: {full_name}")
            return super().find_class(module, name)
        
        # Additional check for known safe modules
        safe_modules = {
            'numpy': ['ndarray', 'dtype', 'int32', 'int64', 'float32', 'float64', 'bool_'],
            'pandas': ['DataFrame', 'Series', 'Index'],
            'builtins': ['dict', 'list', 'tuple', 'set', 'str', 'int', 'float', 'bool'],
        }
        
        if module in safe_modules and name in safe_modules[module]:
            logger.debug(f"Loading safe module class: {full_name}")
            return super().find_class(module, name)
        
        # Log and block unsafe class
        logger.warning(f"Blocked potentially unsafe class: {full_name}")
        raise pickle.UnpicklingError(
            f"Security: Class '{full_name}' is not in the allowlist. "
            f"If this is a legitimate model class, add it to SAFE_PICKLE_CLASSES."
        )

class SecureModelLoader:
    """
    Secure model loading with multiple safety mechanisms.
    """
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path]) -> str:
        """Calculate SHA256 hash of a file for integrity checking."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def safe_pickle_load(file_path: Union[str, Path], *, safe_classes: Optional[Set[str]] = None) -> Any:
        """
        Safely load a pickle file using restricted unpickler.
        
        Args:
            file_path: Path to pickle file
            safe_classes: Optional custom set of safe classes
            
        Returns:
            Unpickled object
            
        Raises:
            SecurityError: If file contains unsafe classes
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Log file hash for audit trail
        file_hash = SecureModelLoader.calculate_file_hash(file_path)
        logger.info(f"Loading model file {file_path.name} (SHA256: {file_hash[:16]}...)")
        
        try:
            with open(file_path, 'rb') as f:
                unpickler = RestrictedUnpickler(f, safe_classes=safe_classes)
                model = unpickler.load()
                logger.info(f"Successfully loaded model of type: {type(model).__name__}")
                return model
        except pickle.UnpicklingError as e:
            logger.error(f"Security: Unsafe model file rejected: {e}")
            raise SecurityError(f"Model file contains unsafe content: {e}") from e
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @staticmethod
    def safe_cloudpickle_load(file_path: Union[str, Path], *, safe_classes: Optional[Set[str]] = None) -> Any:
        """
        Safely load a cloudpickle file using restricted unpickler.
        
        Args:
            file_path: Path to cloudpickle file
            safe_classes: Optional custom set of safe classes
            
        Returns:
            Unpickled object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Calculate file hash for audit trail
        file_hash = SecureModelLoader.calculate_file_hash(file_path)
        logger.info(f"Loading cloudpickle file {file_path.name} (SHA256: {file_hash[:16]}...)")
        
        try:
            with open(file_path, 'rb') as f:
                # Use cloudpickle's load with our custom unpickler
                # Note: This is a simplified approach - in practice, cloudpickle's
                # load function would need to be modified to accept a custom unpickler
                unpickler = RestrictedUnpickler(f, safe_classes=safe_classes)
                model = unpickler.load()
                logger.info(f"Successfully loaded cloudpickle model of type: {type(model).__name__}")
                return model
        except pickle.UnpicklingError as e:
            logger.error(f"Security: Unsafe cloudpickle file rejected: {e}")
            raise SecurityError(f"CloudPickle file contains unsafe content: {e}") from e
        except Exception as e:
            logger.error(f"Error loading cloudpickle model: {e}")
            raise

class SecurityError(Exception):
    """Exception raised for security-related model loading issues."""
    pass

def add_safe_class(class_name: str) -> None:
    """
    Add a class to the safe loading allowlist.
    
    Args:
        class_name: Full class name (e.g., 'mymodule.MyClass')
    """
    SAFE_PICKLE_CLASSES.add(class_name)
    logger.info(f"Added {class_name} to safe loading allowlist")

def remove_safe_class(class_name: str) -> None:
    """
    Remove a class from the safe loading allowlist.
    
    Args:
        class_name: Full class name to remove
    """
    SAFE_PICKLE_CLASSES.discard(class_name)
    logger.info(f"Removed {class_name} from safe loading allowlist")

def get_safe_classes() -> Set[str]:
    """Return a copy of the current safe classes set."""
    return SAFE_PICKLE_CLASSES.copy()