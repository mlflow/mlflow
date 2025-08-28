"""SIMBA alignment optimizer implementation."""

from typing import Any

from mlflow.utils.annotations import experimental

from .dspy import DSPyAlignmentOptimizer


@experimental(version="3.4.0")
class SIMBAAlignmentOptimizer(DSPyAlignmentOptimizer):
    """
    SIMBA (Simplified Multi-Bootstrap Aggregation) alignment optimizer.
    
    Uses DSPy's SIMBA algorithm to optimize judge prompts through
    bootstrap aggregation with simplified parametrization.
    """
    
    # Class constants for default SIMBA parameters
    DEFAULT_BSIZE = 4
    DEFAULT_SEED = 42
    
    def __init__(self, model: str = None, **kwargs):
        """
        Initialize SIMBA optimizer with default parameters.
        
        Args:
            model: Model to use for DSPy optimization. If None, uses get_default_model().
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(model=model, **kwargs)
        # Private instance variables for SIMBA-specific parameters
        self._bsize = self.DEFAULT_BSIZE
        self._seed = self.DEFAULT_SEED
    
    def _dspy_optimize(self, program, train_examples, val_examples, metric_fn) -> Any:
        """
        Perform SIMBA optimization with algorithm-specific parameters.
        
        SIMBA only uses student, trainset, and seed parameters (no validation set).
        
        Args:
            program: The DSPy program to optimize
            train_examples: Training examples
            val_examples: Validation examples (unused for SIMBA)
            metric_fn: Metric function for optimization
            
        Returns:
            Optimized DSPy program
        """
        try:
            import dspy
            
            # Create SIMBA optimizer
            optimizer = dspy.SIMBA(
                metric=metric_fn,
                bsize=self._bsize
            )
            
            # Compile with SIMBA-specific parameters
            return optimizer.compile(
                student=program,
                trainset=train_examples,
                seed=self._seed,
            )
            
        except ImportError:
            from mlflow.exceptions import MlflowException
            raise MlflowException("DSPy library is required but not installed")