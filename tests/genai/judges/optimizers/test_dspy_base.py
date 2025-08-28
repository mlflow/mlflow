"""Tests for DSPyAlignmentOptimizer base class."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer


class ConcreteDSPyOptimizer(DSPyAlignmentOptimizer):
    """Concrete implementation for testing."""
    
    def _dspy_optimize(self, program, train_examples, val_examples, metric_fn):
        # Mock implementation for testing
        mock_program = Mock()
        mock_program.signature = Mock()
        mock_program.signature.instructions = "Optimized instructions"
        return mock_program


class TestDSPyAlignmentOptimizer:
    """Test cases for DSPyAlignmentOptimizer."""
    
    def test_dspy_optimizer_abstract(self):
        """Test that DSPyAlignmentOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DSPyAlignmentOptimizer()
    
    def test_concrete_implementation_required(self):
        """Test that concrete classes must implement _dspy_optimize method."""
        class IncompleteDSPyOptimizer(DSPyAlignmentOptimizer):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDSPyOptimizer()
    
    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        optimizer = ConcreteDSPyOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, '_logger')
        assert hasattr(optimizer, '_kwargs')
    
    def test_extract_text_from_data_string(self):
        """Test extracting text from string data."""
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._extract_text_from_data("simple string", "request")
        assert result == "simple string"
    
    def test_extract_text_from_data_dict_request(self):
        """Test extracting request text from dictionary data."""
        optimizer = ConcreteDSPyOptimizer()
        data = {"inputs": "test input", "other": "ignored"}
        result = optimizer._extract_text_from_data(data, "request")
        assert result == "test input"
    
    def test_extract_text_from_data_dict_response(self):
        """Test extracting response text from dictionary data."""
        optimizer = ConcreteDSPyOptimizer()
        data = {"outputs": "test output", "other": "ignored"}
        result = optimizer._extract_text_from_data(data, "response")
        assert result == "test output"
    
    def test_extract_text_from_data_none(self):
        """Test extracting text from None data."""
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._extract_text_from_data(None, "request")
        assert result == ""
    
    def test_extract_request_from_trace(self, sample_trace_with_assessment):
        """Test extracting request from trace."""
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._extract_request_from_trace(sample_trace_with_assessment)
        assert "test input" in result
    
    def test_extract_response_from_trace(self, sample_trace_with_assessment):
        """Test extracting response from trace."""
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._extract_response_from_trace(sample_trace_with_assessment)
        assert "test output" in result
    
    def test_sanitize_judge_name(self):
        """Test judge name sanitization."""
        optimizer = ConcreteDSPyOptimizer()
        assert optimizer._sanitize_judge_name("  Test Judge  ") == "test judge"
        assert optimizer._sanitize_judge_name("UPPERCASE") == "uppercase"
    
    def test_trace_to_dspy_example_success(self, sample_trace_with_assessment):
        """Test successful conversion of trace to DSPy example."""
        mock_dspy = MagicMock()
        mock_example = MagicMock()
        mock_example.with_inputs.return_value = mock_example
        mock_dspy.Example.return_value = mock_example
        
        with patch.dict('sys.modules', {'dspy': mock_dspy}):
            optimizer = ConcreteDSPyOptimizer()
            result = optimizer._trace_to_dspy_example(sample_trace_with_assessment, "mock_judge")
        
        assert result is not None
        mock_dspy.Example.assert_called_once()
        mock_example.with_inputs.assert_called_once_with('inputs', 'outputs')
    
    def test_trace_to_dspy_example_no_assessment(self):
        """Test trace conversion with no matching assessment."""
        mock_dspy = MagicMock()
        mock_example = MagicMock()
        mock_dspy.Example.return_value = mock_example
        
        # Create trace without assessments
        mock_trace = Mock()
        mock_trace.info.trace_id = "test"
        mock_trace.info.assessments = []
        mock_trace.info.request_preview = "test"
        mock_trace.info.response_preview = "test"
        mock_trace.data.request = "test"
        mock_trace.data.response = "test"
        
        with patch.dict('sys.modules', {'dspy': mock_dspy}):
            optimizer = ConcreteDSPyOptimizer()
            result = optimizer._trace_to_dspy_example(mock_trace, "mock_judge")
        
        assert result is None
    
    def test_trace_to_dspy_example_no_dspy(self):
        """Test trace conversion when DSPy is not available."""
        with patch.dict('sys.modules', {'dspy': None}):
            optimizer = ConcreteDSPyOptimizer()
            with pytest.raises(MlflowException, match="DSPy library is required"):
                optimizer._trace_to_dspy_example(Mock(), "judge")
    
    def test_extract_judge_instructions(self, mock_judge):
        """Test extracting instructions from judge."""
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._extract_judge_instructions(mock_judge)
        assert result == mock_judge.description
    
    def test_create_dspy_signature(self):
        """Test creating DSPy signature."""
        mock_dspy = MagicMock()
        with patch.dict('sys.modules', {'dspy': mock_dspy}):
            optimizer = ConcreteDSPyOptimizer()
            instructions = "Test instructions"
            
            optimizer._create_dspy_signature(instructions)
        
        mock_dspy.make_signature.assert_called_once()
        args, kwargs = mock_dspy.make_signature.call_args
        assert args[1] == instructions  # instructions passed as second argument
    
    def test_create_dspy_signature_no_dspy(self):
        """Test signature creation when DSPy is not available."""
        with patch.dict('sys.modules', {'dspy': None}):
            optimizer = ConcreteDSPyOptimizer()
            with pytest.raises(MlflowException, match="DSPy library is required"):
                optimizer._create_dspy_signature("test")
    
    def test_create_agreement_metric(self):
        """Test creating agreement metric function."""
        optimizer = ConcreteDSPyOptimizer()
        metric_fn = optimizer._create_agreement_metric()
        
        # Test metric with matching results
        example = Mock()
        example.result = "pass"
        pred = Mock()
        pred.result = "pass"
        
        assert metric_fn(example, pred) == 1.0
        
        # Test metric with different results
        pred.result = "fail"
        assert metric_fn(example, pred) == 0.0
    
    def test_create_agreement_metric_error_handling(self):
        """Test agreement metric error handling."""
        optimizer = ConcreteDSPyOptimizer()
        metric_fn = optimizer._create_agreement_metric()
        
        # Test with invalid inputs
        result = metric_fn(None, None)
        assert result == 0.0
    
    def test_align_success(self, mock_judge, sample_traces_with_assessments):
        """Test successful alignment process."""
        mock_dspy = MagicMock()
        # Setup mock DSPy components
        mock_example = Mock()
        mock_example.with_inputs.return_value = mock_example
        mock_dspy.Example.return_value = mock_example
        
        mock_signature = Mock()
        mock_dspy.make_signature.return_value = mock_signature
        
        mock_program = Mock()
        mock_dspy.Predict.return_value = mock_program
        
        with patch.dict('sys.modules', {'dspy': mock_dspy}):
            # Setup concrete optimizer
            optimizer = ConcreteDSPyOptimizer()
            
            result = optimizer.align(mock_judge, sample_traces_with_assessments)
        
        # Should return a judge (even if it's the original for now)
        assert result is not None
    
    def test_align_no_traces(self, mock_judge):
        """Test alignment with no traces provided."""
        optimizer = ConcreteDSPyOptimizer()
        
        with pytest.raises(MlflowException, match="No traces provided"):
            optimizer.align(mock_judge, [])
    
    def test_align_no_valid_examples(self, mock_judge):
        """Test alignment when no valid examples can be created."""
        mock_dspy = MagicMock()
        # Setup DSPy mocks
        mock_dspy.Example.side_effect = Exception("Failed to create example")
        
        # Create trace without proper assessment
        mock_trace = Mock()
        mock_trace.info.assessments = []
        mock_trace.info.request_preview = "test"
        mock_trace.info.response_preview = "test" 
        mock_trace.data.request = "test"
        mock_trace.data.response = "test"
        
        with patch.dict('sys.modules', {'dspy': mock_dspy}):
            optimizer = ConcreteDSPyOptimizer()
            
            with pytest.raises(MlflowException, match="No valid examples could be created"):
                optimizer.align(mock_judge, [mock_trace])
    
    def test_align_insufficient_examples(self, mock_judge):
        """Test alignment with insufficient examples."""
        optimizer = ConcreteDSPyOptimizer()
        
        # Mock _trace_to_dspy_example to return only one example
        with patch.object(optimizer, '_trace_to_dspy_example') as mock_trace_convert:
            mock_trace_convert.return_value = Mock()  # Only one example
            
            mock_trace = Mock()
            
            with pytest.raises(MlflowException, match="At least 2 valid examples are required"):
                optimizer.align(mock_judge, [mock_trace])
    
    def test_align_no_dspy(self, mock_judge, sample_traces_with_assessments):
        """Test alignment when DSPy is not available."""
        with patch.dict('sys.modules', {'dspy': None}):
            optimizer = ConcreteDSPyOptimizer()
            
            with pytest.raises(MlflowException, match="DSPy library is required"):
                optimizer.align(mock_judge, sample_traces_with_assessments)