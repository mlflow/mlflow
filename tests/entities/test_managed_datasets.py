"""
Tests for managed datasets entities in MLflow.

This module tests the entities defined in mlflow.entities.managed_datasets,
including ManagedDataset, DatasetRecord, and various source types.
"""

import json
import time
import uuid
from unittest.mock import patch

import pytest

from mlflow.entities.managed_datasets import (
    DatasetRecord,
    DatasetRecordSource,
    DocumentSource,
    ExpectationValue,
    HumanSource,
    InputValue,
    ManagedDataset,
    TraceSource,
    create_document_source,
    create_human_source,
    create_trace_source,
    get_source_summary,
)
from mlflow.exceptions import MlflowException


class TestInputValue:
    """Test InputValue entity."""

    def test_input_value_creation_and_properties(self):
        """Test InputValue creation and property access."""
        input_value = InputValue("question", "What is MLflow?")
        
        assert input_value.key == "question"
        assert input_value.value == "What is MLflow?"

    def test_input_value_protobuf_conversion(self):
        """Test InputValue protobuf serialization/deserialization."""
        input_value = InputValue("context", {"doc": "mlflow.pdf", "page": 1})
        
        proto = input_value.to_proto()
        input_value_from_proto = InputValue.from_proto(proto)
        
        assert input_value == input_value_from_proto
        assert input_value_from_proto.key == "context"
        assert input_value_from_proto.value == {"doc": "mlflow.pdf", "page": 1}

    def test_input_value_dict_conversion(self):
        """Test InputValue dictionary serialization/deserialization."""
        input_value = InputValue("prompt", "You are a helpful assistant")
        
        input_dict = input_value.to_dict()
        input_value_from_dict = InputValue.from_dict(input_dict)
        
        assert input_value == input_value_from_dict
        assert input_dict == {"key": "prompt", "value": "You are a helpful assistant"}

    def test_input_value_complex_types(self):
        """Test InputValue with complex data types."""
        complex_value = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "metadata": {"temperature": 0.7, "max_tokens": 100}
        }
        input_value = InputValue("chat_input", complex_value)
        
        # Test protobuf round-trip
        proto = input_value.to_proto()
        restored = InputValue.from_proto(proto)
        assert restored.value == complex_value
        
        # Test dict round-trip
        input_dict = input_value.to_dict()
        restored_dict = InputValue.from_dict(input_dict)
        assert restored_dict.value == complex_value


class TestExpectationValue:
    """Test ExpectationValue entity."""

    def test_expectation_value_creation(self):
        """Test ExpectationValue creation and property access."""
        expectation = ExpectationValue("MLflow is an ML lifecycle platform")
        assert expectation.value == "MLflow is an ML lifecycle platform"

    def test_expectation_value_protobuf_conversion(self):
        """Test ExpectationValue protobuf serialization/deserialization."""
        expectation = ExpectationValue({"score": 0.95, "confidence": "high"})
        
        proto = expectation.to_proto()
        expectation_from_proto = ExpectationValue.from_proto(proto)
        
        assert expectation == expectation_from_proto
        assert expectation_from_proto.value == {"score": 0.95, "confidence": "high"}

    def test_expectation_value_dict_conversion(self):
        """Test ExpectationValue dictionary serialization/deserialization."""
        expectation = ExpectationValue(42)
        
        expectation_dict = expectation.to_dict()
        expectation_from_dict = ExpectationValue.from_dict(expectation_dict)
        
        assert expectation == expectation_from_dict
        assert expectation_dict == {"value": 42}


class TestDatasetRecordSource:
    """Test DatasetRecordSource and its subclasses."""

    def test_human_source_creation(self):
        """Test HumanSource creation and properties."""
        source = HumanSource("user123")
        
        assert source.source_type == "human"
        assert source.user_id == "user123"
        assert source.source_data == {"user_id": "user123"}

    def test_document_source_creation(self):
        """Test DocumentSource creation and properties."""
        source = DocumentSource("s3://bucket/doc.pdf", "Sample content")
        
        assert source.source_type == "document"
        assert source.doc_uri == "s3://bucket/doc.pdf"
        assert source.content == "Sample content"
        assert source.source_data == {"doc_uri": "s3://bucket/doc.pdf", "content": "Sample content"}

    def test_document_source_without_content(self):
        """Test DocumentSource creation without content."""
        source = DocumentSource("s3://bucket/doc.pdf")
        
        assert source.source_type == "document"
        assert source.doc_uri == "s3://bucket/doc.pdf"
        assert source.content is None

    def test_trace_source_creation(self):
        """Test TraceSource creation and properties."""
        source = TraceSource("trace-123", "span-456")
        
        assert source.source_type == "trace"
        assert source.trace_id == "trace-123"
        assert source.span_id == "span-456"
        assert source.source_data == {"trace_id": "trace-123", "span_id": "span-456"}

    def test_trace_source_without_span(self):
        """Test TraceSource creation without span ID."""
        source = TraceSource("trace-123")
        
        assert source.source_type == "trace"
        assert source.trace_id == "trace-123"
        assert source.span_id is None

    def test_source_protobuf_conversion(self):
        """Test source protobuf serialization/deserialization."""
        sources = [
            HumanSource("user123"),
            DocumentSource("doc.pdf", "content"),
            TraceSource("trace-123", "span-456")
        ]
        
        for source in sources:
            proto = source.to_proto()
            source_from_proto = DatasetRecordSource.from_proto(proto)
            assert source == source_from_proto

    def test_source_dict_conversion(self):
        """Test source dictionary serialization/deserialization."""
        sources = [
            HumanSource("user123"),
            DocumentSource("doc.pdf", "content"),
            TraceSource("trace-123", "span-456")
        ]
        
        for source in sources:
            source_dict = source.to_dict()
            source_from_dict = DatasetRecordSource.from_dict(source_dict)
            assert source == source_from_dict

    def test_factory_functions(self):
        """Test factory functions for creating sources."""
        human = create_human_source("user123")
        assert isinstance(human, HumanSource)
        assert human.user_id == "user123"
        
        doc = create_document_source("doc.pdf", "content")
        assert isinstance(doc, DocumentSource)
        assert doc.doc_uri == "doc.pdf"
        assert doc.content == "content"
        
        trace = create_trace_source("trace-123", "span-456")
        assert isinstance(trace, TraceSource)
        assert trace.trace_id == "trace-123"
        assert trace.span_id == "span-456"

    def test_get_source_summary(self):
        """Test get_source_summary utility function."""
        human = HumanSource("user123")
        assert get_source_summary(human) == "Human annotator: user123"
        
        doc = DocumentSource("doc.pdf", "This is a long content that should be truncated...")
        summary = get_source_summary(doc)
        assert "doc.pdf" in summary
        assert "This is a long content that should be truncated" in summary
        
        trace = TraceSource("trace-123", "span-456")
        assert get_source_summary(trace) == "Trace: trace-123, Span: span-456"
        
        trace_no_span = TraceSource("trace-123")
        assert get_source_summary(trace_no_span) == "Trace: trace-123"


class TestDatasetRecord:
    """Test DatasetRecord entity."""

    def create_sample_record(self):
        """Create a sample DatasetRecord for testing."""
        inputs = [
            InputValue("question", "What is MLflow?"),
            InputValue("context", "MLflow documentation")
        ]
        expectations = {
            "expected_answer": ExpectationValue("MLflow is an ML lifecycle platform"),
            "confidence": ExpectationValue(0.95)
        }
        tags = {"category": "qa", "difficulty": "easy"}
        source = HumanSource("annotator1")
        
        return DatasetRecord.create_new(
            dataset_id="dataset-123",
            inputs={"question": "What is MLflow?", "context": "MLflow documentation"},
            expectations={"expected_answer": "MLflow is an ML lifecycle platform", "confidence": 0.95},
            tags=tags,
            source=source,
            created_by="user1"
        )

    def test_dataset_record_creation(self):
        """Test DatasetRecord creation and property access."""
        record = self.create_sample_record()
        
        assert record.dataset_id == "dataset-123"
        assert len(record.inputs) == 2
        assert len(record.expectations) == 2
        assert record.tags["category"] == "qa"
        assert isinstance(record.source, HumanSource)

    def test_dataset_record_protobuf_conversion(self):
        """Test DatasetRecord protobuf serialization/deserialization."""
        record = self.create_sample_record()
        
        proto = record.to_proto()
        record_from_proto = DatasetRecord.from_proto(proto)
        
        assert record.dataset_record_id == record_from_proto.dataset_record_id
        assert record.dataset_id == record_from_proto.dataset_id
        assert len(record_from_proto.inputs) == 2
        assert len(record_from_proto.expectations) == 2

    def test_dataset_record_dict_conversion(self):
        """Test DatasetRecord dictionary serialization/deserialization."""
        record = self.create_sample_record()
        
        record_dict = record.to_dict()
        record_from_dict = DatasetRecord.from_dict(record_dict)
        
        assert record.dataset_record_id == record_from_dict.dataset_record_id
        assert record.dataset_id == record_from_dict.dataset_id
        assert len(record_from_dict.inputs) == 2
        assert len(record_from_dict.expectations) == 2

    def test_dataset_record_utility_methods(self):
        """Test DatasetRecord utility methods."""
        record = self.create_sample_record()
        
        # Test get_input_value
        question = record.get_input_value("question")
        assert question == "What is MLflow?"
        
        # Test get_expectation_value
        expected_answer = record.get_expectation_value("expected_answer")
        assert expected_answer == "MLflow is an ML lifecycle platform"
        
        # Test add_input
        record.add_input("additional_context", "More context")
        assert record.get_input_value("additional_context") == "More context"
        
        # Test add_expectation
        record.add_expectation("max_length", 100)
        assert record.get_expectation_value("max_length") == 100
        
        # Test add_tag
        record.add_tag("source", "manual")
        assert record.tags["source"] == "manual"


class TestManagedDataset:
    """Test ManagedDataset entity."""

    def create_sample_dataset(self):
        """Create a sample ManagedDataset for testing."""
        return ManagedDataset.create_new(
            name="QA Evaluation Dataset",
            experiment_ids=["exp-123", "exp-456"],
            source_type="human",
            source="manual_annotation",
            digest="abc123",
            schema='{"inputs": ["question", "context"], "expectations": ["answer"]}',
            profile='{"num_records": 100, "avg_length": 50}',
            created_by="data_scientist"
        )

    def test_managed_dataset_creation(self):
        """Test ManagedDataset creation and property access."""
        dataset = self.create_sample_dataset()
        
        assert dataset.name == "QA Evaluation Dataset"
        assert dataset.experiment_ids == ["exp-123", "exp-456"]
        assert dataset.source_type == "human"
        assert dataset.source == "manual_annotation"
        assert dataset.digest == "abc123"
        assert '"num_records": 100' in dataset.profile
        assert dataset.created_by == "data_scientist"
        assert len(dataset.records) == 0

    def test_managed_dataset_protobuf_conversion(self):
        """Test ManagedDataset protobuf serialization/deserialization."""
        dataset = self.create_sample_dataset()
        
        proto = dataset.to_proto()
        dataset_from_proto = ManagedDataset.from_proto(proto)
        
        assert dataset.dataset_id == dataset_from_proto.dataset_id
        assert dataset.name == dataset_from_proto.name
        assert dataset.experiment_ids == dataset_from_proto.experiment_ids
        assert dataset.source_type == dataset_from_proto.source_type

    def test_managed_dataset_dict_conversion(self):
        """Test ManagedDataset dictionary serialization/deserialization."""
        dataset = self.create_sample_dataset()
        
        dataset_dict = dataset.to_dict()
        dataset_from_dict = ManagedDataset.from_dict(dataset_dict)
        
        assert dataset.dataset_id == dataset_from_dict.dataset_id
        assert dataset.name == dataset_from_dict.name
        assert dataset.experiment_ids == dataset_from_dict.experiment_ids

    def test_managed_dataset_set_profile_immutable(self):
        """Test ManagedDataset set_profile returns new instance."""
        dataset = self.create_sample_dataset()
        original_id = dataset.dataset_id
        
        new_profile = '{"updated": true, "num_records": 150}'
        updated_dataset = dataset.set_profile(new_profile)
        
        # Should be a new instance
        assert updated_dataset is not dataset
        assert updated_dataset.dataset_id == original_id  # Same ID
        assert updated_dataset.profile == new_profile
        assert dataset.profile != new_profile  # Original unchanged

    def test_managed_dataset_merge_records_from_list(self):
        """Test ManagedDataset merge_records with list of dicts."""
        dataset = self.create_sample_dataset()
        
        records_to_merge = [
            {
                "inputs": {"question": "What is MLflow?", "context": "docs"},
                "expectations": {"answer": "ML platform", "confidence": 0.9},
                "tags": {"category": "basic"}
            },
            {
                "inputs": {"question": "How to log models?", "context": "tutorial"},
                "expectations": {"answer": "Use mlflow.log_model", "confidence": 0.95},
                "tags": {"category": "advanced"}
            }
        ]
        
        updated_dataset = dataset.merge_records(records_to_merge)
        
        # Should be a new instance
        assert updated_dataset is not dataset
        assert len(updated_dataset.records) == 2
        assert len(dataset.records) == 0  # Original unchanged
        
        # Check record content
        first_record = updated_dataset.records[0]
        assert first_record.get_input_value("question") == "What is MLflow?"
        assert first_record.get_expectation_value("answer") == "ML platform"

    def test_managed_dataset_merge_records_from_dataframe(self):
        """Test ManagedDataset merge_records with pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        dataset = self.create_sample_dataset()
        
        df = pd.DataFrame([
            {
                "input_question": "What is MLflow?",
                "input_context": "documentation",
                "expected_answer": "ML platform",
                "expected_confidence": 0.9,
                "tag_category": "basic"
            },
            {
                "input_question": "How to track experiments?",
                "input_context": "tutorial",
                "expected_answer": "Use mlflow.start_run()",
                "expected_confidence": 0.95,
                "tag_category": "intermediate"
            }
        ])
        
        updated_dataset = dataset.merge_records(df)
        
        assert len(updated_dataset.records) == 2
        first_record = updated_dataset.records[0]
        assert first_record.get_input_value("question") == "What is MLflow?"
        assert first_record.get_expectation_value("answer") == "ML platform"
        assert first_record.tags["category"] == "basic"

    def test_managed_dataset_to_df(self):
        """Test ManagedDataset to_df conversion."""
        pytest.importorskip("pandas")
        
        dataset = self.create_sample_dataset()
        
        # Add some records
        records_to_merge = [
            {
                "inputs": {"question": "What is MLflow?", "context": "docs"},
                "expectations": {"answer": "ML platform", "confidence": 0.9},
                "tags": {"category": "basic"}
            }
        ]
        
        dataset_with_records = dataset.merge_records(records_to_merge)
        df = dataset_with_records.to_df()
        
        assert len(df) == 1
        assert "input_question" in df.columns
        assert "input_context" in df.columns
        assert "expected_answer" in df.columns
        assert "expected_confidence" in df.columns
        assert "tag_category" in df.columns
        
        row = df.iloc[0]
        assert row["input_question"] == "What is MLflow?"
        assert row["expected_answer"] == "ML platform"
        assert row["tag_category"] == "basic"

    def test_managed_dataset_to_df_empty(self):
        """Test ManagedDataset to_df with no records."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        dataset = self.create_sample_dataset()
        df = dataset.to_df()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_managed_dataset_record_deduplication(self):
        """Test ManagedDataset record deduplication during merge."""
        dataset = self.create_sample_dataset()
        
        # Same record added twice
        record_data = {
            "inputs": {"question": "What is MLflow?", "context": "docs"},
            "expectations": {"answer": "ML platform", "confidence": 0.9},
            "tags": {"category": "basic"}
        }
        
        dataset_with_one = dataset.merge_records([record_data])
        dataset_with_duplicates = dataset_with_one.merge_records([record_data])
        
        # Should still have only one record due to deduplication
        assert len(dataset_with_duplicates.records) == 1

    def test_managed_dataset_timestamp_management(self):
        """Test ManagedDataset timestamp management."""
        dataset = self.create_sample_dataset()
        
        original_created_time = dataset.created_time
        original_update_time = dataset.last_update_time
        
        # Small delay to ensure different timestamp
        time.sleep(0.01)
        
        # Update dataset
        updated_dataset = dataset.set_profile('{"updated": true}')
        
        # Created time should remain the same, update time should change
        assert updated_dataset.created_time == original_created_time
        assert updated_dataset.last_update_time > original_update_time

    def test_managed_dataset_error_handling(self):
        """Test ManagedDataset error handling scenarios."""
        # Test missing pandas for to_df
        with patch.dict('sys.modules', {'pandas': None}):
            dataset = self.create_sample_dataset()
            with pytest.raises(ImportError, match="pandas is required"):
                dataset.to_df()


class TestManagedDatasetIntegration:
    """Integration tests for managed dataset entities."""

    def test_full_workflow(self):
        """Test a complete workflow with all entities."""
        # Create dataset
        dataset = ManagedDataset.create_new(
            name="Full Workflow Test",
            experiment_ids=["exp-123"],
            source_type="mixed",
            created_by="test_user"
        )
        
        # Create records with different sources
        human_record = DatasetRecord.create_new(
            dataset_id=dataset.dataset_id,
            inputs={"question": "Human annotated question"},
            expectations={"answer": "Human provided answer"},
            source=create_human_source("annotator1"),
            created_by="annotator1"
        )
        
        trace_record = DatasetRecord.create_new(
            dataset_id=dataset.dataset_id,
            inputs={"question": "Trace derived question"},
            expectations={"answer": "Trace derived answer"},
            source=create_trace_source("trace-123", "span-456"),
            created_by="trace_processor"
        )
        
        # Merge records using the immutable pattern
        dataset_with_records = dataset.merge_records([
            human_record.to_dict(),
            trace_record.to_dict()
        ])
        
        # Verify final state
        assert len(dataset_with_records.records) == 2
        
        # Test serialization round-trip
        dataset_dict = dataset_with_records.to_dict()
        restored_dataset = ManagedDataset.from_dict(dataset_dict)
        
        assert restored_dataset.name == "Full Workflow Test"
        assert len(restored_dataset.records) == 2
        
        # Verify sources are preserved
        sources = [record.source for record in restored_dataset.records]
        source_types = [source.source_type for source in sources]
        assert "human" in source_types
        assert "trace" in source_types

    def test_json_serialization_compatibility(self):
        """Test JSON serialization for storage compatibility."""
        dataset = ManagedDataset.create_new(
            name="JSON Test Dataset",
            experiment_ids=["exp-123"],
            created_by="test_user"
        )
        
        # Add a record
        dataset_with_records = dataset.merge_records([
            {
                "inputs": {"question": "Test question"},
                "expectations": {"answer": "Test answer"},
                "tags": {"test": "true"}
            }
        ])
        
        # Convert to dict and serialize to JSON
        dataset_dict = dataset_with_records.to_dict()
        json_str = json.dumps(dataset_dict, indent=2)
        
        # Deserialize and restore
        restored_dict = json.loads(json_str)
        restored_dataset = ManagedDataset.from_dict(restored_dict)
        
        # Verify integrity
        assert restored_dataset.name == "JSON Test Dataset"
        assert len(restored_dataset.records) == 1
        assert restored_dataset.records[0].get_input_value("question") == "Test question"