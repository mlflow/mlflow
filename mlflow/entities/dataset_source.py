from typing import Optional, Dict, Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.managed_datasets_pb2 import (
    DatasetRecordSource as ProtoDatasetRecordSource,
    HumanSource as ProtoHumanSource,
    DocumentSource as ProtoDocumentSource,
    TraceSource as ProtoTraceSource,
)


class DatasetRecordSource(_MlflowObject):
    """
    Base class for dataset record sources indicating the origin of a dataset record.
    
    This is a union type that can represent human, document, or trace sources.
    """

    def __init__(self, source_type: str, source_data: Dict[str, Any]) -> None:
        self._source_type = source_type
        self._source_data = source_data

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def source_type(self) -> str:
        """The type of source ('human', 'document', or 'trace')."""
        return self._source_type

    @property
    def source_data(self) -> Dict[str, Any]:
        """The source-specific data."""
        return self._source_data

    def to_proto(self) -> ProtoDatasetRecordSource:
        """Convert this entity to a protobuf message."""
        source = ProtoDatasetRecordSource()
        
        if self.source_type == "human":
            human_source = ProtoHumanSource()
            if "user_id" in self.source_data:
                human_source.user_id = self.source_data["user_id"]
            source.human.CopyFrom(human_source)
        elif self.source_type == "document":
            doc_source = ProtoDocumentSource()
            if "doc_uri" in self.source_data:
                doc_source.doc_uri = self.source_data["doc_uri"]
            if "content" in self.source_data:
                doc_source.content = self.source_data["content"]
            source.document.CopyFrom(doc_source)
        elif self.source_type == "trace":
            trace_source = ProtoTraceSource()
            if "trace_id" in self.source_data:
                trace_source.trace_id = self.source_data["trace_id"]
            if "span_id" in self.source_data:
                trace_source.span_id = self.source_data["span_id"]
            source.trace.CopyFrom(trace_source)
        
        return source

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecordSource) -> "DatasetRecordSource":
        """Create an entity from a protobuf message."""
        which = proto.WhichOneof("source_type")
        
        if which == "human":
            source_data = {}
            if proto.human.HasField("user_id"):
                source_data["user_id"] = proto.human.user_id
            return cls("human", source_data)
        elif which == "document":
            source_data = {}
            if proto.document.HasField("doc_uri"):
                source_data["doc_uri"] = proto.document.doc_uri
            if proto.document.HasField("content"):
                source_data["content"] = proto.document.content
            return cls("document", source_data)
        elif which == "trace":
            source_data = {}
            if proto.trace.HasField("trace_id"):
                source_data["trace_id"] = proto.trace.trace_id
            if proto.trace.HasField("span_id"):
                source_data["span_id"] = proto.trace.span_id
            return cls("trace", source_data)
        else:
            raise ValueError(f"Unknown source type in proto: {which}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "source_type": self.source_type,
            "source_data": self.source_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecordSource":
        """Create an entity from a dictionary."""
        return cls(
            source_type=data["source_type"],
            source_data=data["source_data"],
        )


class HumanSource(DatasetRecordSource):
    """Records that were manually created or annotated by humans."""

    def __init__(self, user_id: str) -> None:
        super().__init__("human", {"user_id": user_id})

    @property
    def user_id(self) -> str:
        """Identifier for the human annotator (e.g., username, email, user_id)."""
        return self.source_data["user_id"]

    @classmethod
    def create(cls, user_id: str) -> "HumanSource":
        """Create a new human source."""
        return cls(user_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "source_type": "human",
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanSource":
        """Create an entity from a dictionary."""
        return cls(user_id=data["user_id"])


class DocumentSource(DatasetRecordSource):
    """Records derived from processing documents or knowledge bases."""

    def __init__(self, doc_uri: str, content: Optional[str] = None) -> None:
        source_data = {"doc_uri": doc_uri}
        if content is not None:
            source_data["content"] = content
        super().__init__("document", source_data)

    @property
    def doc_uri(self) -> str:
        """URI or identifier of the source document."""
        return self.source_data["doc_uri"]

    @property
    def content(self) -> Optional[str]:
        """Optional document content or excerpt for reference."""
        return self.source_data.get("content")

    @classmethod
    def create(cls, doc_uri: str, content: Optional[str] = None) -> "DocumentSource":
        """Create a new document source."""
        return cls(doc_uri, content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        result = {
            "source_type": "document",
            "doc_uri": self.doc_uri,
        }
        if self.content is not None:
            result["content"] = self.content
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSource":
        """Create an entity from a dictionary."""
        return cls(doc_uri=data["doc_uri"], content=data.get("content"))


class TraceSource(DatasetRecordSource):
    """
    Records derived from MLflow traces, typically from model inference
    or evaluation runs that can be converted into evaluation datasets.
    """

    def __init__(self, trace_id: str, span_id: Optional[str] = None) -> None:
        source_data = {"trace_id": trace_id}
        if span_id is not None:
            source_data["span_id"] = span_id
        super().__init__("trace", source_data)

    @property
    def trace_id(self) -> str:
        """ID of the trace from which this record was derived."""
        return self.source_data["trace_id"]

    @property
    def span_id(self) -> Optional[str]:
        """Optional: specific span ID within the trace that generated this record."""
        return self.source_data.get("span_id")

    @classmethod
    def create(cls, trace_id: str, span_id: Optional[str] = None) -> "TraceSource":
        """Create a new trace source."""
        return cls(trace_id, span_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        result = {
            "source_type": "trace",
            "trace_id": self.trace_id,
        }
        if self.span_id is not None:
            result["span_id"] = self.span_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceSource":
        """Create an entity from a dictionary."""
        return cls(trace_id=data["trace_id"], span_id=data.get("span_id"))


# Factory functions for creating sources
def create_human_source(user_id: str) -> HumanSource:
    """Create a human source for manually created records."""
    return HumanSource.create(user_id)


def create_document_source(doc_uri: str, content: Optional[str] = None) -> DocumentSource:
    """Create a document source for records derived from documents."""
    return DocumentSource.create(doc_uri, content)


def create_trace_source(trace_id: str, span_id: Optional[str] = None) -> TraceSource:
    """Create a trace source for records derived from MLflow traces."""
    return TraceSource.create(trace_id, span_id)


def create_source_from_dict(data: Dict[str, Any]) -> DatasetRecordSource:
    """Create the appropriate source type from a dictionary."""
    source_type = data.get("source_type")
    
    if source_type == "human":
        return HumanSource.from_dict(data)
    elif source_type == "document":
        return DocumentSource.from_dict(data)
    elif source_type == "trace":
        return TraceSource.from_dict(data)
    else:
        # Fallback to generic source
        return DatasetRecordSource.from_dict(data)


def create_source_from_proto(proto: ProtoDatasetRecordSource) -> DatasetRecordSource:
    """Create the appropriate source type from a protobuf message."""
    which = proto.WhichOneof("source_type")
    
    if which == "human":
        user_id = proto.human.user_id if proto.human.HasField("user_id") else ""
        return HumanSource.create(user_id)
    elif which == "document":
        doc_uri = proto.document.doc_uri if proto.document.HasField("doc_uri") else ""
        content = proto.document.content if proto.document.HasField("content") else None
        return DocumentSource.create(doc_uri, content)
    elif which == "trace":
        trace_id = proto.trace.trace_id if proto.trace.HasField("trace_id") else ""
        span_id = proto.trace.span_id if proto.trace.HasField("span_id") else None
        return TraceSource.create(trace_id, span_id)
    else:
        raise ValueError(f"Unknown source type in proto: {which}")


# Utility functions for working with sources
def get_source_summary(source: DatasetRecordSource) -> str:
    """Get a human-readable summary of the source."""
    if isinstance(source, HumanSource):
        return f"Human annotator: {source.user_id}"
    elif isinstance(source, DocumentSource):
        if source.content:
            content_preview = source.content[:50] + "..." if len(source.content) > 50 else source.content
            return f"Document: {source.doc_uri} (content: {content_preview})"
        else:
            return f"Document: {source.doc_uri}"
    elif isinstance(source, TraceSource):
        if source.span_id:
            return f"Trace: {source.trace_id}, Span: {source.span_id}"
        else:
            return f"Trace: {source.trace_id}"
    else:
        return f"Source: {source.source_type}"


def is_trace_source(source: DatasetRecordSource) -> bool:
    """Check if the source is a trace source."""
    return isinstance(source, TraceSource) or source.source_type == "trace"


def is_human_source(source: DatasetRecordSource) -> bool:
    """Check if the source is a human source."""
    return isinstance(source, HumanSource) or source.source_type == "human"


def is_document_source(source: DatasetRecordSource) -> bool:
    """Check if the source is a document source."""
    return isinstance(source, DocumentSource) or source.source_type == "document"