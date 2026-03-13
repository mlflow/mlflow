import databricks_pb2 as _databricks_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Issue(_message.Message):
    __slots__ = ("issue_id", "experiment_id", "name", "description", "status", "severity", "root_causes", "source_run_id", "created_timestamp", "last_updated_timestamp", "created_by")
    ISSUE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    ROOT_CAUSES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    issue_id: str
    experiment_id: str
    name: str
    description: str
    status: str
    severity: str
    root_causes: _containers.RepeatedScalarFieldContainer[str]
    source_run_id: str
    created_timestamp: int
    last_updated_timestamp: int
    created_by: str
    def __init__(self, issue_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., status: _Optional[str] = ..., severity: _Optional[str] = ..., root_causes: _Optional[_Iterable[str]] = ..., source_run_id: _Optional[str] = ..., created_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ..., created_by: _Optional[str] = ...) -> None: ...

class CreateIssue(_message.Message):
    __slots__ = ("experiment_id", "name", "description", "status", "severity", "root_causes", "source_run_id", "created_by")
    class Response(_message.Message):
        __slots__ = ("issue",)
        ISSUE_FIELD_NUMBER: _ClassVar[int]
        issue: Issue
        def __init__(self, issue: _Optional[_Union[Issue, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    ROOT_CAUSES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    description: str
    status: str
    severity: str
    root_causes: _containers.RepeatedScalarFieldContainer[str]
    source_run_id: str
    created_by: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., status: _Optional[str] = ..., severity: _Optional[str] = ..., root_causes: _Optional[_Iterable[str]] = ..., source_run_id: _Optional[str] = ..., created_by: _Optional[str] = ...) -> None: ...

class UpdateIssue(_message.Message):
    __slots__ = ("issue_id", "name", "description", "status", "severity")
    class Response(_message.Message):
        __slots__ = ("issue",)
        ISSUE_FIELD_NUMBER: _ClassVar[int]
        issue: Issue
        def __init__(self, issue: _Optional[_Union[Issue, _Mapping]] = ...) -> None: ...
    ISSUE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    issue_id: str
    name: str
    description: str
    status: str
    severity: str
    def __init__(self, issue_id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., status: _Optional[str] = ..., severity: _Optional[str] = ...) -> None: ...

class GetIssue(_message.Message):
    __slots__ = ("issue_id",)
    class Response(_message.Message):
        __slots__ = ("issue",)
        ISSUE_FIELD_NUMBER: _ClassVar[int]
        issue: Issue
        def __init__(self, issue: _Optional[_Union[Issue, _Mapping]] = ...) -> None: ...
    ISSUE_ID_FIELD_NUMBER: _ClassVar[int]
    issue_id: str
    def __init__(self, issue_id: _Optional[str] = ...) -> None: ...

class SearchIssues(_message.Message):
    __slots__ = ("experiment_id", "filter_string", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("issues", "next_page_token")
        ISSUES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        issues: _containers.RepeatedCompositeFieldContainer[Issue]
        next_page_token: str
        def __init__(self, issues: _Optional[_Iterable[_Union[Issue, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    FILTER_STRING_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    filter_string: str
    max_results: int
    page_token: str
    def __init__(self, experiment_id: _Optional[str] = ..., filter_string: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...
