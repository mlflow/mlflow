import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LabelSchemaType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LABEL_SCHEMA_TYPE_UNSPECIFIED: _ClassVar[LabelSchemaType]
    FEEDBACK: _ClassVar[LabelSchemaType]
    EXPECTATION: _ClassVar[LabelSchemaType]

class CategoricalSemanticPolarity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CATEGORICAL_SEMANTIC_POLARITY_UNSPECIFIED: _ClassVar[CategoricalSemanticPolarity]
    ASCENDING: _ClassVar[CategoricalSemanticPolarity]
    DESCENDING: _ClassVar[CategoricalSemanticPolarity]
LABEL_SCHEMA_TYPE_UNSPECIFIED: LabelSchemaType
FEEDBACK: LabelSchemaType
EXPECTATION: LabelSchemaType
CATEGORICAL_SEMANTIC_POLARITY_UNSPECIFIED: CategoricalSemanticPolarity
ASCENDING: CategoricalSemanticPolarity
DESCENDING: CategoricalSemanticPolarity

class InputPassFail(_message.Message):
    __slots__ = ("positive_label", "negative_label")
    POSITIVE_LABEL_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_LABEL_FIELD_NUMBER: _ClassVar[int]
    positive_label: str
    negative_label: str
    def __init__(self, positive_label: _Optional[str] = ..., negative_label: _Optional[str] = ...) -> None: ...

class InputCategorical(_message.Message):
    __slots__ = ("options", "semantic_polarity", "multi_select")
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_POLARITY_FIELD_NUMBER: _ClassVar[int]
    MULTI_SELECT_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedScalarFieldContainer[str]
    semantic_polarity: CategoricalSemanticPolarity
    multi_select: bool
    def __init__(self, options: _Optional[_Iterable[str]] = ..., semantic_polarity: _Optional[_Union[CategoricalSemanticPolarity, str]] = ..., multi_select: bool = ...) -> None: ...

class InputNumeric(_message.Message):
    __slots__ = ("min_value", "max_value")
    MIN_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAX_VALUE_FIELD_NUMBER: _ClassVar[int]
    min_value: float
    max_value: float
    def __init__(self, min_value: _Optional[float] = ..., max_value: _Optional[float] = ...) -> None: ...

class LabelSchemaInput(_message.Message):
    __slots__ = ("pass_fail", "categorical", "numeric")
    PASS_FAIL_FIELD_NUMBER: _ClassVar[int]
    CATEGORICAL_FIELD_NUMBER: _ClassVar[int]
    NUMERIC_FIELD_NUMBER: _ClassVar[int]
    pass_fail: InputPassFail
    categorical: InputCategorical
    numeric: InputNumeric
    def __init__(self, pass_fail: _Optional[_Union[InputPassFail, _Mapping]] = ..., categorical: _Optional[_Union[InputCategorical, _Mapping]] = ..., numeric: _Optional[_Union[InputNumeric, _Mapping]] = ...) -> None: ...

class LabelSchema(_message.Message):
    __slots__ = ("schema_id", "experiment_id", "name", "type", "title", "instruction", "enable_comment", "input", "created_by", "created_at", "last_updated_at")
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_COMMENT_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    schema_id: str
    experiment_id: str
    name: str
    type: LabelSchemaType
    title: str
    instruction: str
    enable_comment: bool
    input: LabelSchemaInput
    created_by: str
    created_at: int
    last_updated_at: int
    def __init__(self, schema_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., type: _Optional[_Union[LabelSchemaType, str]] = ..., title: _Optional[str] = ..., instruction: _Optional[str] = ..., enable_comment: bool = ..., input: _Optional[_Union[LabelSchemaInput, _Mapping]] = ..., created_by: _Optional[str] = ..., created_at: _Optional[int] = ..., last_updated_at: _Optional[int] = ...) -> None: ...

class CreateLabelSchema(_message.Message):
    __slots__ = ("experiment_id", "name", "type", "title", "input", "instruction", "enable_comment")
    class Response(_message.Message):
        __slots__ = ("label_schema",)
        LABEL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        label_schema: LabelSchema
        def __init__(self, label_schema: _Optional[_Union[LabelSchema, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_COMMENT_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    type: LabelSchemaType
    title: str
    input: LabelSchemaInput
    instruction: str
    enable_comment: bool
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., type: _Optional[_Union[LabelSchemaType, str]] = ..., title: _Optional[str] = ..., input: _Optional[_Union[LabelSchemaInput, _Mapping]] = ..., instruction: _Optional[str] = ..., enable_comment: bool = ...) -> None: ...

class GetLabelSchema(_message.Message):
    __slots__ = ("schema_id",)
    class Response(_message.Message):
        __slots__ = ("label_schema",)
        LABEL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        label_schema: LabelSchema
        def __init__(self, label_schema: _Optional[_Union[LabelSchema, _Mapping]] = ...) -> None: ...
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    schema_id: str
    def __init__(self, schema_id: _Optional[str] = ...) -> None: ...

class GetLabelSchemaByName(_message.Message):
    __slots__ = ("experiment_id", "name")
    class Response(_message.Message):
        __slots__ = ("label_schema",)
        LABEL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        label_schema: LabelSchema
        def __init__(self, label_schema: _Optional[_Union[LabelSchema, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ListLabelSchemas(_message.Message):
    __slots__ = ("experiment_id", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("label_schemas", "next_page_token")
        LABEL_SCHEMAS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        label_schemas: _containers.RepeatedCompositeFieldContainer[LabelSchema]
        next_page_token: str
        def __init__(self, label_schemas: _Optional[_Iterable[_Union[LabelSchema, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    max_results: int
    page_token: str
    def __init__(self, experiment_id: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class UpdateLabelSchema(_message.Message):
    __slots__ = ("schema_id", "name", "title", "instruction", "enable_comment", "input")
    class Response(_message.Message):
        __slots__ = ("label_schema",)
        LABEL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        label_schema: LabelSchema
        def __init__(self, label_schema: _Optional[_Union[LabelSchema, _Mapping]] = ...) -> None: ...
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_COMMENT_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    schema_id: str
    name: str
    title: str
    instruction: str
    enable_comment: bool
    input: LabelSchemaInput
    def __init__(self, schema_id: _Optional[str] = ..., name: _Optional[str] = ..., title: _Optional[str] = ..., instruction: _Optional[str] = ..., enable_comment: bool = ..., input: _Optional[_Union[LabelSchemaInput, _Mapping]] = ...) -> None: ...

class UpsertLabelSchema(_message.Message):
    __slots__ = ("experiment_id", "name", "type", "title", "input", "instruction", "enable_comment")
    class Response(_message.Message):
        __slots__ = ("label_schema",)
        LABEL_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        label_schema: LabelSchema
        def __init__(self, label_schema: _Optional[_Union[LabelSchema, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_COMMENT_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    type: LabelSchemaType
    title: str
    input: LabelSchemaInput
    instruction: str
    enable_comment: bool
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., type: _Optional[_Union[LabelSchemaType, str]] = ..., title: _Optional[str] = ..., input: _Optional[_Union[LabelSchemaInput, _Mapping]] = ..., instruction: _Optional[str] = ..., enable_comment: bool = ...) -> None: ...

class DeleteLabelSchema(_message.Message):
    __slots__ = ("schema_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    schema_id: str
    def __init__(self, schema_id: _Optional[str] = ...) -> None: ...
