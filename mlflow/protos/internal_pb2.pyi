from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class InputVertexType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN: _ClassVar[InputVertexType]
    DATASET: _ClassVar[InputVertexType]
    MODEL: _ClassVar[InputVertexType]

class OutputVertexType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN_OUTPUT: _ClassVar[OutputVertexType]
    MODEL_OUTPUT: _ClassVar[OutputVertexType]
RUN: InputVertexType
DATASET: InputVertexType
MODEL: InputVertexType
RUN_OUTPUT: OutputVertexType
MODEL_OUTPUT: OutputVertexType
