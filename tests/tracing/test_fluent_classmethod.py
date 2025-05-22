import pytest
import inspect

import mlflow
from mlflow.entities import SpanType


def test_trace_with_classmethod():
    class TestModel:
        @mlflow.trace
        @classmethod
        def predict(cls, x, y):
            return x + y

    result = TestModel.predict(1, 2)
    assert result == 3


def test_trace_with_classmethod_order_reversed():
    class TestModel:
        @classmethod
        @mlflow.trace
        def predict(cls, x, y):
            return x + y

    result = TestModel.predict(1, 2)
    assert result == 3


def test_trace_with_classmethod_with_params():
    class TestModel:
        @mlflow.trace(name="custom_predict", span_type=SpanType.MODEL_INFERENCE)
        @classmethod
        def predict(cls, x, y):
            return x + y

    result = TestModel.predict(1, 2)
    assert result == 3


def test_trace_with_staticmethod():
    class TestModel:
        @mlflow.trace
        @staticmethod
        def predict(x, y):
            return x + y

    result = TestModel.predict(1, 2)
    assert result == 3


def test_trace_classmethod_object_detection():
    """Test to verify we can detect if a function is a classmethod."""

    # Regular function
    def regular_function():
        pass

    # Method in a class
    class TestClass:
        def instance_method(self):
            pass

        @classmethod
        def class_method(cls):
            pass

        @staticmethod
        def static_method():
            pass

    # Verify our detection logic works
    assert not inspect.ismethod(regular_function)
    
    # Check an instance method (requires an instance)
    instance = TestClass()
    assert inspect.ismethod(instance.instance_method)
    
    # Class methods are methods too
    assert inspect.ismethod(TestClass.class_method)
    
    # Static methods aren't methods, they're functions
    assert not inspect.ismethod(TestClass.static_method)
    
    # The raw descriptors
    assert not inspect.ismethod(TestClass.__dict__["instance_method"])
    assert inspect.isfunction(TestClass.__dict__["instance_method"])
    
    # Class methods are not functions but descriptors
    assert not inspect.isfunction(TestClass.__dict__["class_method"])
    assert isinstance(TestClass.__dict__["class_method"], classmethod)
    
    # Static methods are descriptors too
    assert not inspect.isfunction(TestClass.__dict__["static_method"])
    assert isinstance(TestClass.__dict__["static_method"], staticmethod)