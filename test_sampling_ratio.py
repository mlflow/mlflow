# REMOVE THIS ONCE READY FOR REVIEW
# ruff: noqa: T201
# clint: noqa
"""
Test script for per-decorator sampling ratio feature.

This script tests the following scenarios:
1. sampling_ratio=1.0 - all traces should be sampled
2. sampling_ratio=0.0 - no traces should be sampled
3. sampling_ratio=0.5 - approximately 50% of traces should be sampled
4. Nested functions - child spans follow parent's sampling decision
5. Default behavior - uses global sampler when sampling_ratio is None
"""

import mlflow


def test_sampling_ratio_100_percent():
    """Test that sampling_ratio=1.0 samples all traces."""
    print("=" * 60)
    print("Test 1: sampling_ratio=1.0 (100% sampling)")
    print("=" * 60)

    trace_ids = []

    @mlflow.trace(sampling_ratio=1.0)
    def always_traced():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_ids.append(trace_id)
        return "always traced"

    # Call 10 times
    for _ in range(10):
        always_traced()

    print(f"Called 10 times, got {len(trace_ids)} trace IDs")
    assert len(trace_ids) == 10, f"Expected 10 traces, got {len(trace_ids)}"
    print("PASSED!\n")


def test_sampling_ratio_0_percent():
    """Test that sampling_ratio=0.0 samples no traces."""
    print("=" * 60)
    print("Test 2: sampling_ratio=0.0 (0% sampling)")
    print("=" * 60)

    trace_ids = []

    @mlflow.trace(sampling_ratio=0.0)
    def never_traced():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_ids.append(trace_id)
        return "never traced"

    # Call 10 times
    for _ in range(10):
        result = never_traced()
        assert result == "never traced", "Function should still return its value"

    print(f"Called 10 times, got {len(trace_ids)} trace IDs")
    assert len(trace_ids) == 0, f"Expected 0 traces, got {len(trace_ids)}"
    print("PASSED!\n")


def test_sampling_ratio_50_percent():
    """Test that sampling_ratio=0.5 samples approximately 50% of traces."""
    print("=" * 60)
    print("Test 3: sampling_ratio=0.5 (50% sampling)")
    print("=" * 60)

    trace_ids = []

    @mlflow.trace(sampling_ratio=0.5)
    def half_traced():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_ids.append(trace_id)
        return "half traced"

    # Call 100 times for statistical significance
    num_calls = 100
    for _ in range(num_calls):
        half_traced()

    sample_rate = len(trace_ids) / num_calls
    print(f"Called {num_calls} times, got {len(trace_ids)} traces ({sample_rate:.1%})")

    # Allow for some variance (between 30% and 70%)
    assert 30 <= len(trace_ids) <= 70, f"Expected ~50 traces, got {len(trace_ids)}"
    print("PASSED!\n")


def test_nested_functions():
    """Test that child spans follow parent's sampling decision."""
    print("=" * 60)
    print("Test 4: Nested functions (child follows parent)")
    print("=" * 60)

    outer_trace_ids = []
    inner_trace_ids = []

    @mlflow.trace(sampling_ratio=1.0)
    def outer():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            outer_trace_ids.append(trace_id)
        return inner()

    @mlflow.trace(sampling_ratio=0.0)  # This should be ignored when called from outer
    def inner():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            inner_trace_ids.append(trace_id)
        return "inner result"

    # Call outer 5 times - inner should also be traced even though it has sampling_ratio=0.0
    for _ in range(5):
        outer()

    print("Called outer() 5 times")
    print(f"  outer_trace_ids: {len(outer_trace_ids)}")
    print(f"  inner_trace_ids: {len(inner_trace_ids)}")

    assert len(outer_trace_ids) == 5, f"Expected 5 outer traces, got {len(outer_trace_ids)}"
    assert len(inner_trace_ids) == 5, f"Expected 5 inner traces, got {len(inner_trace_ids)}"

    # Verify they share the same trace ID (same trace)
    for i in range(5):
        assert outer_trace_ids[i] == inner_trace_ids[i], (
            f"Outer and inner should share trace ID at index {i}"
        )

    print("PASSED!\n")


def test_nested_not_sampled():
    """Test that when parent is not sampled, child is also not traced."""
    print("=" * 60)
    print("Test 5: Nested functions (parent not sampled)")
    print("=" * 60)

    outer_trace_ids = []
    inner_trace_ids = []

    @mlflow.trace(sampling_ratio=0.0)
    def outer_not_sampled():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            outer_trace_ids.append(trace_id)
        return inner_always()

    @mlflow.trace(sampling_ratio=1.0)  # This should also not trace when called from outer
    def inner_always():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            inner_trace_ids.append(trace_id)
        return "inner result"

    # Call outer 5 times - nothing should be traced
    for _ in range(5):
        result = outer_not_sampled()
        assert result == "inner result", "Function should still return its value"

    print("Called outer_not_sampled() 5 times")
    print(f"  outer_trace_ids: {len(outer_trace_ids)}")
    print(f"  inner_trace_ids: {len(inner_trace_ids)}")

    assert len(outer_trace_ids) == 0, f"Expected 0 outer traces, got {len(outer_trace_ids)}"
    assert len(inner_trace_ids) == 0, f"Expected 0 inner traces, got {len(inner_trace_ids)}"
    print("PASSED!\n")


def test_default_behavior():
    """Test that default behavior (sampling_ratio=None) uses global sampler."""
    print("=" * 60)
    print("Test 6: Default behavior (uses global sampler)")
    print("=" * 60)

    trace_ids = []

    @mlflow.trace()  # No sampling_ratio - should use global setting (default 1.0)
    def default_traced():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_ids.append(trace_id)
        return "default traced"

    # Call 5 times
    for _ in range(5):
        default_traced()

    print(f"Called 5 times, got {len(trace_ids)} trace IDs")
    assert len(trace_ids) == 5, f"Expected 5 traces, got {len(trace_ids)}"
    print("PASSED!\n")


def test_generator_function():
    """Test that sampling works for generator functions."""
    print("=" * 60)
    print("Test 7: Generator function with sampling_ratio")
    print("=" * 60)

    traced_gen_ids = []
    untraced_gen_ids = []

    @mlflow.trace(sampling_ratio=1.0)
    def traced_generator():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            traced_gen_ids.append(trace_id)
        for i in range(3):
            yield i

    @mlflow.trace(sampling_ratio=0.0)
    def untraced_generator():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            untraced_gen_ids.append(trace_id)
        for i in range(3):
            yield i

    # Consume traced generator
    result1 = list(traced_generator())
    result2 = list(traced_generator())
    assert result1 == [0, 1, 2], "Generator should still yield values"
    assert result2 == [0, 1, 2], "Generator should still yield values"

    print(f"Called traced_generator() 2 times, got {len(traced_gen_ids)} trace IDs")
    assert len(traced_gen_ids) == 2, f"Expected 2 traces, got {len(traced_gen_ids)}"

    # Consume untraced generator
    result3 = list(untraced_generator())
    result4 = list(untraced_generator())
    assert result3 == [0, 1, 2], "Generator should still yield values"
    assert result4 == [0, 1, 2], "Generator should still yield values"

    print(f"Called untraced_generator() 2 times, got {len(untraced_gen_ids)} trace IDs")
    assert len(untraced_gen_ids) == 0, f"Expected 0 traces, got {len(untraced_gen_ids)}"

    print("PASSED!\n")


def test_mixed_sampling_scenario():
    """Test a realistic scenario with mixed sampling ratios."""
    print("=" * 60)
    print("Test 8: Mixed sampling scenario (realistic use case)")
    print("=" * 60)

    high_volume_traces = []
    critical_traces = []

    @mlflow.trace(sampling_ratio=0.1)  # 10% sampling for high-volume
    def high_volume_endpoint():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            high_volume_traces.append(trace_id)
        return "page loaded"

    @mlflow.trace(sampling_ratio=1.0)  # 100% sampling for critical
    def critical_transaction():
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            critical_traces.append(trace_id)
        return "transaction completed"

    # Simulate high-volume endpoint being called 100 times
    for _ in range(100):
        high_volume_endpoint()

    # Simulate critical transaction being called 10 times
    for _ in range(10):
        critical_transaction()

    high_volume_rate = len(high_volume_traces) / 100
    critical_rate = len(critical_traces) / 10

    print(f"High-volume endpoint: {len(high_volume_traces)}/100 traced ({high_volume_rate:.1%})")
    print(f"Critical transaction: {len(critical_traces)}/10 traced ({critical_rate:.1%})")

    # High volume should be roughly 10% (allow 2-25%)
    assert 2 <= len(high_volume_traces) <= 25, (
        f"Expected ~10 high-volume traces, got {len(high_volume_traces)}"
    )

    # Critical should be 100%
    assert len(critical_traces) == 10, f"Expected 10 critical traces, got {len(critical_traces)}"

    print("PASSED!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Per-Decorator Sampling Ratio Feature Tests")
    print("=" * 60 + "\n")

    test_sampling_ratio_100_percent()
    test_sampling_ratio_0_percent()
    test_sampling_ratio_50_percent()
    test_nested_functions()
    test_nested_not_sampled()
    test_default_behavior()
    test_generator_function()
    test_mixed_sampling_scenario()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
