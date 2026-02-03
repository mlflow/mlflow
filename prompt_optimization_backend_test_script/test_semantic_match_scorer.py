#!/usr/bin/env python
"""
Quick test to verify SemanticMatch scorer returns continuous scores (0.0-1.0).

Usage:
    python prompt_optimization_backend_test_script/test_semantic_match_scorer.py
"""

from mlflow.genai.scorers import SemanticMatch


def test_semantic_match_returns_float():
    """Test that SemanticMatch returns a float score in 0.0-1.0 range."""
    scorer = SemanticMatch(model="openai:/gpt-4o-mini")

    # Test with semantically equivalent responses
    result_equivalent = scorer(
        inputs={"question": "What is 2 + 2?"},
        outputs="The answer is 4.",
        expectations={"expected_response": "2 + 2 equals 4"},
    )

    print("Equivalent responses test:")
    print(f"  Value: {result_equivalent.value} (type: {type(result_equivalent.value).__name__})")
    print(f"  Rationale: {result_equivalent.rationale[:150]}...")

    # Test with partially different responses
    result_partial = scorer(
        inputs={"question": "Explain photosynthesis"},
        outputs="Plants use sunlight to make food.",
        expectations={
            "expected_response": "Photosynthesis is a complex process where plants "
            "use chlorophyll to convert light energy, water, and CO2 into glucose and oxygen."
        },
    )

    print("\nPartially equivalent responses test:")
    print(f"  Value: {result_partial.value} (type: {type(result_partial.value).__name__})")
    print(f"  Rationale: {result_partial.rationale[:150]}...")

    # Test with completely different responses
    result_different = scorer(
        inputs={"question": "What is the capital of France?"},
        outputs="The Eiffel Tower is tall.",
        expectations={"expected_response": "Paris is the capital of France."},
    )

    print("\nDifferent responses test:")
    print(f"  Value: {result_different.value} (type: {type(result_different.value).__name__})")
    print(f"  Rationale: {result_different.rationale[:150]}...")

    # Verify scores are floats in expected range
    for name, result in [
        ("equivalent", result_equivalent),
        ("partial", result_partial),
        ("different", result_different),
    ]:
        assert isinstance(
            result.value, float
        ), f"{name}: Expected float, got {type(result.value).__name__}"
        assert (
            0.0 <= result.value <= 1.0
        ), f"{name}: Expected 0.0-1.0, got {result.value}"

    # Verify ordering makes sense (equivalent > partial > different)
    print("\nScore ordering check:")
    print(f"  Equivalent: {result_equivalent.value:.2f}")
    print(f"  Partial: {result_partial.value:.2f}")
    print(f"  Different: {result_different.value:.2f}")

    # Soft check - these should generally be in order
    if result_equivalent.value >= result_partial.value >= result_different.value:
        print("  ✓ Scores are in expected order")
    else:
        print("  ⚠ Scores not in expected order (this can happen with LLM variance)")

    # Check that the stricter rubric produces reasonable scores
    print("\nStrictness check:")
    if result_equivalent.value < 0.95:
        print(f"  ✓ Equivalent score ({result_equivalent.value:.2f}) is not too lenient (< 0.95)")
    else:
        print(f"  ⚠ Equivalent score ({result_equivalent.value:.2f}) might be too lenient")

    if result_partial.value < 0.70:
        print(f"  ✓ Partial score ({result_partial.value:.2f}) is appropriately strict (< 0.70)")
    else:
        print(f"  ⚠ Partial score ({result_partial.value:.2f}) might be too lenient")

    print("\n✓ All tests passed! SemanticMatch returns float scores 0.0-1.0.")


if __name__ == "__main__":
    test_semantic_match_returns_float()
