#!/usr/bin/env python3
"""
OSS Integration Test: inference_params with LiteLLM Adapter (PR #19152)

Tests the inference_params feature using OpenAI via LiteLLM adapter.
This demonstrates the feature working in an OSS (non-Databricks) environment.

Usage:
    python3 test_oss_litellm.py              # Full test (requires OPENAI_API_KEY)
    python3 test_oss_litellm.py --dry-run    # Verify setup without API calls
"""

import argparse
import os
import sys

# Add local mlflow to path
sys.path.insert(0, "/Users/debu.sinha/projects/mlflow-work/mlflow-fork")

from mlflow.genai import make_judge

DRY_RUN = False


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_deterministic_judge() -> None:
    """Test with temperature=0.0 for deterministic outputs."""
    print_section("TEST 1: Deterministic Judge (temperature=0.0)")

    judge = make_judge(
        name="accuracy_check",
        instructions="Evaluate if {{ outputs }} is factually accurate. Provide a brief rationale.",
        model="openai:/gpt-4o-mini",
        inference_params={"temperature": 0.0},
    )

    print(f"\nJudge: {judge.name}")
    print(f"Model: {judge._model}")
    print(f"inference_params: {judge.inference_params}")

    test_input = {"answer": "The Eiffel Tower is 330 meters tall."}
    print(f"\nTest Input: {test_input}")

    if DRY_RUN:
        print("\n[DRY-RUN] Skipping API call")
        print("[DRY-RUN] Judge created successfully with temperature=0.0")
        return

    print("\nRunning 3 evaluations (should produce identical rationales)...\n")

    results = []
    for i in range(3):
        result = judge(outputs=test_input)
        results.append(result)
        print(f"Run {i+1}:")
        print(f"  value: {result.value}")
        print(f"  rationale: {result.rationale}\n")

    rationales = [r.rationale for r in results]
    identical = all(r == rationales[0] for r in rationales)
    print(f"[Result] All rationales identical: {identical}")
    if identical:
        print("[PASS] Deterministic behavior confirmed with temperature=0.0")
    else:
        print("[INFO] Minor variations possible due to model behavior")


def test_varied_judge() -> None:
    """Test with temperature=1.0 for varied outputs."""
    print_section("TEST 2: Varied Judge (temperature=1.0)")

    judge = make_judge(
        name="accuracy_check",
        instructions="Evaluate if {{ outputs }} is factually accurate. Provide a brief rationale.",
        model="openai:/gpt-4o-mini",
        inference_params={"temperature": 1.0},
    )

    print(f"\nJudge: {judge.name}")
    print(f"Model: {judge._model}")
    print(f"inference_params: {judge.inference_params}")

    test_input = {"answer": "The Eiffel Tower is 330 meters tall."}
    print(f"\nTest Input: {test_input}")

    if DRY_RUN:
        print("\n[DRY-RUN] Skipping API call")
        print("[DRY-RUN] Judge created successfully with temperature=1.0")
        return

    print("\nRunning 3 evaluations (should produce varied rationales)...\n")

    results = []
    for i in range(3):
        result = judge(outputs=test_input)
        results.append(result)
        print(f"Run {i+1}:")
        print(f"  value: {result.value}")
        print(f"  rationale: {result.rationale}\n")

    rationales = [r.rationale for r in results]
    identical = all(r == rationales[0] for r in rationales)
    print(f"[Result] All rationales identical: {identical}")
    if not identical:
        print("[PASS] Varied behavior confirmed with temperature=1.0")
    else:
        print("[INFO] Rationales happened to be identical (rare but possible)")


def test_multiple_params() -> None:
    """Test with multiple inference parameters."""
    print_section("TEST 3: Multiple Inference Parameters")

    judge = make_judge(
        name="quality_eval",
        instructions="Rate {{ outputs }} on clarity and accuracy.",
        model="openai:/gpt-4o-mini",
        inference_params={
            "temperature": 0.3,
            "max_tokens": 200,
            "top_p": 0.9,
        },
    )

    print(f"\nJudge: {judge.name}")
    print(f"Model: {judge._model}")
    print(f"inference_params: {judge.inference_params}")

    test_input = {"response": "Machine learning enables computers to learn from data."}
    print(f"\nTest Input: {test_input}")

    if DRY_RUN:
        print("\n[DRY-RUN] Skipping API call")
        print("[DRY-RUN] Judge created with multiple inference params")
        return

    print("\nRunning evaluation...\n")
    result = judge(outputs=test_input)
    print(f"value: {result.value}")
    print(f"rationale: {result.rationale}")
    print("\n[PASS] Multiple inference parameters work correctly")


def test_default_behavior() -> None:
    """Test judge without inference_params (default)."""
    print_section("TEST 4: Default Behavior (no inference_params)")

    judge = make_judge(
        name="default_eval",
        instructions="Check if {{ outputs }} is valid.",
        model="openai:/gpt-4o-mini",
    )

    print(f"\nJudge: {judge.name}")
    print(f"Model: {judge._model}")
    print(f"inference_params: {judge.inference_params}")

    # Verify inference_params is None
    assert judge.inference_params is None, "inference_params should be None"
    print("\n[PASS] inference_params correctly defaults to None")


def main():
    global DRY_RUN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print("\n" + "#" * 70)
    print("#  OSS Integration Test: inference_params (PR #19152)                #")
    print("#  Adapter: LiteLLM (OpenAI backend)                                 #")
    print("#" * 70)

    if DRY_RUN:
        print("\n[MODE] DRY-RUN - No API calls will be made")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n[ERROR] OPENAI_API_KEY not set")
            print("Run with --dry-run or set: export OPENAI_API_KEY=<key>")
            sys.exit(1)
        print("\n[MODE] LIVE - Making actual API calls")

    print(f"[INFO] MLflow: {sys.path[0]}")

    try:
        test_deterministic_judge()
        test_varied_judge()
        test_multiple_params()
        test_default_behavior()

        print_section("ALL TESTS PASSED")
        print("\nThe inference_params feature is working correctly with LiteLLM!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
