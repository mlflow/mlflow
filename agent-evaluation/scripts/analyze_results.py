"""
Analyze MLflow evaluation results and generate actionable insights.

This script parses the JSON output from `mlflow traces evaluate` and generates:
- Pass rate analysis per scorer
- Failure pattern detection (multi-failure queries)
- Actionable recommendations
- Markdown evaluation report (NOT HTML)

Usage:
    python scripts/analyze_results.py evaluation_results.json

    # Or with custom output file
    python scripts/analyze_results.py evaluation_results.json --output report.md
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text.

    This handles color codes, cursor movement, and other terminal control sequences
    that may appear in mlflow traces evaluate output.

    Args:
        text: Text that may contain ANSI escape sequences

    Returns:
        Text with all ANSI escape sequences removed
    """
    # Standard ANSI escape sequence pattern
    # Matches: ESC [ <parameters> <command>
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def load_evaluation_results(json_file: str) -> list[dict[str, Any]]:
    """Load evaluation results from JSON file, skipping console output.

    Handles mlflow traces evaluate output which contains:
    - Lines 1-N: Console output (progress bars, warnings, logging)
    - Line N+1: Start of JSON array '['
    """
    try:
        with open(json_file) as f:
            content = f.read()

        # Strip ANSI codes before processing
        content = strip_ansi_codes(content)

        # Find the start of JSON array (skip console output)
        json_start = content.find("[")
        if json_start == -1:
            print("✗ No JSON array found in file")
            sys.exit(1)

        json_content = content[json_start:]
        data = json.loads(json_content)

        if not isinstance(data, list):
            print(f"✗ Expected JSON array, got {type(data).__name__}")
            sys.exit(1)

        return data

    except FileNotFoundError:
        print(f"✗ File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON starting at position {json_start}: {e}")
        print(f"  First 100 chars: {json_content[:100]}")
        sys.exit(1)


def extract_scorer_results(data: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """Extract scorer results from assessments array structure.

    Parses the actual mlflow traces evaluate structure:
    [{
        "trace_id": "tr-...",
        "assessments": [
            {"name": "scorer", "result": "yes/no/pass/fail", "rationale": "...", "error": null}
        ]
    }]

    Returns:
        Dictionary mapping scorer names to list of result dictionaries.
        Each result dict contains: {query, trace_id, passed, rationale}
    """
    scorer_results = defaultdict(list)

    for trace_result in data:
        trace_id = trace_result.get("trace_id", "unknown")

        # Extract query from inputs if available
        inputs = trace_result.get("inputs", {})
        query = inputs.get("query", inputs.get("question", "unknown"))

        # Parse assessments array
        assessments = trace_result.get("assessments", [])

        for assessment in assessments:
            scorer_name = assessment.get("name", "unknown")
            result = assessment.get("result", "fail")
            result_str = result.lower() if result else "fail"
            rationale = assessment.get("rationale", "")
            error = assessment.get("error")

            # Map string results to boolean
            # "yes" / "pass" → True
            # "no" / "fail" → False
            passed = result_str in ["yes", "pass"]

            # Skip if there was an error
            if error:
                print(f"  ⚠ Warning: Scorer {scorer_name} had error for trace {trace_id}: {error}")
                continue

            scorer_results[scorer_name].append(
                {"query": query, "trace_id": trace_id, "passed": passed, "rationale": rationale}
            )

    return scorer_results


def calculate_pass_rates(scorer_results: dict[str, list[dict]]) -> dict[str, dict]:
    """Calculate pass rates for each scorer.

    Returns:
        Dictionary mapping scorer names to {pass_rate, passed, total, grade}
    """
    pass_rates = {}

    for scorer_name, results in scorer_results.items():
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Assign grade
        if pass_rate >= 90:
            grade = "A"
            emoji = "✓✓"
        elif pass_rate >= 80:
            grade = "B"
            emoji = "✓"
        elif pass_rate >= 70:
            grade = "C"
            emoji = "⚠"
        elif pass_rate >= 60:
            grade = "D"
            emoji = "⚠⚠"
        else:
            grade = "F"
            emoji = "✗"

        pass_rates[scorer_name] = {
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
            "grade": grade,
            "emoji": emoji,
        }

    return pass_rates


def detect_failure_patterns(scorer_results: dict[str, list[dict]]) -> list[dict]:
    """Detect patterns in failed queries.

    Returns:
        List of pattern dictionaries with {name, queries, scorers, description}
    """
    patterns = []

    # Collect all failures
    failures_by_query = defaultdict(list)

    for scorer_name, results in scorer_results.items():
        for result in results:
            if not result["passed"]:
                failures_by_query[result["query"]].append(
                    {
                        "scorer": scorer_name,
                        "rationale": result["rationale"],
                        "trace_id": result["trace_id"],
                    }
                )

    # Pattern: Multi-failure queries (queries failing 3+ scorers)
    multi_failures = []
    for query, failures in failures_by_query.items():
        if len(failures) >= 3:
            multi_failures.append(
                {"query": query, "scorers": [f["scorer"] for f in failures], "count": len(failures)}
            )

    if multi_failures:
        patterns.append(
            {
                "name": "Multi-Failure Queries",
                "description": "Queries failing 3 or more scorers - need comprehensive fixes",
                "queries": multi_failures,
                "priority": "CRITICAL",
            }
        )

    return patterns


def generate_recommendations(pass_rates: dict[str, dict], patterns: list[dict]) -> list[dict]:
    """Generate actionable recommendations based on analysis.

    Returns:
        List of recommendation dictionaries with {title, issue, impact, effort, priority}
    """
    recommendations = []

    # Recommendations from low-performing scorers
    for scorer_name, metrics in pass_rates.items():
        if metrics["pass_rate"] < 80:
            recommendations.append(
                {
                    "title": f"Improve {scorer_name} performance",
                    "issue": f"Only {metrics['pass_rate']:.1f}% pass rate ({metrics['passed']}/{metrics['total']})",
                    "impact": "Will improve overall evaluation quality",
                    "effort": "Medium",
                    "priority": "HIGH" if metrics["pass_rate"] < 70 else "MEDIUM",
                }
            )

    # Recommendations from patterns
    for pattern in patterns:
        if pattern["priority"] == "CRITICAL":
            recommendations.append(
                {
                    "title": f"Fix {pattern['name'].lower()}",
                    "issue": f"{len(pattern['queries'])} queries failing multiple scorers",
                    "impact": "Critical for baseline quality",
                    "effort": "High",
                    "priority": "CRITICAL",
                }
            )
        elif len(pattern["queries"]) >= 3:
            recommendations.append(
                {
                    "title": f"Address {pattern['name'].lower()}",
                    "issue": pattern["description"],
                    "impact": f"Affects {len(pattern['queries'])} queries",
                    "effort": "Medium",
                    "priority": "HIGH",
                }
            )

    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return recommendations


def generate_report(
    scorer_results: dict[str, list[dict]],
    pass_rates: dict[str, dict],
    patterns: list[dict],
    recommendations: list[dict],
    output_file: str,
) -> None:
    """Generate markdown evaluation report."""

    total_queries = len(next(iter(scorer_results.values()))) if scorer_results else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "# Agent Evaluation Results Analysis",
        "",
        f"**Generated**: {timestamp}",
        f"**Dataset**: {total_queries} queries evaluated",
        f"**Scorers**: {len(scorer_results)} ({', '.join(scorer_results.keys())})",
        "",
        "## Overall Pass Rates",
        "",
    ]

    # Pass rates table
    for scorer_name, metrics in pass_rates.items():
        emoji = metrics["emoji"]
        report_lines.append(
            f"  {scorer_name:30} {metrics['pass_rate']:5.1f}% ({metrics['passed']}/{metrics['total']}) {emoji}"
        )

    report_lines.extend(["", ""])

    # Average pass rate
    avg_pass_rate = (
        sum(m["pass_rate"] for m in pass_rates.values()) / len(pass_rates) if pass_rates else 0
    )
    report_lines.append(f"**Average Pass Rate**: {avg_pass_rate:.1f}%")
    report_lines.extend(["", ""])

    # Failure patterns
    if patterns:
        report_lines.extend(["## Failure Patterns Detected", ""])

        for i, pattern in enumerate(patterns, 1):
            report_lines.extend(
                [
                    f"### {i}. {pattern['name']} [{pattern['priority']}]",
                    "",
                    f"**Description**: {pattern['description']}",
                    "",
                    f"**Affected Queries**: {len(pattern['queries'])}",
                    "",
                ]
            )

            for query_info in pattern["queries"][:5]:  # Show first 5
                report_lines.append(
                    f'- **Query**: "{query_info["query"][:100]}{"..." if len(query_info["query"]) > 100 else ""}"'
                )
                report_lines.append(f"  - Failed scorers: {', '.join(query_info['scorers'])}")
                report_lines.append("")

            if len(pattern["queries"]) > 5:
                report_lines.append(f"  _(+{len(pattern['queries']) - 5} more queries)_")
                report_lines.append("")

            report_lines.append("")

    # Recommendations
    if recommendations:
        report_lines.extend(["## Recommendations", ""])

        for i, rec in enumerate(recommendations, 1):
            report_lines.extend(
                [
                    f"### {i}. {rec['title']} [{rec['priority']}]",
                    "",
                    f"- **Issue**: {rec['issue']}",
                    f"- **Expected Impact**: {rec['impact']}",
                    f"- **Effort**: {rec['effort']}",
                    "",
                ]
            )

    # Next steps
    report_lines.extend(
        [
            "## Next Steps",
            "",
            "1. Address CRITICAL and HIGH priority recommendations first",
            "2. Re-run evaluation after implementing fixes",
            "3. Compare results to measure improvement",
            "4. Consider expanding dataset to cover identified gaps",
            "",
            "---",
            "",
            f"**Report Generated**: {timestamp}",
            "**Evaluation Framework**: MLflow Agent Evaluation",
            "",
        ]
    )

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ Report saved to: {output_file}")


def main():
    """Main analysis workflow."""
    print("=" * 60)
    print("MLflow Evaluation Results Analysis")
    print("=" * 60)
    print()

    # Parse arguments
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/analyze_results.py <evaluation_results.json> [--output report.md]"
        )
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = "evaluation_report.md"

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    # Load results
    print(f"Loading evaluation results from: {json_file}")
    data = load_evaluation_results(json_file)
    print("✓ Results loaded")
    print()

    # Extract scorer results
    print("Extracting scorer results...")
    scorer_results = extract_scorer_results(data)

    if not scorer_results:
        print("✗ No scorer results found in JSON")
        print("  Check that the JSON file contains evaluation results")
        sys.exit(1)

    print(f"✓ Found {len(scorer_results)} scorer(s)")
    print()

    # Calculate pass rates
    print("Calculating pass rates...")
    pass_rates = calculate_pass_rates(scorer_results)

    print("\nOverall Pass Rates:")
    for scorer_name, metrics in pass_rates.items():
        emoji = metrics["emoji"]
        print(
            f"  {scorer_name:30} {metrics['pass_rate']:5.1f}% ({metrics['passed']}/{metrics['total']}) {emoji}"
        )
    print()

    # Detect patterns
    print("Detecting failure patterns...")
    patterns = detect_failure_patterns(scorer_results)

    if patterns:
        print(f"✓ Found {len(patterns)} pattern(s)")
        for pattern in patterns:
            print(
                f"  - {pattern['name']}: {len(pattern['queries'])} queries [{pattern['priority']}]"
            )
    else:
        print("  No significant patterns detected")
    print()

    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(pass_rates, patterns)
    print(f"✓ Generated {len(recommendations)} recommendation(s)")
    print()

    # Generate report
    print("Generating markdown report...")
    generate_report(scorer_results, pass_rates, patterns, recommendations, output_file)
    print()

    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print()
    print(f"Review the report at: {output_file}")
    print()


if __name__ == "__main__":
    main()
