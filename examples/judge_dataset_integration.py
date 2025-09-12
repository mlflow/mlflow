#!/usr/bin/env python
"""
Example: Using Custom Judges with Evaluation Datasets

Demonstrates the complete workflow of using custom judges with evaluation datasets
for systematic AI application evaluation.
"""

import mlflow
import pandas as pd
from mlflow.genai.judges import make_judge
from mlflow.entities import AssessmentSource, AssessmentSourceType


def create_sample_dataset():
    """Create a sample evaluation dataset with customer support scenarios."""
    test_cases = [
        {
            "inputs": {
                "ticket_id": "T001",
                "issue": "Cannot log into my account - getting error 403",
                "severity": "high",
                "customer_tier": "premium",
                "previous_contacts": 2
            },
            "outputs": {
                "response": (
                    "I sincerely apologize for the login issues you're experiencing. "
                    "Error 403 typically indicates an authentication problem. Let me help you resolve this immediately:\n\n"
                    "1. First, please clear your browser cache and cookies\n"
                    "2. Try resetting your password using the 'Forgot Password' link\n"
                    "3. Ensure you're using the correct email associated with your premium account\n\n"
                    "If these steps don't work, I'm escalating this to our technical team right now. "
                    "As a premium customer, you'll receive priority support within the next hour. "
                    "Your ticket reference is T001-PRIORITY."
                ),
                "resolved": False,
                "escalated": True
            },
            "expectations": {
                "quality_rating": "excellent",
                "criteria_met": {
                    "acknowledged_issue": True,
                    "provided_solution": True,
                    "appropriate_tone": True,
                    "followed_sla": True
                }
            }
        },
        {
            "inputs": {
                "ticket_id": "T002",
                "issue": "How do I change my billing address?",
                "severity": "low",
                "customer_tier": "standard",
                "previous_contacts": 0
            },
            "outputs": {
                "response": (
                    "To change your billing address, please follow these steps:\n"
                    "1. Log into your account\n"
                    "2. Go to Settings > Billing Information\n"
                    "3. Click 'Edit' next to your current address\n"
                    "4. Enter your new address and click 'Save'\n\n"
                    "The change will take effect immediately for future invoices."
                ),
                "resolved": True,
                "escalated": False
            },
            "expectations": {
                "quality_rating": "good",
                "criteria_met": {
                    "acknowledged_issue": True,
                    "provided_solution": True,
                    "appropriate_tone": True,
                    "followed_sla": True
                }
            }
        },
        {
            "inputs": {
                "ticket_id": "T003",
                "issue": "Your product is terrible and I want a refund NOW!!!",
                "severity": "high",
                "customer_tier": "free",
                "previous_contacts": 5
            },
            "outputs": {
                "response": "Have you tried turning it off and on again?",
                "resolved": False,
                "escalated": False
            },
            "expectations": {
                "quality_rating": "poor",
                "criteria_met": {
                    "acknowledged_issue": False,
                    "provided_solution": False,
                    "appropriate_tone": False,
                    "followed_sla": False
                }
            }
        }
    ]
    
    return pd.DataFrame(test_cases)


def create_judges():
    """Create custom judges for evaluation."""
    quality_judge = make_judge(
        name="support_quality",
        instructions=(
            "Evaluate the customer support response in {{ outputs }} "
            "for the ticket in {{ inputs }}.\n\n"
            "Consider the following criteria:\n"
            "1. **Issue Acknowledgment**: Does the response acknowledge the customer's problem?\n"
            "2. **Solution Quality**: Is a clear, actionable solution provided?\n"
            "3. **Tone and Empathy**: Is the tone appropriate for the situation and customer tier?\n"
            "4. **SLA Compliance**: Does it meet service level expectations for {{ inputs.customer_tier }} customers?\n\n"
            "For reference, if expectations are provided, the expected quality is {{ expectations.quality_rating }}.\n\n"
            "Rate the response as one of: 'excellent', 'good', 'fair', or 'poor'\n"
            "Provide detailed reasoning for your rating."
        ),
        model="anthropic:/claude-3-opus-20240229"
    )
    
    sla_judge = make_judge(
        name="sla_compliance",
        instructions=(
            "Check if the response in {{ outputs }} meets SLA requirements "
            "for a {{ inputs.customer_tier }} tier customer with {{ inputs.severity }} severity issue.\n\n"
            "SLA Requirements by Tier:\n"
            "- **Enterprise**: Immediate response, dedicated specialist, proactive solutions\n"
            "- **Premium**: Priority handling, escalation available, response within 1 hour\n"
            "- **Standard**: Clear solution, 24-hour response time\n"
            "- **Free**: Basic guidance, best-effort support\n\n"
            "Return 'compliant' or 'non_compliant' with specific explanation."
        ),
        model="anthropic:/claude-3-opus-20240229"
    )
    
    return quality_judge, sla_judge


def evaluate_and_analyze(df, judges):
    """Run evaluation and analyze results against expectations."""
    results = mlflow.genai.evaluate(
        data=df,
        scorers=judges,
    )
    
    results_df = results.tables["eval_results_table"]
    
    # Compare with expectations
    accuracy_data = []
    for idx in range(len(df)):
        expected = df.iloc[idx]["expectations"]["quality_rating"]
        actual = results_df.iloc[idx].get("support_quality/v1/value", "N/A")
        accuracy_data.append({
            "ticket_id": df.iloc[idx]["inputs"]["ticket_id"],
            "expected": expected,
            "actual": actual,
            "match": expected == actual
        })
    
    accuracy = sum(d["match"] for d in accuracy_data) / len(accuracy_data) * 100
    
    return results, accuracy_data, accuracy


def main():
    """Main execution function."""
    # Create dataset
    dataset_df = create_sample_dataset()
    
    # Create judges
    quality_judge, sla_judge = create_judges()
    
    # Run evaluation
    results, accuracy_data, accuracy_pct = evaluate_and_analyze(
        dataset_df, 
        [quality_judge, sla_judge]
    )
    
    # Return results for documentation
    return {
        "dataset_size": len(dataset_df),
        "judges_used": 2,
        "accuracy": accuracy_pct,
        "metrics": results.metrics,
        "accuracy_details": accuracy_data
    }


if __name__ == "__main__":
    result = main()
    print(f"Evaluation complete: {result['accuracy']:.1f}% accuracy")
    for detail in result['accuracy_details']:
        print(f"  {detail['ticket_id']}: Expected={detail['expected']}, Got={detail['actual']}, Match={detail['match']}")