"""
Example of using Amazon Bedrock models (including Amazon Nova) for LLM evaluation with MLflow.

This example demonstrates how to use Amazon Nova models via Bedrock for evaluating
question-answering systems. Amazon Nova provides cost-effective and high-performance
foundation models for various evaluation tasks.

Prerequisites:
1. AWS credentials configured (via AWS CLI or environment variables)
2. Access to Amazon Bedrock service
3. Nova models enabled in your AWS region
4. Required Python packages: mlflow, pandas, boto3

Usage:
    python evaluate_with_bedrock_judge.py
"""

import pandas as pd
import mlflow
from mlflow.metrics.genai import EvaluationExample, answer_similarity


def create_evaluation_examples():
    """
    Create evaluation examples for few-shot learning.
    These examples help the model understand the scoring criteria.
    """
    examples = [
        EvaluationExample(
            input="What is MLflow?",
            output="MLflow is an open-source platform for managing machine learning workflows.",
            score=4,
            justification="The answer correctly identifies MLflow's purpose but could provide more details about its specific components like tracking, projects, and models.",
            grading_context={
                "ground_truth": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for experiment tracking, model packaging, versioning, and deployment."
            }
        ),
        EvaluationExample(
            input="Explain neural networks",
            output="Neural networks are computing systems inspired by biological brains.",
            score=3,
            justification="Basic definition is correct but lacks details about layers, activation functions, or training processes.",
            grading_context={
                "ground_truth": "Neural networks are computational models composed of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions, commonly trained using backpropagation."
            }
        )
    ]
    return examples


def setup_nova_metrics():
    """
    Set up evaluation metrics using different Amazon Nova models.
    
    Returns:
        dict: Dictionary containing configured metrics for different Nova variants
    """
    examples = create_evaluation_examples()
    
    metrics = {
        # Amazon Nova Lite - Cost-effective for most evaluation tasks
        "nova_lite": answer_similarity(
            model="bedrock/amazon.nova-lite-v1",
            parameters={
                "temperature": 0.0,      # Deterministic scoring
                "maxTokens": 512,        # Reasonable response length
                "topP": 0.9             # Balanced creativity vs consistency
            },
            examples=examples
        ),
        
        # Amazon Nova Pro - For complex evaluations requiring higher reasoning
        "nova_pro": answer_similarity(
            model="bedrock/amazon.nova-pro-v1",
            parameters={
                "temperature": 0.1,      # Slight variation for nuanced scoring
                "maxTokens": 1024,       # More detailed justifications
                "topP": 0.95            # Broader token consideration
            },
            examples=examples
        )
    }
    
    return metrics


def create_evaluation_data():
    """
    Create sample evaluation dataset for question-answering system.
    """
    eval_df = pd.DataFrame({
        "inputs": [
            "What is MLflow?",
            "What is Apache Spark?",
            "How does gradient descent work?",
            "What is Python used for?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for experiment tracking, model packaging, versioning, and deployment.",
            "Apache Spark is an open-source distributed computing system for big data processing. It provides APIs in Java, Scala, Python and R, and supports SQL, streaming, machine learning, and graph processing.",
            "Gradient descent is an optimization algorithm used to minimize functions by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. It's commonly used in machine learning to update model parameters.",
            "Python is a high-level programming language used for web development, data analysis, artificial intelligence, scientific computing, and automation. It's known for its simplicity and extensive library ecosystem.",
        ],
    })
    return eval_df


def main():
    """
    Main function to demonstrate Amazon Nova evaluation with MLflow.
    """
    print("🚀 Setting up Amazon Nova evaluation metrics...")
    
    # Get configured metrics
    metrics = setup_nova_metrics()
    
    # Create evaluation data
    eval_df = create_evaluation_data()
    
    print("✅ Amazon Nova metrics configured successfully!")
    print("📊 Available metrics:")
    for metric_name, metric in metrics.items():
        print(f"   - {metric_name}: {metric.name}")
    
    print("\n🔧 To run evaluation with these metrics:")
    print("""
    with mlflow.start_run():
        results = mlflow.evaluate(
            model_uri=your_model_uri,
            data=eval_df,
            targets="ground_truth", 
            model_type="question-answering",
            extra_metrics=list(metrics.values())
        )
    """)
    
    print("\n📝 Note: Ensure AWS credentials are configured and Bedrock access is enabled.")
    print("   AWS CLI: aws configure")
    print("   Environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    print("   Region: Set desired AWS region with Nova model access")


if __name__ == "__main__":
    main()
