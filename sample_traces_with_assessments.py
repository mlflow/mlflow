#!/usr/bin/env python3
"""
Script to log sample traces with assessments (from judges) to MLflow.
This demonstrates how traces look with various types of assessments in the UI.
"""

import mlflow
from mlflow.entities import Feedback, Expectation, AssessmentSource, AssessmentSourceType
import time
import random

# Set the tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("Chatbot Traces Demo")

def create_chatbot_trace_with_assessments(question: str, answer: str, context: str, assessments_data: dict):
    """Create a trace with request/response data and log assessments."""
    
    # Calculate token counts
    prompt_tokens = max(10, len(question.split()) * 1.3)
    completion_tokens = max(10, len(answer.split()) * 1.3)
    total_tokens = prompt_tokens + completion_tokens
    
    # Create request data
    request_data = {
        "messages": [
            {"role": "user", "content": question}
        ],
        "context": context,
        "model": "gpt-4",
        "temperature": 0.7,
    }
    
    # Create response data
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens)
        }
    }
    
    # Start a trace with spans
    with mlflow.start_span(name="chatbot_qa", span_type="CHAT_MODEL") as span:
        # Set inputs and outputs
        span.set_inputs(request_data)
        span.set_outputs(response_data)
        
        # Set token attributes
        span.set_attributes({
            "mlflow.tokensPrompt": int(prompt_tokens),
            "mlflow.tokensCompletion": int(completion_tokens),
            "mlflow.tokensTotal": int(total_tokens),
            "context": context,
        })
        
        # Simulate some processing spans
        with mlflow.start_span(name="retrieve_context", span_type="RETRIEVER") as retriever_span:
            retriever_span.set_inputs({"query": question})
            time.sleep(0.05)
            retriever_span.set_outputs({"context": context, "num_docs": 3})
        
        with mlflow.start_span(name="generate_answer", span_type="LLM") as llm_span:
            llm_span.set_inputs({"prompt": question, "context": context})
            time.sleep(0.1)
            llm_span.set_outputs({"answer": answer})
            llm_span.set_attributes({
                "mlflow.tokensPrompt": int(prompt_tokens),
                "mlflow.tokensCompletion": int(completion_tokens),
            })
    
    # Get the trace ID from the current trace
    trace_id = mlflow.get_last_active_trace_id()
    if not trace_id:
        print("   ❌ Failed to get trace")
        return None
    
    print(f"   Trace ID: {trace_id}")
    
    # Get the full trace object to extract the root span ID
    from mlflow.tracing.client import TracingClient
    client = TracingClient()
    trace = client.get_trace(trace_id)
    
    # Get the root span ID - this is critical for assessments to show up
    root_span_id = None
    if trace and trace.data and trace.data.spans:
        # Find the root span (span with no parent)
        for span in trace.data.spans:
            if not span.parent_id:
                root_span_id = span.span_id
                break
    
    if not root_span_id:
        print("   ⚠️  Warning: Could not find root span ID")
    else:
        print(f"   Root Span ID: {root_span_id}")
    
    # Log assessments from different judges
    
    # 1. Correctness - LLM Judge (GPT-4)
    correctness_source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id="gpt-4",
    )
    correctness_feedback = Feedback(
        name="correctness",
        value=float(assessments_data['correctness']),  # Explicitly convert to float
        rationale=f"The answer is {'correct and accurate' if assessments_data['correctness'] > 0.7 else 'incorrect or inaccurate'}.",
        source=correctness_source,
        metadata={"judge_model": "gpt-4", "temperature": 0.0},
        span_id=root_span_id,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=correctness_feedback)
    print(f"   ✓ Logged correctness: {assessments_data['correctness']} (type: {type(correctness_feedback.value).__name__})")
    
    # 2. Relevance - LLM Judge (Claude)
    relevance_source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id="claude-3-opus",
    )
    relevance_feedback = Feedback(
        name="relevance",
        value=float(assessments_data['relevance']),  # Explicitly convert to float
        rationale=f"The answer is {'highly relevant' if assessments_data['relevance'] > 0.7 else 'not very relevant'} to the question.",
        source=relevance_source,
        metadata={"judge_model": "claude-3-opus", "temperature": 0.0},
        span_id=root_span_id,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=relevance_feedback)
    print(f"   ✓ Logged relevance: {assessments_data['relevance']} (type: {type(relevance_feedback.value).__name__})")
    
    # 3. Faithfulness - LLM Judge (GPT-4)
    faithfulness_source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id="gpt-4-turbo",
    )
    faithfulness_feedback = Feedback(
        name="faithfulness",
        value=float(assessments_data['faithfulness']),  # Explicitly convert to float
        rationale=f"The answer {'faithfully represents' if assessments_data['faithfulness'] > 0.7 else 'does not faithfully represent'} the provided context.",
        source=faithfulness_source,
        metadata={"judge_model": "gpt-4-turbo", "temperature": 0.0},
        span_id=root_span_id,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=faithfulness_feedback)
    print(f"   ✓ Logged faithfulness: {assessments_data['faithfulness']} (type: {type(faithfulness_feedback.value).__name__})")
    
    # 4. Coherence - Human Judge
    coherence_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="expert_reviewer@example.com",
    )
    coherence_feedback = Feedback(
        name="coherence",
        value=float(assessments_data['coherence']),  # Explicitly convert to float
        rationale=f"The answer is {'well-structured and coherent' if assessments_data['coherence'] > 0.7 else 'poorly structured or incoherent'}.",
        source=coherence_source,
        metadata={"reviewer_role": "subject_matter_expert"},
        span_id=root_span_id,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=coherence_feedback)
    print(f"   ✓ Logged coherence: {assessments_data['coherence']} (type: {type(coherence_feedback.value).__name__})")
    
    # 5. Add an expectation (ground truth)
    expectation_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="data_annotator@example.com",
    )
    
    # Create a simple ground truth based on whether the answer seems correct
    if assessments_data['correctness'] > 0.8:
        expected_value = "correct"
    elif assessments_data['correctness'] > 0.5:
        expected_value = "partially_correct"
    else:
        expected_value = "incorrect"
    
    expectation = Expectation(
        name="ground_truth",
        value=expected_value,
        source=expectation_source,
        span_id=root_span_id,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=expectation)
    print(f"   ✓ Logged expectation: {expected_value}")
    
    return trace_id

# Sample questions and answers
qa_pairs = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "context": "Geography facts about European countries",
        "assessments": {
            "correctness": 1.0,
            "relevance": 0.95,
            "faithfulness": 0.98,
            "coherence": 0.92,
        }
    },
    {
        "question": "Explain quantum computing in simple terms",
        "answer": "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1. This allows quantum computers to process many possibilities at once.",
        "context": "Technology and computer science explanations",
        "assessments": {
            "correctness": 0.88,
            "relevance": 0.92,
            "faithfulness": 0.85,
            "coherence": 0.90,
        }
    },
    {
        "question": "What are the health benefits of exercise?",
        "answer": "Exercise improves cardiovascular health, strengthens muscles, helps maintain healthy weight, boosts mood and energy levels, and reduces risk of chronic diseases.",
        "context": "Health and wellness information",
        "assessments": {
            "correctness": 0.95,
            "relevance": 0.97,
            "faithfulness": 0.93,
            "coherence": 0.94,
        }
    },
    {
        "question": "How do I make a cake?",
        "answer": "The weather today is sunny with a high of 75 degrees.",
        "context": "Cooking and recipe instructions",
        "assessments": {
            "correctness": 0.1,
            "relevance": 0.05,
            "faithfulness": 0.2,
            "coherence": 0.8,
        }
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions.",
        "context": "AI and machine learning concepts",
        "assessments": {
            "correctness": 0.92,
            "relevance": 0.96,
            "faithfulness": 0.94,
            "coherence": 0.91,
        }
    },
]

print("Logging traces with assessments from judges...")
print(f"Experiment: {experiment.name} (ID: {experiment.experiment_id})")

for idx, qa in enumerate(qa_pairs):
    print(f"\n{idx + 1}. Processing: {qa['question'][:50]}...")
    
    # Create the trace with assessments
    trace_id = create_chatbot_trace_with_assessments(
        question=qa['question'],
        answer=qa['answer'],
        context=qa['context'],
        assessments_data=qa['assessments']
    )
    
    # Small delay to make trace IDs more distinct
    time.sleep(0.5)

print("\n" + "="*60)
print("✅ Successfully logged all traces with assessments!")
print("="*60)
print("\nYou can now view these traces in the MLflow UI at:")
print("http://localhost:3000/#/experiments/[experiment-id]/traces")
print("\nThe traces include assessments from:")
print("  • LLM Judges: GPT-4, Claude-3-Opus, GPT-4-Turbo")
print("  • Human reviewers: Expert reviewers and data annotators")
print("  • Metrics: Correctness, Relevance, Faithfulness, Coherence")
print("  • Ground truth expectations for each trace")


