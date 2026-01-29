"""
Sample chatbot application to generate traces for MLflow UI testing.
This script creates a simple chatbot and logs multiple traces to an experiment.
"""

import mlflow
from openai import OpenAI
import os
import time

# Set up OpenAI client (you can use a mock or real API key)
# For demo purposes, we'll create traces even without actual API calls

def create_sample_experiment():
    """Create a sample experiment for traces"""
    experiment_name = "Chatbot Traces Demo"
    
    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name)
    print(f"Created/Using experiment: {experiment_name} (ID: {experiment.experiment_id})")
    
    return experiment

def simulate_chatbot_conversation(user_message, response_message, metadata=None):
    """Simulate a chatbot conversation and create a trace"""
    
    # Calculate token counts (more realistic estimates)
    prompt_tokens = max(10, len(user_message.split()) * 1.3)  # Approximate tokenization
    completion_tokens = max(10, len(response_message.split()) * 1.3)
    total_tokens = prompt_tokens + completion_tokens
    
    # Create request/input data
    request_data = {
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 150,
    }
    
    # Start a trace with a new run and set inputs
    with mlflow.start_span(name="chatbot_conversation", span_type="CHAT_MODEL") as span:
        # Set the input/request
        span.set_inputs(request_data)
        
        # Add attributes/metadata
        if metadata:
            span.set_attributes(metadata)
        
        # Add token usage attributes at the root span level
        span.set_attributes({
            "mlflow.traceInputs": str(request_data),
            "mlflow.traceOutputs": "",  # Will be set later
            "mlflow.spanType": "CHAT_MODEL",
        })
        
        # Simulate preprocessing
        with mlflow.start_span(name="preprocess_input", span_type="PARSER") as preprocess_span:
            preprocess_span.set_inputs({"text": user_message})
            time.sleep(0.1)
            processed_input = user_message.strip().lower()
            preprocess_span.set_outputs({"processed_text": processed_input, "length": len(processed_input)})
        
        # Simulate LLM call
        with mlflow.start_span(name="llm_call", span_type="LLM") as llm_span:
            llm_span.set_inputs({
                "prompt": user_message,
                "model": "gpt-4o-mini",
            })
            llm_span.set_attributes({
                "provider": "openai",
                "mlflow.tokensPrompt": int(prompt_tokens),
                "mlflow.tokensCompletion": int(completion_tokens),
                "mlflow.tokensTotal": int(total_tokens),
            })
            time.sleep(0.2)
            llm_span.set_outputs({"content": response_message})
        
        # Simulate postprocessing
        with mlflow.start_span(name="postprocess_output", span_type="PARSER") as postprocess_span:
            postprocess_span.set_inputs({"raw_response": response_message})
            time.sleep(0.05)
            postprocess_span.set_outputs({"formatted_response": response_message, "length": len(response_message)})
        
        # Create response/output data
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_message
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
        
        # Set the output/response and token attributes
        span.set_outputs(response_data)
        span.set_attributes({
            "mlflow.tokensPrompt": int(prompt_tokens),
            "mlflow.tokensCompletion": int(completion_tokens),
            "mlflow.tokensTotal": int(total_tokens),
            "status": "success",
        })
        
        return response_message

def main():
    """Main function to create sample traces"""
    
    # Create experiment
    experiment = create_sample_experiment()
    
    # Sample conversations
    conversations = [
        {
            "user": "What is MLflow?",
            "assistant": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
            "metadata": {"topic": "mlflow_basics", "user_id": "user_001"}
        },
        {
            "user": "How do I log a model in MLflow?",
            "assistant": "You can log a model using mlflow.log_model() or by using autologging with mlflow.<framework>.autolog().",
            "metadata": {"topic": "model_logging", "user_id": "user_002"}
        },
        {
            "user": "What is tracing in MLflow?",
            "assistant": "Tracing in MLflow provides observability for LLM applications, allowing you to track the execution flow of your GenAI apps.",
            "metadata": {"topic": "tracing", "user_id": "user_001"}
        },
        {
            "user": "Can MLflow integrate with LangChain?",
            "assistant": "Yes! MLflow has native integration with LangChain. You can enable it with mlflow.langchain.autolog().",
            "metadata": {"topic": "integrations", "user_id": "user_003"}
        },
        {
            "user": "How do I deploy a model with MLflow?",
            "assistant": "MLflow provides several deployment options including local REST API, Docker, Kubernetes, and cloud platforms like AWS SageMaker.",
            "metadata": {"topic": "deployment", "user_id": "user_002"}
        },
        {
            "user": "What are the main components of MLflow?",
            "assistant": "MLflow has four main components: Tracking, Projects, Models, and Model Registry. Each serves a specific purpose in the ML lifecycle.",
            "metadata": {"topic": "mlflow_basics", "user_id": "user_004"}
        },
        {
            "user": "How do I compare different model runs?",
            "assistant": "You can use the MLflow UI to compare runs side-by-side, viewing metrics, parameters, and artifacts in a unified interface.",
            "metadata": {"topic": "experiment_tracking", "user_id": "user_001"}
        },
        {
            "user": "Does MLflow support Python?",
            "assistant": "Yes, MLflow has comprehensive Python support and is primarily used with Python, though it also supports R, Java, and REST APIs.",
            "metadata": {"topic": "languages", "user_id": "user_005"}
        },
    ]
    
    print(f"\nGenerating {len(conversations)} sample traces...\n")
    
    # Create traces for each conversation
    for idx, conv in enumerate(conversations, 1):
        print(f"[{idx}/{len(conversations)}] Processing: '{conv['user'][:50]}...'")
        
        # Create a trace for this conversation
        simulate_chatbot_conversation(
            user_message=conv['user'],
            response_message=conv['assistant'],
            metadata=conv['metadata']
        )
        
        time.sleep(0.5)  # Small delay between traces
    
    print(f"\n✅ Successfully created {len(conversations)} traces!")
    print(f"\n🌐 View them in the MLflow UI at: http://localhost:3000")
    print(f"   Navigate to the experiment: '{experiment.name}'")
    print(f"   Then click on the 'Traces' tab\n")

if __name__ == "__main__":
    main()
