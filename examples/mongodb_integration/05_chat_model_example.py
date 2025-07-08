#!/usr/bin/env python
"""
Genesis-Flow ChatModel Example with MongoDB Integration

This example demonstrates how to use Genesis-Flow's ChatModel functionality
with MongoDB storage. It shows:

1. Creating custom ChatModel implementations
2. Logging and registering ChatModels with MongoDB
3. Tool calling capabilities
4. Streaming responses
5. Model evaluation and comparison
6. Production deployment patterns

Note: ChatModel is deprecated in MLflow 3.0+ in favor of ResponsesAgent,
but is still supported in Genesis-Flow for compatibility.
"""

import json
import time
import uuid
from typing import Generator
from abc import abstractmethod

import mlflow
import mlflow.pyfunc
from mlflow.types.llm import (
    ChatMessage,
    ChatParams,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    TokenUsageStats,
    ToolDefinition,
    FunctionToolDefinition,
    ToolParamsSchema,
    ParamProperty,
    ToolCall,
    FunctionToolCallArguments
)


class SimpleChatModel(mlflow.pyfunc.ChatModel):
    """
    A simple ChatModel that simulates conversational AI responses.
    This demonstrates the basic ChatModel interface.
    """
    
    def predict(self, context, messages: list[ChatMessage], params: ChatParams) -> ChatCompletionResponse:
        """Generate a non-streaming chat response."""
        # Get the latest user message
        user_message = messages[-1].content if messages else "Hello!"
        
        # Create a simple response based on the user input
        if "weather" in user_message.lower():
            response_text = f"The weather is sunny today! You asked: {user_message}"
        elif "time" in user_message.lower():
            response_text = f"The current time is {time.strftime('%Y-%m-%d %H:%M:%S')}. You asked: {user_message}"
        else:
            response_text = f"I understand you said: {user_message}. How can I help you further?"
        
        # Create response message
        response_message = ChatMessage(
            role="assistant", 
            content=response_text
        )
        
        # Create chat choice
        choice = ChatChoice(
            index=0,
            message=response_message,
            finish_reason="stop"
        )
        
        # Create usage statistics
        usage = TokenUsageStats(
            prompt_tokens=len(user_message.split()) if user_message else 0,
            completion_tokens=len(response_text.split()),
            total_tokens=len(user_message.split()) + len(response_text.split()) if user_message else len(response_text.split())
        )
        
        # Return complete response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model="simple-chat-model",
            choices=[choice],
            usage=usage
        )
    
    def predict_stream(self, context, messages: list[ChatMessage], params: ChatParams) -> Generator[ChatCompletionChunk, None, None]:
        """Generate a streaming chat response."""
        user_message = messages[-1].content if messages else "Hello!"
        
        # Create response text
        if "weather" in user_message.lower():
            response_text = f"The weather is sunny today! You asked: {user_message}"
        elif "time" in user_message.lower():
            response_text = f"The current time is {time.strftime('%Y-%m-%d %H:%M:%S')}. You asked: {user_message}"
        else:
            response_text = f"I understand you said: {user_message}. How can I help you further?"
        
        # Stream the response word by word
        words = response_text.split()
        
        for i, word in enumerate(words):
            # Create delta with current word
            delta = ChatChoiceDelta(
                role="assistant" if i == 0 else None,
                content=word + " " if i < len(words) - 1 else word
            )
            
            # Create chunk choice
            choice = ChatChunkChoice(
                index=0,
                delta=delta,
                finish_reason="stop" if i == len(words) - 1 else None
            )
            
            # Create and yield chunk
            chunk = ChatCompletionChunk(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="simple-chat-model",
                choices=[choice]
            )
            
            yield chunk
            time.sleep(0.1)  # Simulate streaming delay


class AdvancedChatModel(mlflow.pyfunc.ChatModel):
    """
    An advanced ChatModel with tool calling capabilities.
    This demonstrates more complex ChatModel features.
    """
    
    def predict(self, context, messages: list[ChatMessage], params: ChatParams) -> ChatCompletionResponse:
        """Generate response with potential tool calling."""
        user_message = messages[-1].content if messages else "Hello!"
        
        # Check if tools are available and if we should use them
        if params.tools and "calculate" in user_message.lower():
            # Use tool calling for calculation
            return self._handle_tool_call(user_message, params)
        else:
            # Regular response
            return self._generate_regular_response(user_message)
    
    def _handle_tool_call(self, user_message: str, params: ChatParams) -> ChatCompletionResponse:
        """Handle tool calling for calculations."""
        # Create a tool call for calculation
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionToolCallArguments(
                name="calculate",
                arguments=json.dumps({"expression": "2 + 2", "operation": "addition"})
            )
        )
        
        # Create response message with tool call
        response_message = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
        
        # Create chat choice
        choice = ChatChoice(
            index=0,
            message=response_message,
            finish_reason="tool_calls"
        )
        
        # Create usage statistics
        usage = TokenUsageStats(
            prompt_tokens=len(user_message.split()),
            completion_tokens=10,  # Estimated for tool call
            total_tokens=len(user_message.split()) + 10
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model="advanced-chat-model",
            choices=[choice],
            usage=usage
        )
    
    def _generate_regular_response(self, user_message: str) -> ChatCompletionResponse:
        """Generate regular chat response."""
        # Advanced response logic
        if "explain" in user_message.lower():
            response_text = f"Let me explain that topic in detail. You asked about: {user_message}"
        elif "summarize" in user_message.lower():
            response_text = f"Here's a summary of what you mentioned: {user_message}"
        else:
            response_text = f"I'm an advanced assistant. You said: {user_message}. I can help with explanations, summaries, and calculations."
        
        # Create response message
        response_message = ChatMessage(
            role="assistant",
            content=response_text
        )
        
        # Create chat choice
        choice = ChatChoice(
            index=0,
            message=response_message,
            finish_reason="stop"
        )
        
        # Create usage statistics
        usage = TokenUsageStats(
            prompt_tokens=len(user_message.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(user_message.split()) + len(response_text.split())
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model="advanced-chat-model",
            choices=[choice],
            usage=usage
        )


def create_sample_tools():
    """Create sample tools for tool calling demonstration."""
    # Define calculation tool
    calculate_tool = ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters=ToolParamsSchema(
                type="object",
                properties={
                    "expression": ParamProperty(
                        type="string",
                        description="The mathematical expression to evaluate"
                    ),
                    "operation": ParamProperty(
                        type="string",
                        description="The type of operation (addition, subtraction, etc.)",
                        enum=["addition", "subtraction", "multiplication", "division"]
                    )
                },
                required=["expression"]
            )
        )
    )
    
    return [calculate_tool]


def test_chat_model_functionality():
    """Test ChatModel functionality before logging."""
    print("🧪 Testing ChatModel Functionality")
    print("=" * 40)
    
    # Test simple model
    print("\n1. Testing Simple ChatModel:")
    simple_model = SimpleChatModel()
    
    # Test regular prediction
    messages = [ChatMessage(role="user", content="What's the weather like?")]
    params = ChatParams(temperature=0.7, max_tokens=100)
    
    response = simple_model.predict(context=None, messages=messages, params=params)
    print(f"Response: {response.choices[0].message.content}")
    
    # Test streaming
    print("\n2. Testing Streaming Response:")
    stream_messages = [ChatMessage(role="user", content="Tell me about the time")]
    stream_chunks = list(simple_model.predict_stream(context=None, messages=stream_messages, params=params))
    
    streamed_content = " ".join([
        chunk.choices[0].delta.content or ""
        for chunk in stream_chunks
        if chunk.choices[0].delta.content
    ])
    print(f"Streamed content: {streamed_content}")
    
    # Test advanced model
    print("\n3. Testing Advanced ChatModel:")
    advanced_model = AdvancedChatModel()
    
    # Test with tools
    tools = create_sample_tools()
    tool_params = ChatParams(temperature=0.7, max_tokens=100, tools=tools)
    tool_messages = [ChatMessage(role="user", content="Please calculate 2 + 2")]
    
    tool_response = advanced_model.predict(context=None, messages=tool_messages, params=tool_params)
    if tool_response.choices[0].message.tool_calls:
        print(f"Tool call: {tool_response.choices[0].message.tool_calls[0].function.name}")
        print(f"Arguments: {tool_response.choices[0].message.tool_calls[0].function.arguments}")
    
    print("✅ ChatModel functionality tests completed!")


def main():
    """Main function demonstrating ChatModel with MongoDB integration."""
    print("🤖 Genesis-Flow ChatModel MongoDB Integration Example")
    print("=" * 60)
    
    # Configure MongoDB URIs
    tracking_uri = "mongodb://localhost:27017/genesis_flow_chat_models"
    registry_uri = "mongodb://localhost:27017/genesis_flow_chat_models"
    
    print(f"🔗 MongoDB Tracking URI: {tracking_uri}")
    print(f"📝 MongoDB Registry URI: {registry_uri}")
    
    # Set MLflow URIs
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    # Test functionality first
    test_chat_model_functionality()
    
    # Create experiment
    experiment_name = f"chatmodel_experiment_{int(time.time())}"
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"\\n🧪 Created experiment: {experiment_name} (ID: {experiment_id})")
    
    # Example 1: Log Simple ChatModel
    print("\\n📝 Example 1: Logging Simple ChatModel")
    with mlflow.start_run(experiment_id=experiment_id, run_name="simple_chat_model") as run:
        # Create and log simple model
        simple_model = SimpleChatModel()
        
        # Log model parameters
        mlflow.log_param("model_type", "simple_chat")
        mlflow.log_param("supports_streaming", True)
        mlflow.log_param("supports_tools", False)
        
        # Log model with pyfunc flavor
        model_info = mlflow.pyfunc.log_model(
            python_model=simple_model,
            artifact_path="simple_chat_model",
            registered_model_name="SimpleChatModel",
            pip_requirements=[
                "mlflow",
                "pydantic"
            ]
        )
        
        print(f"✅ Simple ChatModel logged: {model_info.model_uri}")
    
    # Example 2: Log Advanced ChatModel with Tools
    print("\\n📝 Example 2: Logging Advanced ChatModel with Tools")
    with mlflow.start_run(experiment_id=experiment_id, run_name="advanced_chat_model") as run:
        # Create and log advanced model
        advanced_model = AdvancedChatModel()
        
        # Log model parameters
        mlflow.log_param("model_type", "advanced_chat")
        mlflow.log_param("supports_streaming", False)
        mlflow.log_param("supports_tools", True)
        mlflow.log_param("tool_functions", ["calculate"])
        
        # Log sample tools configuration
        tools = create_sample_tools()
        mlflow.log_dict(
            {
                "tools": [tool.to_dict() for tool in tools]
            },
            "tools_config.json"
        )
        
        # Log model with pyfunc flavor
        model_info = mlflow.pyfunc.log_model(
            python_model=advanced_model,
            artifact_path="advanced_chat_model",
            registered_model_name="AdvancedChatModel",
            pip_requirements=[
                "mlflow",
                "pydantic"
            ]
        )
        
        print(f"✅ Advanced ChatModel logged: {model_info.model_uri}")
    
    # Example 3: Model Evaluation
    print("\\n📊 Example 3: ChatModel Evaluation")
    with mlflow.start_run(experiment_id=experiment_id, run_name="chatmodel_evaluation") as run:
        # Load the simple model for evaluation
        simple_model = SimpleChatModel()
        
        # Create evaluation dataset
        eval_data = [
            {"messages": [{"role": "user", "content": "What's the weather?"}], "expected": "weather"},
            {"messages": [{"role": "user", "content": "What time is it?"}], "expected": "time"},
            {"messages": [{"role": "user", "content": "Hello there"}], "expected": "greeting"},
        ]
        
        # Evaluate responses
        total_tests = len(eval_data)
        correct_responses = 0
        
        for i, test_case in enumerate(eval_data):
            messages = [ChatMessage.model_validate(msg) for msg in test_case["messages"]]
            params = ChatParams(temperature=0.7, max_tokens=100)
            
            response = simple_model.predict(context=None, messages=messages, params=params)
            response_text = response.choices[0].message.content.lower()
            
            # Simple evaluation - check if expected keyword is in response
            if test_case["expected"] in response_text:
                correct_responses += 1
            
            print(f"Test {i+1}: Expected '{test_case['expected']}' in response")
        
        accuracy = correct_responses / total_tests
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("total_tests", total_tests)
        mlflow.log_metric("correct_responses", correct_responses)
        
        print(f"✅ Evaluation completed: {accuracy:.2%} accuracy")
    
    # Example 4: Model Registry Operations
    print("\\n📦 Example 4: Model Registry Operations")
    
    # Get the latest version of SimpleChatModel
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get latest version
        latest_version = client.get_latest_versions("SimpleChatModel", stages=["None"])[0]
        print(f"Latest SimpleChatModel version: {latest_version.version}")
        
        # Transition to Staging
        client.transition_model_version_stage(
            name="SimpleChatModel",
            version=latest_version.version,
            stage="Staging"
        )
        
        # Add model version tags
        client.set_model_version_tag(
            name="SimpleChatModel",
            version=latest_version.version,
            key="chat_model_type",
            value="conversational_ai"
        )
        
        client.set_model_version_tag(
            name="SimpleChatModel",
            version=latest_version.version,
            key="supports_streaming",
            value="true"
        )
        
        print(f"✅ Model promoted to Staging with tags")
        
    except Exception as e:
        print(f"⚠️ Registry operations: {e}")
    
    # Example 5: Model Loading and Inference
    print("\\n🔄 Example 5: Loading and Testing ChatModel")
    
    try:
        # Load model from registry
        model_uri = "models:/SimpleChatModel/Staging"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Test loaded model
        test_input = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        # Make prediction
        response = loaded_model.predict(test_input)
        print(f"✅ Loaded model response: {response}")
        
    except Exception as e:
        print(f"⚠️ Model loading: {e}")
    
    print("\\n🎉 ChatModel MongoDB Integration Example Complete!")
    print("\\n📋 Summary:")
    print("  ✅ Created and tested ChatModel implementations")
    print("  ✅ Logged ChatModels with MongoDB tracking")
    print("  ✅ Demonstrated tool calling capabilities")
    print("  ✅ Performed model evaluation")
    print("  ✅ Used model registry for version management")
    print("  ✅ Loaded and tested deployed models")
    print("\\n🔗 All data stored in MongoDB - no MLflow server required!")


if __name__ == "__main__":
    main()