#!/usr/bin/env python3
"""
Simple test script to verify ChatCompletion reconstruction from streaming chunks.
"""

import json
import sys
from typing import Any, Dict, List


class MockChatCompletionChunk:
    """Mock ChatCompletionChunk for testing."""
    
    def __init__(self, id: str, model: str, created: int, chunk_data: Dict[str, Any]):
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.system_fingerprint = "fp_test123"
        self.choices = [MockChoice(chunk_data)]
        self.usage = chunk_data.get("usage")


class MockChoice:
    """Mock choice object for testing."""
    
    def __init__(self, chunk_data: Dict[str, Any]):
        self.index = 0
        self.delta = MockDelta(chunk_data.get("delta", {}))
        self.finish_reason = chunk_data.get("finish_reason")
        self.logprobs = None


class MockDelta:
    """Mock delta object for testing."""
    
    def __init__(self, delta_data: Dict[str, Any]):
        self.content = delta_data.get("content")
        self.role = delta_data.get("role")
        self.tool_calls = delta_data.get("tool_calls", [])
        self.function_call = delta_data.get("function_call")


class MockUsage:
    """Mock usage object for testing."""
    
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def create_test_chunks() -> List[MockChatCompletionChunk]:
    """Create a series of test chunks that simulate a streaming response."""
    
    # First chunk with role
    chunk1 = MockChatCompletionChunk(
        id="chatcmpl-test123",
        model="gpt-4o-mini",
        created=1677652288,
        chunk_data={
            "delta": {"role": "assistant", "content": "Hello"},
            "finish_reason": None
        }
    )
    
    # Second chunk with content
    chunk2 = MockChatCompletionChunk(
        id="chatcmpl-test123",
        model="gpt-4o-mini", 
        created=1677652288,
        chunk_data={
            "delta": {"content": " world"},
            "finish_reason": None
        }
    )
    
    # Third chunk with more content
    chunk3 = MockChatCompletionChunk(
        id="chatcmpl-test123",
        model="gpt-4o-mini",
        created=1677652288,
        chunk_data={
            "delta": {"content": "! How can I help you today?"},
            "finish_reason": None
        }
    )
    
    # Final chunk with finish reason and usage
    chunk4 = MockChatCompletionChunk(
        id="chatcmpl-test123",
        model="gpt-4o-mini", 
        created=1677652288,
        chunk_data={
            "delta": {},
            "finish_reason": "stop",
            "usage": MockUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        }
    )
    
    return [chunk1, chunk2, chunk3, chunk4]


def test_reconstruction():
    """Test the ChatCompletion reconstruction functionality."""
    print("Testing ChatCompletion reconstruction from chunks...")
    
    # Add the MLflow directory to the path
    sys.path.insert(0, '/Users/yuki.watanabe/Workspace/claude-workspace/mlflow')
    
    try:
        from mlflow.openai.utils.streaming import (
            reconstruct_chat_completion_from_chunks,
            is_streaming_chat_completion
        )
        
        # Create test chunks
        chunks = create_test_chunks()
        
        # Test if chunks are recognized as streaming
        print(f"Chunks recognized as streaming: {is_streaming_chat_completion(chunks)}")
        
        # Reconstruct the ChatCompletion
        reconstructed = reconstruct_chat_completion_from_chunks(chunks)
        
        # Print the result
        print("\nReconstructed ChatCompletion:")
        if hasattr(reconstructed, 'model_dump'):
            # If it's a Pydantic model
            print(json.dumps(reconstructed.model_dump(), indent=2))
        elif isinstance(reconstructed, dict):
            # If it's a dict
            print(json.dumps(reconstructed, indent=2))
        else:
            # If it's something else
            print(f"Unexpected type: {type(reconstructed)}")
            print(reconstructed)
        
        # Verify key properties
        expected_content = "Hello world! How can I help you today?"
        if isinstance(reconstructed, dict):
            actual_content = reconstructed.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"\nExpected content: '{expected_content}'")
            print(f"Actual content: '{actual_content}'")
            print(f"Content matches: {expected_content == actual_content}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_reconstruction()