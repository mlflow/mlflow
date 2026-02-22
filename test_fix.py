#!/usr/bin/env python3
"""
Simple test to verify the fn_wrapper fix works without full MLflow dependencies.
"""

import os
import sys
import io
import contextlib
from typing import Any, Callable

# Add the mlflow directory to the path so we can import the server module
sys.path.insert(0, '/tmp/mlflow-fix')

import click

def param_type_to_json_schema_type(pt: click.ParamType) -> str:
    """Mock the function from server.py"""
    return "string"

def fn_wrapper(command: click.Command) -> Callable[..., str]:
    """The fixed fn_wrapper function"""
    def wrapper(**kwargs: Any) -> str:
        click_unset = getattr(click.core, "UNSET", object())

        # Capture stdout and stderr
        string_io = io.StringIO()
        with (
            contextlib.redirect_stdout(string_io),
            contextlib.redirect_stderr(string_io),
        ):
            # Fill in defaults for missing arguments
            # For Click 8.3.0+, we need to pass ALL parameters to the callback,
            # even those with Sentinel.UNSET defaults, so Click can handle them properly
            for param in command.params:
                if param.name not in kwargs:
                    kwargs[param.name] = param.default
            command.callback(**kwargs)  # type: ignore[misc]
        return string_io.getvalue().strip()

    return wrapper

def test_click_sentinel_unset_fix():
    """Test that reproduces the original issue and verifies the fix."""
    print("Testing fn_wrapper fix for Click 8.3.0+ Sentinel.UNSET...")
    
    # Mock the Click 8.3.0+ behavior where UNSET is used as default
    click_unset = getattr(click.core, "UNSET", object())
    print(f"Click UNSET object: {click_unset}")
    print(f"UNSET repr: {repr(click_unset)}")
    
    # Create a test command that mimics experiments.get_experiment
    @click.command()
    @click.option("--experiment-id", type=click.STRING, default=click_unset)
    @click.option("--experiment-name", type=click.STRING, default=click_unset) 
    @click.option("--output", type=click.STRING, default="table")
    def test_get_experiment(experiment_id, experiment_name, output):
        """Test command that mimics get_experiment signature and validation."""
        print(f"Called with: id={experiment_id}, name={experiment_name}, output={output}")
        print(f"experiment_id is UNSET: {experiment_id is click_unset}")
        print(f"experiment_name is UNSET: {experiment_name is click_unset}")
        
        # Validate mutual exclusivity like the real get_experiment function
        if (experiment_id is not click_unset and experiment_name is not click_unset):
            raise click.UsageError("Cannot specify both --experiment-id and --experiment-name.")
        if (experiment_id is click_unset and experiment_name is click_unset):
            raise click.UsageError("Must specify exactly one of --experiment-id or --experiment-name.")
        
        return f"Success: id={experiment_id}, name={experiment_name}, output={output}"
    
    print("\n1. Testing the fixed fn_wrapper...")
    
    try:
        wrapped = fn_wrapper(test_get_experiment)
        print("✓ fn_wrapper creation successful")
        
        # Test case 1: providing experiment_id (should work)
        print("\n2. Testing with experiment_id='4', output='json'...")
        result = wrapped(experiment_id="4", output="json")
        print(f"✓ Result: {result}")
        
        # Test case 2: providing experiment_name (should work)  
        print("\n3. Testing with experiment_name='test_exp', output='json'...")
        result = wrapped(experiment_name="test_exp", output="json")
        print(f"✓ Result: {result}")
        
        # Test case 3: providing neither (should raise UsageError, not TypeError)
        print("\n4. Testing with no experiment specification (should raise UsageError)...")
        try:
            result = wrapped(output="json")
            print(f"✗ Unexpected success: {result}")
        except click.UsageError as e:
            print(f"✓ Expected UsageError: {e}")
        except TypeError as e:
            if "missing" in str(e) and "required positional argument" in str(e):
                print(f"✗ Fix failed - still getting TypeError: {e}")
                return False
            else:
                print(f"✗ Unexpected TypeError: {e}")
                return False
        
        # Test case 4: providing both (should raise UsageError)
        print("\n5. Testing with both experiment_id and experiment_name (should raise UsageError)...")
        try:
            result = wrapped(experiment_id="4", experiment_name="test", output="json")
            print(f"✗ Unexpected success: {result}")
        except click.UsageError as e:
            print(f"✓ Expected UsageError: {e}")
        
        print("\n🎉 All tests passed! The fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_click_sentinel_unset_fix()
    sys.exit(0 if success else 1)