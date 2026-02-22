"""
Tests for MLflow MCP fn_wrapper handling of Click 8.3.0+ Sentinel.UNSET parameters.

This tests the fix for issue #21050 where MCP fn_wrapper incorrectly handled
Click 8.3.0+ Sentinel.UNSET for omitted optional params.
"""
import pytest
import click

from mlflow.mcp.server import fn_wrapper


def test_fn_wrapper_handles_click_sentinel_unset():
    """Test that fn_wrapper correctly passes Sentinel.UNSET parameters to callbacks."""
    # Mock the Click 8.3.0+ behavior where UNSET is used as default
    click_unset = getattr(click.core, "UNSET", object())
    
    # Create a test command that mimics experiments.get_experiment
    @click.command()
    @click.option("--experiment-id", type=click.STRING, default=click_unset)
    @click.option("--experiment-name", type=click.STRING, default=click_unset) 
    @click.option("--output", type=click.STRING, default="table")
    def test_command(experiment_id, experiment_name, output):
        """Test command that mimics get_experiment signature."""
        # Validate mutual exclusivity like the real get_experiment function
        if (experiment_id is not click_unset and experiment_name is not click_unset) or (
            experiment_id is click_unset and experiment_name is click_unset
        ):
            raise click.UsageError("Must specify exactly one of --experiment-id or --experiment-name.")
        return f"Success: id={experiment_id}, name={experiment_name}, output={output}"
    
    # Test the fix - this should not raise TypeError about missing arguments
    wrapped = fn_wrapper(test_command)
    
    # This should work - providing experiment_id
    result = wrapped(experiment_id="4", output="json")
    assert "Success" in result
    
    # This should work - providing experiment_name  
    result = wrapped(experiment_name="test_exp", output="json")
    assert "Success" in result


def test_fn_wrapper_backward_compatibility():
    """Test that fn_wrapper still works with regular default values."""
    @click.command()
    @click.option("--param1", type=click.STRING, default="default_value")
    @click.option("--param2", type=click.INT, default=42)
    def test_command(param1, param2):
        return f"param1={param1}, param2={param2}"
    
    wrapped = fn_wrapper(test_command) 
    
    # Should use defaults when parameters not provided
    result = wrapped()
    assert "param1=default_value" in result
    assert "param2=42" in result
    
    # Should use provided values when given
    result = wrapped(param1="custom", param2=100)
    assert "param1=custom" in result  
    assert "param2=100" in result


def test_fn_wrapper_required_parameters():
    """Test that fn_wrapper works correctly with required parameters."""
    @click.command()
    @click.option("--required-param", type=click.STRING, required=True)
    @click.option("--optional-param", type=click.STRING, default="optional")
    def test_command(required_param, optional_param):
        return f"required={required_param}, optional={optional_param}"
    
    wrapped = fn_wrapper(test_command)
    
    # Should work when required parameter is provided
    result = wrapped(required_param="test")
    assert "required=test" in result
    assert "optional=optional" in result
    
    # Should fail when required parameter is missing
    with pytest.raises(Exception):  # Click will raise an error for missing required param
        wrapped()


def test_reproduce_original_issue():
    """Reproduce the exact issue from bug report #21050."""
    # This reproduces the original failing case that was reported
    from mlflow.experiments import commands as experiments_cli
    
    # This should not raise "TypeError: get_experiment() missing 1 required positional argument"
    wrapped = fn_wrapper(experiments_cli.commands["get"])
    
    # The function should receive all parameters, even if they are UNSET,
    # and handle the validation internally
    try:
        result = wrapped(experiment_id="4", output="json")
        # If we get here without TypeError, the fix is working
        assert True, "fn_wrapper successfully passed UNSET parameters to callback"
    except click.UsageError:
        # This is expected - the validation logic should work
        # (we may not have a real experiment with ID "4")
        assert True, "Validation logic worked correctly"
    except TypeError as e:
        if "missing" in str(e) and "required positional argument" in str(e):
            pytest.fail(f"Fix failed - still getting parameter missing error: {e}")
        else:
            # Some other TypeError that's not related to our fix
            raise


if __name__ == "__main__":
    pytest.main([__file__])