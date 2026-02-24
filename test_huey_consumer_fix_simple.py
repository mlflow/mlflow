#!/usr/bin/env python3
"""
Simple test script to verify the huey_consumer path fix for issue #21062
"""

import os
import sys
import shutil
import subprocess
from unittest.mock import patch, Mock

# Test the _find_huey_consumer function in isolation
def _find_huey_consumer():
    """
    Find the huey_consumer executable, handling different installation methods.
    
    This function handles cases where huey is installed via:
    - Traditional pip (huey_consumer.py in PATH)
    - uv or other modern package managers (huey_consumer without .py)
    - Module execution fallback
    
    Returns:
        str or list: Path to huey_consumer executable, or command list for module execution
        
    Raises:
        Exception: If huey_consumer cannot be found
    """
    # Try common executable names
    for name in ["huey_consumer", "huey_consumer.py"]:
        path = shutil.which(name)
        if path:
            return path
    
    # Try via python module execution (fallback for uv installations)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "huey.bin.huey_consumer", "--help"], 
            capture_output=True, 
            timeout=5
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "huey.bin.huey_consumer"]
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Final fallback: try to find huey_consumer via importlib
    try:
        import huey.bin.huey_consumer
        huey_path = os.path.dirname(huey.bin.huey_consumer.__file__)
        potential_script = os.path.join(huey_path, "huey_consumer.py")
        if os.path.exists(potential_script):
            return [sys.executable, potential_script]
    except ImportError:
        pass
    
    raise Exception(
        "Could not find huey_consumer executable. This may be due to using uv or another "
        "package manager that doesn't install scripts to PATH. Please ensure huey is "
        "properly installed and accessible."
    )


def test_original_issue():
    """Test that reproduces the original issue from #21062."""
    print("Testing original issue reproduction...")
    
    # Simulate the problematic case where shutil.which("huey_consumer.py") returns None
    with patch('shutil.which') as mock_which:
        mock_which.return_value = None
        
        # This should NOT result in "//None" being used in subprocess calls
        try:
            result = _find_huey_consumer()
            print(f"✅ Found huey_consumer: {result}")
            
            # Verify the result is valid (not None)
            assert result is not None
            if isinstance(result, list):
                assert result[0] == sys.executable
                assert "-m" in result
                assert "huey.bin.huey_consumer" in result
            else:
                assert isinstance(result, str)
                assert "huey_consumer" in result
                
        except Exception as e:
            # If huey is not installed, we should get a clear error message
            print(f"✅ Got expected exception when huey not found: {e}")
            assert "Could not find huey_consumer executable" in str(e)

def test_traditional_pip():
    """Test traditional pip installation scenario."""
    print("Testing traditional pip scenario...")
    
    with patch('shutil.which') as mock_which:
        mock_which.return_value = '/usr/local/bin/huey_consumer.py'
        
        result = _find_huey_consumer()
        print(f"✅ Traditional pip result: {result}")
        assert result == '/usr/local/bin/huey_consumer.py'

def test_uv_installation():
    """Test uv installation scenario."""
    print("Testing uv installation scenario...")
    
    with patch('shutil.which') as mock_which:
        def side_effect(name):
            if name == 'huey_consumer':
                return '/home/user/.venv/bin/huey_consumer'
            return None
        
        mock_which.side_effect = side_effect
        result = _find_huey_consumer()
        print(f"✅ UV installation result: {result}")
        assert result == '/home/user/.venv/bin/huey_consumer'

def test_module_execution_fallback():
    """Test module execution fallback."""
    print("Testing module execution fallback...")
    
    with patch('shutil.which') as mock_which, \
         patch('subprocess.run') as mock_run:
        
        mock_which.return_value = None
        mock_run.return_value = Mock(returncode=0)
        
        result = _find_huey_consumer()
        print(f"✅ Module execution result: {result}")
        assert result == [sys.executable, '-m', 'huey.bin.huey_consumer']

def test_command_construction():
    """Test that command construction handles the fix properly."""
    print("Testing command construction...")
    
    # Simulate the fixed command construction
    with patch('shutil.which') as mock_which:
        mock_which.return_value = '/usr/bin/huey_consumer'
        
        huey_consumer = _find_huey_consumer()
        
        if isinstance(huey_consumer, list):
            # Module execution approach
            cmd = huey_consumer + [
                "mlflow.server.jobs._huey_consumer.huey_instance",
                "-w",
                "1",
            ]
        else:
            # Direct executable approach
            cmd = [
                sys.executable,
                huey_consumer,
                "mlflow.server.jobs._huey_consumer.huey_instance",
                "-w",
                "1",
            ]
        
        print(f"✅ Constructed command: {cmd}")
        
        # Verify command doesn't contain None
        assert None not in cmd
        assert all(arg is not None for arg in cmd)
        assert any('huey_consumer' in str(arg) for arg in cmd)

if __name__ == '__main__':
    print("Running huey_consumer path fix tests...")
    print("=" * 60)
    
    test_original_issue()
    test_traditional_pip()
    test_uv_installation()
    test_module_execution_fallback()
    test_command_construction()
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED! The fix should resolve issue #21062")
    print()
    print("Summary of fix:")
    print("- Handles uv installations where huey_consumer.py is not in PATH")
    print("- Falls back to module execution: python -m huey.bin.huey_consumer")
    print("- Provides clear error messages when huey is not installed")
    print("- Prevents '//None' subprocess errors described in the issue")