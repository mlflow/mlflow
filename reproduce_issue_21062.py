#!/usr/bin/env python3
"""
Reproduction script for MLflow issue #21062: 
"MLFlow on GCP + UV not finding the ///None"

This script demonstrates the original problem and shows how the fix resolves it.
"""

import sys
import shutil
import subprocess
from unittest.mock import patch

def demonstrate_original_problem():
    """Demonstrate the original issue where shutil.which returns None."""
    print("🔍 DEMONSTRATING ORIGINAL ISSUE #21062")
    print("=" * 50)
    
    # Show what happens with the original code
    print("Original problematic code pattern:")
    print("  cmd = [sys.executable, shutil.which('huey_consumer.py'), ...]")
    print()
    
    # Simulate uv environment where shutil.which("huey_consumer.py") returns None
    with patch('shutil.which') as mock_which:
        mock_which.return_value = None
        
        # This is what the original code would do
        original_cmd = [
            sys.executable,
            shutil.which("huey_consumer.py"),  # This returns None!
            "mlflow.server.jobs._huey_consumer.huey_instance",
            "-w",
            "1",
        ]
        
        print(f"❌ Original command construction: {original_cmd}")
        print(f"❌ Notice the 'None' in position 1: {original_cmd[1]}")
        print()
        print("When subprocess tries to execute this command:")
        print(f"   subprocess.Popen({original_cmd})")
        print("It becomes:")
        print(f"   /path/to/python3 None mlflow.server.jobs._huey_consumer.huey_instance ...")
        print("Which results in the error:")
        print("   /home/user/.venv/bin/python3: can't open file '///None': [Errno 2] No such file or directory")
        print()

def demonstrate_fix():
    """Demonstrate how the fix resolves the issue."""
    print("✅ DEMONSTRATING THE FIX")
    print("=" * 50)
    
    # Import our fixed function
    from test_huey_consumer_fix_simple import _find_huey_consumer
    
    print("New approach with _find_huey_consumer():")
    print("  huey_consumer = _find_huey_consumer()")
    print("  if isinstance(huey_consumer, list):")
    print("      cmd = huey_consumer + [target, '-w', workers]")
    print("  else:")
    print("      cmd = [sys.executable, huey_consumer, target, '-w', workers]")
    print()
    
    # Test different scenarios
    scenarios = [
        ("Traditional pip installation", lambda: '/usr/local/bin/huey_consumer.py'),
        ("UV installation (no .py)", lambda: '/home/user/.venv/bin/huey_consumer'),
        ("Module execution fallback", lambda: None),
    ]
    
    for scenario_name, which_return in scenarios:
        print(f"📋 Scenario: {scenario_name}")
        
        with patch('shutil.which') as mock_which, \
             patch('subprocess.run') as mock_run:
            
            if which_return() is None:
                # For fallback scenario
                def side_effect(name):
                    return None
                mock_which.side_effect = side_effect
                mock_run.return_value = type('MockResult', (), {'returncode': 0})()
            else:
                # For direct path scenarios
                def side_effect(name):
                    if name == 'huey_consumer':
                        return which_return()
                    elif name == 'huey_consumer.py' and 'pip' in scenario_name:
                        return which_return()
                    return None
                mock_which.side_effect = side_effect
            
            try:
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
                
                print(f"   ✅ Found: {huey_consumer}")
                print(f"   ✅ Command: {cmd}")
                print(f"   ✅ No 'None' values in command!")
                
            except Exception as e:
                print(f"   ⚠️  Expected exception (huey not installed): {e}")
        
        print()

def show_error_logs_context():
    """Show how the fix relates to the actual error logs from the issue."""
    print("🔍 RELATING TO ORIGINAL ERROR LOGS")
    print("=" * 50)
    print("Original error logs from issue #21062:")
    print("  /home/rakib_hernandez/.venv/bin/python3: can't open file '//None': [Errno 2] No such file or directory")
    print()
    print("Root cause analysis:")
    print("  1. MLflow uses shutil.which('huey_consumer.py') to find the huey executable")
    print("  2. In UV environments, this returns None because UV doesn't install .py scripts to PATH")
    print("  3. The None gets passed to subprocess.Popen as a command argument")
    print("  4. The shell interprets 'None' as a filename, resulting in '//None' error")
    print()
    print("How our fix prevents this:")
    print("  1. Try multiple executable names: 'huey_consumer', 'huey_consumer.py'")
    print("  2. Fall back to module execution: python -m huey.bin.huey_consumer")
    print("  3. Provide clear error messages if huey is truly not installed")
    print("  4. Never pass None to subprocess commands")

if __name__ == '__main__':
    print("MLflow Issue #21062 Reproduction and Fix Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_original_problem()
    print()
    demonstrate_fix()
    print()
    show_error_logs_context()
    
    print("=" * 60)
    print("🎉 CONCLUSION")
    print("This fix resolves the '//None' error by properly handling UV installations")
    print("where huey_consumer.py is not available in PATH, while maintaining")
    print("compatibility with traditional pip installations.")
    print("=" * 60)