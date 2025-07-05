#!/usr/bin/env python3
"""
Quick test to verify Genesis-Flow security validation is working
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mlflow.utils.security_validation import InputValidator, SecurityValidationError

def test_security_validation():
    """Test that security validation properly blocks malicious inputs"""
    
    print("Testing Genesis-Flow Security Validation...")
    print("=" * 50)
    
    # Test 1: Valid inputs should pass
    try:
        InputValidator.validate_experiment_name("test_experiment")
        InputValidator.validate_tag_key("model_type")
        InputValidator.validate_tag_value("classification")
        InputValidator.validate_param_key("learning_rate")
        InputValidator.validate_param_value("0.01")
        InputValidator.validate_artifact_path("model/artifacts/model.pkl")
        print("✓ Valid inputs passed validation")
    except SecurityValidationError as e:
        print(f"✗ Valid input failed: {e}")
        return False
    
    # Test 2: Path traversal should be blocked
    try:
        InputValidator.validate_experiment_name("../../../etc/passwd")
        print("✗ Path traversal was not blocked")
        return False
    except SecurityValidationError:
        print("✓ Path traversal blocked in experiment name")
    
    # Test 3: Injection attempts should be blocked
    try:
        InputValidator.validate_tag_key("key'; DROP TABLE experiments; --")
        print("✗ SQL injection was not blocked")
        return False
    except SecurityValidationError:
        print("✓ SQL injection blocked in tag key")
    
    # Test 4: Control characters should be blocked
    try:
        InputValidator.validate_tag_value("value\x00\r\n")
        print("✗ Control characters were not blocked")
        return False
    except SecurityValidationError:
        print("✓ Control characters blocked in tag value")
    
    # Test 5: Private IP addresses should be blocked
    try:
        InputValidator.validate_uri("http://192.168.1.1/malicious")
        print("✗ Private IP was not blocked")
        return False
    except SecurityValidationError:
        print("✓ Private IP blocked in URI validation")
    
    # Test 6: Localhost should be blocked for HTTP
    try:
        InputValidator.validate_uri("http://localhost:8080/exploit")
        print("✗ Localhost was not blocked")
        return False
    except SecurityValidationError:
        print("✓ Localhost blocked in URI validation")
    
    # Test 7: HTTPS should work for safe domains
    try:
        InputValidator.validate_uri("https://github.com/safe/repo")
        print("✓ Safe HTTPS URI allowed")
    except SecurityValidationError as e:
        print(f"✗ Safe HTTPS URI was blocked: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All security validation tests passed!")
    print("Genesis-Flow successfully blocks malicious inputs.")
    return True

if __name__ == "__main__":
    if test_security_validation():
        sys.exit(0)
    else:
        sys.exit(1)