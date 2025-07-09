#!/usr/bin/env python3
"""
Genesis-Flow Deployment Validation Script

Comprehensive validation script to verify Genesis-Flow deployment
readiness and configuration correctness before production deployment.
"""

import os
import sys
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from urllib.parse import urlparse

# Add Genesis-Flow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validation for Genesis-Flow."""
    
    def __init__(self, tracking_uri: str, artifact_root: Optional[str] = None):
        """
        Initialize deployment validator.
        
        Args:
            tracking_uri: MLflow tracking URI to validate
            artifact_root: Artifact storage root to validate
        """
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root
        self.validation_results = {
            "timestamp": time.time(),
            "tracking_uri": tracking_uri,
            "tests": {},
            "overall_status": "unknown",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }
        
    def validate_all(self) -> Dict:
        """Run all deployment validation tests."""
        logger.info("Starting Genesis-Flow deployment validation")
        logger.info(f"Tracking URI: {self.tracking_uri}")
        
        # Core connectivity tests
        self.validation_results["tests"]["connectivity"] = self._test_connectivity()
        self.validation_results["tests"]["basic_operations"] = self._test_basic_operations()
        
        # Store-specific tests
        self.validation_results["tests"]["store_configuration"] = self._test_store_configuration()
        
        # Security validation
        self.validation_results["tests"]["security"] = self._test_security_features()
        
        # Performance validation
        self.validation_results["tests"]["performance"] = self._test_performance_baseline()
        
        # Plugin system validation
        self.validation_results["tests"]["plugins"] = self._test_plugin_system()
        
        # Database/storage validation
        self.validation_results["tests"]["storage"] = self._test_storage_systems()
        
        # Configuration validation
        self.validation_results["tests"]["configuration"] = self._test_configuration()
        
        # Generate overall assessment
        self._generate_overall_assessment()
        
        return self.validation_results
    
    def _test_connectivity(self) -> Dict:
        """Test basic connectivity to tracking server."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            # Set tracking URI
            original_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Test basic connectivity
            start_time = time.time()
            experiments = mlflow.search_experiments(max_results=1)
            connection_time = time.time() - start_time
            
            test_result["details"]["connection_time"] = connection_time
            test_result["details"]["initial_experiments"] = len(experiments)
            
            if connection_time > 5.0:
                test_result["issues"].append(f"Slow connection time: {connection_time:.2f}s")
            
            test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Connection failed: {str(e)}")
            logger.error(f"Connectivity test failed: {e}")
        
        finally:
            try:
                mlflow.set_tracking_uri(original_uri)
            except:
                pass
        
        return test_result
    
    def _test_basic_operations(self) -> Dict:
        """Test basic MLflow operations."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            original_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Test experiment creation
            test_exp_name = f"deployment_validation_{int(time.time())}"
            exp_id = mlflow.create_experiment(test_exp_name)
            test_result["details"]["experiment_creation"] = "success"
            
            # Test run creation and logging
            with mlflow.start_run(experiment_id=exp_id) as run:
                run_id = run.info.run_id
                
                # Test parameter logging
                mlflow.log_param("test_param", "test_value")
                
                # Test metric logging
                mlflow.log_metric("test_metric", 0.95)
                
                # Test tag setting
                mlflow.set_tag("validation_test", "deployment_check")
            
            test_result["details"]["run_creation"] = "success"
            test_result["details"]["test_run_id"] = run_id
            
            # Test run retrieval
            retrieved_run = mlflow.get_run(run_id)
            assert retrieved_run.data.params["test_param"] == "test_value"
            assert retrieved_run.data.metrics["test_metric"] == 0.95
            
            test_result["details"]["data_retrieval"] = "success"
            
            # Test search functionality
            runs = mlflow.search_runs(experiment_ids=[exp_id])
            assert len(runs) == 1
            
            test_result["details"]["search_functionality"] = "success"
            test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Basic operations failed: {str(e)}")
            logger.error(f"Basic operations test failed: {e}")
        
        finally:
            try:
                mlflow.set_tracking_uri(original_uri)
            except:
                pass
        
        return test_result
    
    def _test_store_configuration(self) -> Dict:
        """Test store-specific configuration."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            # Detect store type
            store_type = self._detect_store_type(self.tracking_uri)
            test_result["details"]["store_type"] = store_type
            
            if store_type == "file":
                # Test file store permissions
                test_result.update(self._test_file_store())
            elif store_type == "sql":
                # Test SQL store connection
                test_result.update(self._test_sql_store())
            
            test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Store configuration test failed: {str(e)}")
        
        return test_result
    
    
    def _test_file_store(self) -> Dict:
        """Test file store permissions and structure."""
        details = {}
        issues = []
        
        try:
            # Parse file URI
            parsed = urlparse(self.tracking_uri)
            store_path = parsed.path if parsed.scheme == "file" else self.tracking_uri
            
            # Check if path exists and is writable
            if os.path.exists(store_path):
                details["store_path"] = store_path
                details["writable"] = os.access(store_path, os.W_OK)
                details["readable"] = os.access(store_path, os.R_OK)
                
                if not details["writable"]:
                    issues.append("Store path is not writable")
                    
            else:
                issues.append(f"Store path does not exist: {store_path}")
                
        except Exception as e:
            issues.append(f"File store test failed: {str(e)}")
        
        return {"details": details, "issues": issues}
    
    def _test_sql_store(self) -> Dict:
        """Test SQL store connection."""
        details = {}
        issues = []
        
        try:
            # Test SQL connection (simplified)
            parsed = urlparse(self.tracking_uri)
            details["database_type"] = parsed.scheme
            details["host"] = parsed.hostname
            details["port"] = parsed.port
            
            # Could add more detailed SQL connection tests here
            
        except Exception as e:
            issues.append(f"SQL store test failed: {str(e)}")
        
        return {"details": details, "issues": issues}
    
    def _test_security_features(self) -> Dict:
        """Test security validation features."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            from mlflow.utils.security_validation import InputValidator, SecurityValidationError
            
            # Test input validation is working
            try:
                InputValidator.validate_experiment_name("../../../etc/passwd")
                test_result["issues"].append("Path traversal validation not working")
            except SecurityValidationError:
                test_result["details"]["path_traversal_protection"] = "enabled"
            
            # Test SQL injection protection
            try:
                InputValidator.validate_metric_key("key'; DROP TABLE experiments; --")
                test_result["issues"].append("SQL injection protection not working")
            except SecurityValidationError:
                test_result["details"]["sql_injection_protection"] = "enabled"
            
            # Test secure model loading
            try:
                from mlflow.utils.secure_loading import SecureModelLoader
                test_result["details"]["secure_model_loading"] = "available"
            except ImportError:
                test_result["issues"].append("Secure model loading not available")
            
            if not test_result["issues"]:
                test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Security test failed: {str(e)}")
        
        return test_result
    
    def _test_performance_baseline(self) -> Dict:
        """Test basic performance characteristics."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            original_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Test experiment creation performance
            start_time = time.time()
            exp_id = mlflow.create_experiment(f"perf_test_{int(time.time())}")
            creation_time = time.time() - start_time
            
            test_result["details"]["experiment_creation_time"] = creation_time
            
            # Test run logging performance
            start_time = time.time()
            with mlflow.start_run(experiment_id=exp_id):
                for i in range(10):
                    mlflow.log_metric(f"metric_{i}", i * 0.1)
            logging_time = time.time() - start_time
            
            test_result["details"]["run_logging_time"] = logging_time
            test_result["details"]["metrics_per_second"] = 10 / logging_time
            
            # Performance thresholds
            if creation_time > 2.0:
                test_result["issues"].append(f"Slow experiment creation: {creation_time:.2f}s")
            
            if logging_time > 5.0:
                test_result["issues"].append(f"Slow metric logging: {logging_time:.2f}s")
            
            if not test_result["issues"]:
                test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Performance test failed: {str(e)}")
        
        finally:
            try:
                mlflow.set_tracking_uri(original_uri)
            except:
                pass
        
        return test_result
    
    def _test_plugin_system(self) -> Dict:
        """Test plugin system functionality."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            from mlflow.plugins import get_plugin_manager
            
            plugin_manager = get_plugin_manager()
            plugin_manager.initialize(auto_discover=True, auto_enable_builtin=False)
            
            # Test plugin discovery
            plugins = plugin_manager.list_plugins()
            test_result["details"]["discovered_plugins"] = [p["name"] for p in plugins]
            test_result["details"]["plugin_count"] = len(plugins)
            
            # Test plugin loading (without enabling)
            builtin_plugins = ["sklearn", "pytorch", "transformers"]
            loadable_plugins = []
            
            for plugin_name in builtin_plugins:
                try:
                    plugin = plugin_manager.get_plugin(plugin_name)
                    if plugin and plugin.check_dependencies():
                        loadable_plugins.append(plugin_name)
                except Exception:
                    pass
            
            test_result["details"]["loadable_plugins"] = loadable_plugins
            test_result["status"] = "passed"
            
        except ImportError:
            test_result["issues"].append("Plugin system not available")
        except Exception as e:
            test_result["issues"].append(f"Plugin system test failed: {str(e)}")
        
        return test_result
    
    def _test_storage_systems(self) -> Dict:
        """Test artifact storage systems."""
        test_result = {
            "status": "failed", 
            "details": {},
            "issues": [],
        }
        
        try:
            if self.artifact_root:
                # Test artifact storage
                parsed = urlparse(self.artifact_root)
                test_result["details"]["artifact_scheme"] = parsed.scheme
                
                if parsed.scheme == "file":
                    # Test file system storage
                    path = parsed.path
                    if os.path.exists(path):
                        test_result["details"]["artifact_path_exists"] = True
                        test_result["details"]["artifact_writable"] = os.access(path, os.W_OK)
                    else:
                        test_result["issues"].append(f"Artifact path does not exist: {path}")
                
                elif parsed.scheme in ["s3", "s3a"]:
                    test_result["details"]["storage_type"] = "s3"
                    # Could add S3 connectivity tests
                
                elif parsed.scheme.startswith("azure"):
                    test_result["details"]["storage_type"] = "azure_blob"
                    # Could add Azure Blob tests
            
            test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Storage test failed: {str(e)}")
        
        return test_result
    
    def _test_configuration(self) -> Dict:
        """Test configuration and environment setup."""
        test_result = {
            "status": "failed",
            "details": {},
            "issues": [],
        }
        
        try:
            # Check environment variables
            env_vars = {
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
                "MLFLOW_DEFAULT_ARTIFACT_ROOT": os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT"),
                "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            }
            
            test_result["details"]["environment_variables"] = {
                k: "set" if v else "not_set" for k, v in env_vars.items()
            }
            
            # Check Genesis-Flow version
            try:
                import mlflow
                test_result["details"]["genesis_flow_version"] = mlflow.__version__
            except:
                test_result["issues"].append("Could not determine Genesis-Flow version")
            
            # Check Python version compatibility
            import sys
            python_version = sys.version_info
            test_result["details"]["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version < (3, 8):
                test_result["issues"].append(f"Python version {python_version} may not be supported")
            
            test_result["status"] = "passed"
            
        except Exception as e:
            test_result["issues"].append(f"Configuration test failed: {str(e)}")
        
        return test_result
    
    def _detect_store_type(self, uri: str) -> str:
        """Detect store type from URI."""
        if uri.startswith("file://") or uri.startswith("/") or uri.startswith("./"):
            return "file"
        elif uri.startswith(("mysql://", "postgresql://", "sqlite://")):
            return "sql"
        else:
            return "unknown"
    
    def _generate_overall_assessment(self):
        """Generate overall deployment assessment."""
        passed_tests = 0
        total_tests = 0
        critical_failures = 0
        
        for test_name, test_result in self.validation_results["tests"].items():
            total_tests += 1
            if test_result["status"] == "passed":
                passed_tests += 1
            elif test_result["status"] == "failed":
                # Categorize critical vs non-critical failures
                if test_name in ["connectivity", "basic_operations", "security"]:
                    critical_failures += 1
                    self.validation_results["critical_issues"].extend(test_result["issues"])
                else:
                    self.validation_results["warnings"].extend(test_result["issues"])
        
        # Determine overall status
        if critical_failures > 0:
            self.validation_results["overall_status"] = "failed"
        elif passed_tests == total_tests:
            self.validation_results["overall_status"] = "passed"
        else:
            self.validation_results["overall_status"] = "warning"
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Add summary
        self.validation_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "critical_failures": critical_failures,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
        }
    
    def _generate_recommendations(self):
        """Generate deployment recommendations."""
        recommendations = []
        
        # Performance recommendations
        perf_test = self.validation_results["tests"].get("performance", {})
        if perf_test.get("details", {}).get("experiment_creation_time", 0) > 1.0:
            recommendations.append("Consider using PostgreSQL or cloud storage for better performance")
        
        # Security recommendations
        security_test = self.validation_results["tests"].get("security", {})
        if security_test.get("issues"):
            recommendations.append("Address security validation issues before production deployment")
        
        # Storage recommendations
        storage_test = self.validation_results["tests"].get("storage", {})
        if storage_test.get("details", {}).get("storage_type") == "file":
            recommendations.append("Consider cloud storage (S3/Azure Blob) for production deployments")
        
        # Plugin recommendations
        plugin_test = self.validation_results["tests"].get("plugins", {})
        loadable_plugins = plugin_test.get("details", {}).get("loadable_plugins", [])
        if len(loadable_plugins) < 2:
            recommendations.append("Consider installing ML framework dependencies for better functionality")
        
        self.validation_results["recommendations"] = recommendations
    
    def print_results(self):
        """Print formatted validation results."""
        print("\\n" + "=" * 70)
        print("GENESIS-FLOW DEPLOYMENT VALIDATION RESULTS")
        print("=" * 70)
        
        status = self.validation_results["overall_status"]
        status_symbols = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌"
        }
        
        print(f"Overall Status: {status_symbols.get(status, '?')} {status.upper()}")
        print(f"Tracking URI: {self.tracking_uri}")
        
        summary = self.validation_results["summary"]
        print(f"\\nTest Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        # Print test details
        print("\\nTest Results:")
        print("-" * 40)
        for test_name, test_result in self.validation_results["tests"].items():
            status_symbol = "✅" if test_result["status"] == "passed" else "❌"
            print(f"{status_symbol} {test_name}: {test_result['status']}")
            
            if test_result["issues"]:
                for issue in test_result["issues"]:
                    print(f"    ⚠️  {issue}")
        
        # Print critical issues
        if self.validation_results["critical_issues"]:
            print("\\nCritical Issues:")
            print("-" * 40)
            for issue in self.validation_results["critical_issues"]:
                print(f"❌ {issue}")
        
        # Print warnings
        if self.validation_results["warnings"]:
            print("\\nWarnings:")
            print("-" * 40)
            for warning in self.validation_results["warnings"]:
                print(f"⚠️  {warning}")
        
        # Print recommendations
        if self.validation_results["recommendations"]:
            print("\\nRecommendations:")
            print("-" * 40)
            for rec in self.validation_results["recommendations"]:
                print(f"💡 {rec}")

def main():
    """Main CLI interface for deployment validation."""
    parser = argparse.ArgumentParser(description="Validate Genesis-Flow deployment")
    
    parser.add_argument("--tracking-uri", required=True,
                       help="MLflow tracking URI to validate")
    parser.add_argument("--artifact-root",
                       help="Artifact storage root to validate")
    parser.add_argument("--output", "-o",
                       help="Output file for validation results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    # Create validator
    validator = DeploymentValidator(
        tracking_uri=args.tracking_uri,
        artifact_root=args.artifact_root
    )
    
    try:
        # Run validation
        results = validator.validate_all()
        
        # Print results
        validator.print_results()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\nDetailed results saved to: {args.output}")
        
        # Exit with appropriate code
        if results["overall_status"] == "failed":
            print("\\n❌ Deployment validation failed. Address critical issues before deployment.")
            sys.exit(1)
        elif results["overall_status"] == "warning":
            print("\\n⚠️  Deployment validation passed with warnings. Review recommendations.")
            sys.exit(0)
        else:
            print("\\n✅ Deployment validation passed successfully!")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Deployment validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()