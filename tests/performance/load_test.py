#!/usr/bin/env python3
"""
Genesis-Flow Performance and Load Testing

Comprehensive performance testing suite to validate Genesis-Flow
performance characteristics under various load conditions.
"""

import sys
import os
import time
import threading
import multiprocessing
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Tuple
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import mlflow
from mlflow.entities import ViewType

class PerformanceTestSuite:
    """Comprehensive performance testing for Genesis-Flow."""
    
    def __init__(self, tracking_uri: str = None, num_workers: int = 4):
        """
        Initialize performance test suite.
        
        Args:
            tracking_uri: MLflow tracking URI (uses temp dir if None)
            num_workers: Number of concurrent workers for tests
        """
        self.tracking_uri = tracking_uri
        self.num_workers = num_workers
        self.temp_dir = None
        self.results = {}
        
        if not tracking_uri:
            self.temp_dir = tempfile.mkdtemp()
            self.tracking_uri = f"file://{self.temp_dir}"
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
    def run_all_tests(self) -> Dict:
        """Run all performance tests and return results."""
        print("Genesis-Flow Performance Test Suite")
        print("=" * 50)
        
        self.results = {
            "test_timestamp": time.time(),
            "tracking_uri": self.tracking_uri,
            "num_workers": self.num_workers,
            "tests": {}
        }
        
        # Run individual test categories
        self.results["tests"]["experiment_operations"] = self.test_experiment_operations()
        self.results["tests"]["run_operations"] = self.test_run_operations()
        self.results["tests"]["metric_logging"] = self.test_metric_logging_performance()
        self.results["tests"]["parameter_logging"] = self.test_parameter_logging_performance()
        self.results["tests"]["concurrent_operations"] = self.test_concurrent_operations()
        self.results["tests"]["search_performance"] = self.test_search_performance()
        self.results["tests"]["large_data_handling"] = self.test_large_data_handling()
        self.results["tests"]["memory_usage"] = self.test_memory_usage()
        
        # Generate summary
        self.results["summary"] = self._generate_performance_summary()
        
        return self.results
    
    def test_experiment_operations(self) -> Dict:
        """Test experiment creation, retrieval, and deletion performance."""
        print("\nTesting Experiment Operations Performance...")
        
        results = {
            "creation": {"times": [], "operations_per_second": 0},
            "retrieval": {"times": [], "operations_per_second": 0},
            "search": {"times": [], "operations_per_second": 0},
        }
        
        experiment_ids = []
        
        # Test experiment creation
        print("  Testing experiment creation...")
        for i in range(50):
            start_time = time.time()
            exp_id = mlflow.create_experiment(f"perf_test_exp_{i}_{time.time()}")
            end_time = time.time()
            
            experiment_ids.append(exp_id)
            results["creation"]["times"].append(end_time - start_time)
        
        results["creation"]["operations_per_second"] = 1 / statistics.mean(results["creation"]["times"])
        
        # Test experiment retrieval
        print("  Testing experiment retrieval...")
        for exp_id in experiment_ids[:20]:  # Test subset
            start_time = time.time()
            mlflow.get_experiment(exp_id)
            end_time = time.time()
            
            results["retrieval"]["times"].append(end_time - start_time)
        
        results["retrieval"]["operations_per_second"] = 1 / statistics.mean(results["retrieval"]["times"])
        
        # Test experiment search
        print("  Testing experiment search...")
        for _ in range(10):
            start_time = time.time()
            mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            end_time = time.time()
            
            results["search"]["times"].append(end_time - start_time)
        
        results["search"]["operations_per_second"] = 1 / statistics.mean(results["search"]["times"])
        
        return results
    
    def test_run_operations(self) -> Dict:
        """Test run creation and management performance."""
        print("\nTesting Run Operations Performance...")
        
        results = {
            "creation": {"times": [], "operations_per_second": 0},
            "context_management": {"times": [], "operations_per_second": 0},
        }
        
        # Create test experiment
        experiment_id = mlflow.create_experiment(f"run_perf_test_{time.time()}")
        
        # Test run creation
        print("  Testing run creation...")
        run_ids = []
        for i in range(100):
            start_time = time.time()
            with mlflow.start_run(experiment_id=experiment_id) as run:
                run_ids.append(run.info.run_id)
            end_time = time.time()
            
            results["creation"]["times"].append(end_time - start_time)
        
        results["creation"]["operations_per_second"] = 1 / statistics.mean(results["creation"]["times"])
        
        # Test context management overhead
        print("  Testing context management...")
        for _ in range(50):
            start_time = time.time()
            with mlflow.start_run(experiment_id=experiment_id):
                pass  # Minimal operation
            end_time = time.time()
            
            results["context_management"]["times"].append(end_time - start_time)
        
        results["context_management"]["operations_per_second"] = 1 / statistics.mean(results["context_management"]["times"])
        
        return results
    
    def test_metric_logging_performance(self) -> Dict:
        """Test metric logging performance under various conditions."""
        print("\nTesting Metric Logging Performance...")
        
        results = {
            "single_metrics": {"times": [], "operations_per_second": 0},
            "batch_metrics": {"times": [], "operations_per_second": 0},
            "metric_history": {"times": [], "operations_per_second": 0},
        }
        
        experiment_id = mlflow.create_experiment(f"metric_perf_test_{time.time()}")
        
        # Test single metric logging
        print("  Testing single metric logging...")
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(1000):
                start_time = time.time()
                mlflow.log_metric(f"metric_{i % 10}", i * 0.01)
                end_time = time.time()
                
                results["single_metrics"]["times"].append(end_time - start_time)
        
        results["single_metrics"]["operations_per_second"] = 1 / statistics.mean(results["single_metrics"]["times"])
        
        # Test metric history logging
        print("  Testing metric history logging...")
        with mlflow.start_run(experiment_id=experiment_id):
            metric_times = []
            for step in range(100):
                start_time = time.time()
                mlflow.log_metric("training_loss", 1.0 - (step * 0.01), step=step)
                end_time = time.time()
                
                metric_times.append(end_time - start_time)
            
            results["metric_history"]["times"] = metric_times
        
        results["metric_history"]["operations_per_second"] = 1 / statistics.mean(results["metric_history"]["times"])
        
        return results
    
    def test_parameter_logging_performance(self) -> Dict:
        """Test parameter logging performance."""
        print("\nTesting Parameter Logging Performance...")
        
        results = {
            "parameter_logging": {"times": [], "operations_per_second": 0},
        }
        
        experiment_id = mlflow.create_experiment(f"param_perf_test_{time.time()}")
        
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(500):
                start_time = time.time()
                mlflow.log_param(f"param_{i}", f"value_{i}")
                end_time = time.time()
                
                results["parameter_logging"]["times"].append(end_time - start_time)
        
        results["parameter_logging"]["operations_per_second"] = 1 / statistics.mean(results["parameter_logging"]["times"])
        
        return results
    
    def test_concurrent_operations(self) -> Dict:
        """Test performance under concurrent load."""
        print("\nTesting Concurrent Operations Performance...")
        
        results = {
            "concurrent_experiments": {"total_time": 0, "operations_per_second": 0},
            "concurrent_runs": {"total_time": 0, "operations_per_second": 0},
            "concurrent_logging": {"total_time": 0, "operations_per_second": 0},
        }
        
        # Test concurrent experiment creation
        print("  Testing concurrent experiment creation...")
        def create_experiment(worker_id):
            exp_id = mlflow.create_experiment(f"concurrent_exp_{worker_id}_{time.time()}")
            return exp_id
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(create_experiment, i) for i in range(20)]
            exp_ids = [f.result() for f in futures]
        end_time = time.time()
        
        results["concurrent_experiments"]["total_time"] = end_time - start_time
        results["concurrent_experiments"]["operations_per_second"] = len(exp_ids) / (end_time - start_time)
        
        # Test concurrent run creation
        print("  Testing concurrent run creation...")
        experiment_id = exp_ids[0]
        
        def create_run_with_data(worker_id):
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_param(f"worker", str(worker_id))
                mlflow.log_metric(f"worker_metric", worker_id * 0.1)
                return mlflow.active_run().info.run_id
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(create_run_with_data, i) for i in range(50)]
            run_ids = [f.result() for f in futures]
        end_time = time.time()
        
        results["concurrent_runs"]["total_time"] = end_time - start_time
        results["concurrent_runs"]["operations_per_second"] = len(run_ids) / (end_time - start_time)
        
        return results
    
    def test_search_performance(self) -> Dict:
        """Test search performance with various query types."""
        print("\nTesting Search Performance...")
        
        results = {
            "experiment_search": {"times": [], "operations_per_second": 0},
            "run_search": {"times": [], "operations_per_second": 0},
            "filtered_search": {"times": [], "operations_per_second": 0},
        }
        
        # Create test data
        experiment_id = mlflow.create_experiment(f"search_perf_test_{time.time()}")
        
        # Create multiple runs with varied data
        for i in range(100):
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_param("model_type", "linear" if i % 2 == 0 else "tree")
                mlflow.log_metric("accuracy", 0.7 + (i % 30) * 0.01)
                mlflow.set_tag("version", f"v{i % 5}")
        
        # Test experiment search
        print("  Testing experiment search...")
        for _ in range(20):
            start_time = time.time()
            mlflow.search_experiments(view_type=ViewType.ALL)
            end_time = time.time()
            
            results["experiment_search"]["times"].append(end_time - start_time)
        
        results["experiment_search"]["operations_per_second"] = 1 / statistics.mean(results["experiment_search"]["times"])
        
        # Test run search
        print("  Testing run search...")
        for _ in range(20):
            start_time = time.time()
            mlflow.search_runs(experiment_ids=[experiment_id])
            end_time = time.time()
            
            results["run_search"]["times"].append(end_time - start_time)
        
        results["run_search"]["operations_per_second"] = 1 / statistics.mean(results["run_search"]["times"])
        
        # Test filtered search
        print("  Testing filtered search...")
        filters = [
            "metrics.accuracy > 0.8",
            "params.model_type = 'linear'",
            "tags.version = 'v1'",
            "metrics.accuracy > 0.9 and params.model_type = 'tree'"
        ]
        
        for filter_str in filters:
            start_time = time.time()
            mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_str)
            end_time = time.time()
            
            results["filtered_search"]["times"].append(end_time - start_time)
        
        results["filtered_search"]["operations_per_second"] = 1 / statistics.mean(results["filtered_search"]["times"])
        
        return results
    
    def test_large_data_handling(self) -> Dict:
        """Test performance with large amounts of data."""
        print("\nTesting Large Data Handling...")
        
        results = {
            "large_parameter_set": {"time": 0, "operations_per_second": 0},
            "long_metric_history": {"time": 0, "operations_per_second": 0},
            "many_tags": {"time": 0, "operations_per_second": 0},
            "artifact_handling": {"time": 0, "operations_per_second": 0},
        }
        
        experiment_id = mlflow.create_experiment(f"large_data_test_{time.time()}")
        
        # Test large parameter set
        print("  Testing large parameter set...")
        start_time = time.time()
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(1000):
                mlflow.log_param(f"param_{i:04d}", f"value_{i}")
        end_time = time.time()
        
        results["large_parameter_set"]["time"] = end_time - start_time
        results["large_parameter_set"]["operations_per_second"] = 1000 / (end_time - start_time)
        
        # Test long metric history
        print("  Testing long metric history...")
        start_time = time.time()
        with mlflow.start_run(experiment_id=experiment_id):
            for step in range(1000):
                mlflow.log_metric("training_loss", 1.0 - (step * 0.001), step=step)
                mlflow.log_metric("validation_loss", 1.1 - (step * 0.0009), step=step)
        end_time = time.time()
        
        results["long_metric_history"]["time"] = end_time - start_time
        results["long_metric_history"]["operations_per_second"] = 2000 / (end_time - start_time)
        
        # Test many tags
        print("  Testing many tags...")
        start_time = time.time()
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(500):
                mlflow.set_tag(f"tag_{i:03d}", f"tag_value_{i}")
        end_time = time.time()
        
        results["many_tags"]["time"] = end_time - start_time
        results["many_tags"]["operations_per_second"] = 500 / (end_time - start_time)
        
        # Test artifact handling
        print("  Testing artifact handling...")
        start_time = time.time()
        with mlflow.start_run(experiment_id=experiment_id):
            # Create multiple test artifacts
            import tempfile
            temp_artifact_dir = tempfile.mkdtemp()
            for i in range(20):
                artifact_content = f"Test artifact content {i}\n" * 100  # ~2KB each
                artifact_file = os.path.join(temp_artifact_dir, f"test_artifact_{i}.txt")
                with open(artifact_file, 'w') as f:
                    f.write(artifact_content)
                mlflow.log_artifact(artifact_file, f"artifacts/batch_{i//5}")
        end_time = time.time()
        
        results["artifact_handling"]["time"] = end_time - start_time
        results["artifact_handling"]["operations_per_second"] = 20 / (end_time - start_time)
        
        return results
    
    def test_memory_usage(self) -> Dict:
        """Test memory usage characteristics."""
        print("\nTesting Memory Usage...")
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            print("  psutil not available, skipping memory tests")
            return {
                "baseline_memory": 0,
                "after_experiments": 0,
                "after_runs": 0,
                "memory_growth": 0,
                "peak_memory": 0,
                "memory_efficiency": "unknown",
            }
        
        results = {
            "baseline_memory": 0,
            "after_experiments": 0,
            "after_runs": 0,
            "memory_growth": 0,
            "peak_memory": 0,
            "memory_efficiency": "good",
        }
        
        # Baseline memory
        results["baseline_memory"] = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create experiments and measure memory
        for i in range(50):
            mlflow.create_experiment(f"memory_test_exp_{i}_{time.time()}")
        
        results["after_experiments"] = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create runs and measure memory
        experiment_id = mlflow.create_experiment(f"memory_test_runs_{time.time()}")
        peak_memory = results["after_experiments"]
        
        for i in range(100):
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 0.5)
                
                # Track peak memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > peak_memory:
                    peak_memory = current_memory
        
        results["after_runs"] = process.memory_info().rss / 1024 / 1024  # MB
        results["memory_growth"] = results["after_runs"] - results["baseline_memory"]
        results["peak_memory"] = peak_memory
        
        # Determine memory efficiency
        if results["memory_growth"] > 200:  # 200MB
            results["memory_efficiency"] = "poor"
        elif results["memory_growth"] > 100:  # 100MB
            results["memory_efficiency"] = "fair"
        else:
            results["memory_efficiency"] = "good"
        
        return results
    
    def _generate_performance_summary(self) -> Dict:
        """Generate overall performance summary."""
        summary = {
            "overall_rating": "unknown",
            "bottlenecks": [],
            "recommendations": [],
            "key_metrics": {},
        }
        
        # Calculate key metrics
        tests = self.results["tests"]
        
        # Experiment operations per second
        if "experiment_operations" in tests:
            exp_ops = tests["experiment_operations"]
            summary["key_metrics"]["experiment_creation_ops_per_sec"] = exp_ops["creation"]["operations_per_second"]
            summary["key_metrics"]["experiment_retrieval_ops_per_sec"] = exp_ops["retrieval"]["operations_per_second"]
        
        # Run operations per second
        if "run_operations" in tests:
            run_ops = tests["run_operations"]
            summary["key_metrics"]["run_creation_ops_per_sec"] = run_ops["creation"]["operations_per_second"]
        
        # Metric logging performance
        if "metric_logging" in tests:
            metric_ops = tests["metric_logging"]
            summary["key_metrics"]["metric_logging_ops_per_sec"] = metric_ops["single_metrics"]["operations_per_second"]
        
        # Identify bottlenecks
        if summary["key_metrics"].get("experiment_creation_ops_per_sec", 0) < 10:
            summary["bottlenecks"].append("Slow experiment creation")
        
        if summary["key_metrics"].get("metric_logging_ops_per_sec", 0) < 100:
            summary["bottlenecks"].append("Slow metric logging")
        
        # Generate recommendations
        if "memory_usage" in tests and tests["memory_usage"]["memory_growth"] > 100:  # 100MB
            summary["recommendations"].append("High memory usage detected - consider optimizing for large workloads")
        
        if "concurrent_operations" in tests:
            concurrent_ops = tests["concurrent_operations"]
            if concurrent_ops["concurrent_runs"]["operations_per_second"] < 5:
                summary["recommendations"].append("Poor concurrent performance - consider database optimization")
        
        # Overall rating
        bottleneck_count = len(summary["bottlenecks"])
        if bottleneck_count == 0:
            summary["overall_rating"] = "excellent"
        elif bottleneck_count <= 2:
            summary["overall_rating"] = "good"
        elif bottleneck_count <= 4:
            summary["overall_rating"] = "fair"
        else:
            summary["overall_rating"] = "poor"
        
        return summary
    
    def print_results(self):
        """Print formatted test results."""
        print("\n" + "=" * 70)
        print("GENESIS-FLOW PERFORMANCE TEST RESULTS")
        print("=" * 70)
        
        summary = self.results["summary"]
        print(f"Overall Rating: {summary['overall_rating'].upper()}")
        print(f"Test URI: {self.tracking_uri}")
        print(f"Workers: {self.num_workers}")
        
        print("\nKey Performance Metrics:")
        print("-" * 40)
        for metric, value in summary["key_metrics"].items():
            print(f"{metric}: {value:.2f}")
        
        if summary["bottlenecks"]:
            print("\nIdentified Bottlenecks:")
            print("-" * 40)
            for bottleneck in summary["bottlenecks"]:
                print(f"⚠️  {bottleneck}")
        
        if summary["recommendations"]:
            print("\nRecommendations:")
            print("-" * 40)
            for rec in summary["recommendations"]:
                print(f"💡 {rec}")
        
        print("\nDetailed Results Available in JSON output")
    
    def cleanup(self):
        """Clean up test resources."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def main():
    """Main CLI interface for performance testing."""
    parser = argparse.ArgumentParser(description="Genesis-Flow Performance Testing")
    
    parser.add_argument("--tracking-uri",
                       help="MLflow tracking URI (uses temporary directory if not specified)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of concurrent workers")
    parser.add_argument("--output", "-o",
                       help="Output file for JSON results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = PerformanceTestSuite(
        tracking_uri=args.tracking_uri,
        num_workers=args.workers
    )
    
    try:
        # Run tests
        results = test_suite.run_all_tests()
        
        # Print results
        test_suite.print_results()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
        
        # Exit with appropriate code
        overall_rating = results["summary"]["overall_rating"]
        if overall_rating in ["excellent", "good"]:
            sys.exit(0)
        else:
            print("\n⚠️  Performance issues detected. Review recommendations.")
            sys.exit(1)
    
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    main()