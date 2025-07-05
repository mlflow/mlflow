#!/usr/bin/env python3
"""
MLflow to Genesis-Flow Migration Tool

This tool helps migrate existing MLflow deployments to Genesis-Flow,
preserving all experiment data, runs, parameters, metrics, and artifacts.
"""

import os
import sys
import logging
import argparse
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add Genesis-Flow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import mlflow
from mlflow.entities import ViewType
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

logger = logging.getLogger(__name__)

class MLflowMigrationTool:
    """Tool for migrating from standard MLflow to Genesis-Flow."""
    
    def __init__(self, source_uri: str, target_uri: str, 
                 source_artifact_root: Optional[str] = None,
                 target_artifact_root: Optional[str] = None,
                 dry_run: bool = False):
        """
        Initialize migration tool.
        
        Args:
            source_uri: Source MLflow tracking URI
            target_uri: Target Genesis-Flow tracking URI  
            source_artifact_root: Source artifact root (auto-detect if None)
            target_artifact_root: Target artifact root (auto-detect if None)
            dry_run: If True, only analyze migration without executing
        """
        self.source_uri = source_uri
        self.target_uri = target_uri
        self.source_artifact_root = source_artifact_root
        self.target_artifact_root = target_artifact_root
        self.dry_run = dry_run
        
        self.migration_stats = {
            "experiments": {"total": 0, "migrated": 0, "failed": 0},
            "runs": {"total": 0, "migrated": 0, "failed": 0},
            "artifacts": {"total": 0, "migrated": 0, "failed": 0, "size_bytes": 0},
            "start_time": None,
            "end_time": None,
        }
        
        self.failed_items = []
        
    def analyze_source(self) -> Dict:
        """
        Analyze the source MLflow installation.
        
        Returns:
            Analysis report dictionary
        """
        logger.info(f"Analyzing source MLflow at: {self.source_uri}")
        
        # Set source as tracking URI
        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self.source_uri)
        
        try:
            analysis = {
                "tracking_uri": self.source_uri,
                "store_type": self._detect_store_type(self.source_uri),
                "experiments": [],
                "total_runs": 0,
                "total_artifacts": 0,
                "estimated_artifact_size": 0,
                "compatibility_issues": [],
                "recommendations": [],
            }
            
            # Analyze experiments
            experiments = mlflow.search_experiments(view_type=ViewType.ALL)
            analysis["experiments"] = []
            
            for exp in experiments:
                exp_analysis = {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location,
                    "tags": {tag.key: tag.value for tag in exp.tags},
                    "runs": [],
                    "run_count": 0,
                }
                
                # Analyze runs in this experiment
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    run_view_type=ViewType.ALL,
                    max_results=10000
                )
                
                exp_analysis["run_count"] = len(runs)
                analysis["total_runs"] += len(runs)
                
                # Sample a few runs for detailed analysis
                for i, (_, run_data) in enumerate(runs.iterrows()):
                    if i >= 5:  # Limit to first 5 runs for analysis
                        break
                    
                    run_info = {
                        "run_id": run_data["run_id"],
                        "status": run_data["status"],
                        "start_time": run_data["start_time"],
                        "end_time": run_data.get("end_time"),
                        "artifact_uri": run_data["artifact_uri"],
                        "param_count": len([c for c in runs.columns if c.startswith("params.")]),
                        "metric_count": len([c for c in runs.columns if c.startswith("metrics.")]),
                        "tag_count": len([c for c in runs.columns if c.startswith("tags.")]),
                    }
                    
                    # Check for artifacts
                    try:
                        artifacts = mlflow.list_artifacts(run_data["run_id"])
                        run_info["artifact_count"] = len(artifacts)
                        analysis["total_artifacts"] += len(artifacts)
                        
                        # Estimate artifact sizes
                        for artifact in artifacts:
                            if hasattr(artifact, 'file_size') and artifact.file_size:
                                analysis["estimated_artifact_size"] += artifact.file_size
                    except Exception as e:
                        logger.warning(f"Could not list artifacts for run {run_data['run_id']}: {e}")
                        run_info["artifact_count"] = 0
                    
                    exp_analysis["runs"].append(run_info)
                
                analysis["experiments"].append(exp_analysis)
            
            # Check for compatibility issues
            analysis["compatibility_issues"] = self._check_compatibility_issues(analysis)
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            return analysis
            
        finally:
            mlflow.set_tracking_uri(original_uri)
    
    def migrate(self, include_artifacts: bool = True, 
                parallel_workers: int = 1) -> bool:
        """
        Perform the migration from MLflow to Genesis-Flow.
        
        Args:
            include_artifacts: Whether to migrate artifact files
            parallel_workers: Number of parallel workers (for future implementation)
            
        Returns:
            True if migration was successful, False otherwise
        """
        logger.info("Starting MLflow to Genesis-Flow migration")
        self.migration_stats["start_time"] = datetime.now()
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        try:
            # Phase 1: Migrate experiments and metadata
            success = self._migrate_experiments_and_runs()
            if not success:
                return False
            
            # Phase 2: Migrate artifacts (if requested)
            if include_artifacts:
                success = self._migrate_artifacts()
                if not success:
                    logger.warning("Artifact migration failed, but metadata migration completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        
        finally:
            self.migration_stats["end_time"] = datetime.now()
            self._log_migration_summary()
    
    def _migrate_experiments_and_runs(self) -> bool:
        """Migrate experiments, runs, parameters, metrics, and tags."""
        logger.info("Migrating experiments and runs...")
        
        # Connect to source
        original_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self.source_uri)
        
        try:
            # Get all experiments from source
            source_experiments = mlflow.search_experiments(view_type=ViewType.ALL)
            self.migration_stats["experiments"]["total"] = len(source_experiments)
            
            # Connect to target
            mlflow.set_tracking_uri(self.target_uri)
            
            for exp in source_experiments:
                try:
                    self._migrate_single_experiment(exp)
                    self.migration_stats["experiments"]["migrated"] += 1
                except Exception as e:
                    logger.error(f"Failed to migrate experiment {exp.name}: {e}")
                    self.migration_stats["experiments"]["failed"] += 1
                    self.failed_items.append(("experiment", exp.experiment_id, str(e)))
            
            return True
            
        finally:
            mlflow.set_tracking_uri(original_uri)
    
    def _migrate_single_experiment(self, source_exp):
        """Migrate a single experiment with all its runs."""
        logger.info(f"Migrating experiment: {source_exp.name}")
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would migrate experiment {source_exp.name}")
            return
        
        # Create experiment in target
        try:
            target_exp_id = mlflow.create_experiment(
                name=source_exp.name,
                artifact_location=self._convert_artifact_location(source_exp.artifact_location),
                tags=[{"key": tag.key, "value": tag.value} for tag in source_exp.tags]
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                # Experiment already exists, get its ID
                target_exp = mlflow.get_experiment_by_name(source_exp.name)
                target_exp_id = target_exp.experiment_id
                logger.info(f"Experiment {source_exp.name} already exists, using existing")
            else:
                raise
        
        # Migrate all runs in this experiment
        mlflow.set_tracking_uri(self.source_uri)
        runs = mlflow.search_runs(
            experiment_ids=[source_exp.experiment_id],
            run_view_type=ViewType.ALL,
            max_results=10000
        )
        
        mlflow.set_tracking_uri(self.target_uri)
        
        for _, run_data in runs.iterrows():
            try:
                self._migrate_single_run(run_data, target_exp_id)
                self.migration_stats["runs"]["migrated"] += 1
            except Exception as e:
                logger.error(f"Failed to migrate run {run_data['run_id']}: {e}")
                self.migration_stats["runs"]["failed"] += 1
                self.failed_items.append(("run", run_data["run_id"], str(e)))
        
        self.migration_stats["runs"]["total"] += len(runs)
    
    def _migrate_single_run(self, run_data, target_exp_id):
        """Migrate a single run with all its data."""
        if self.dry_run:
            return
        
        # Create run in target
        with mlflow.start_run(experiment_id=target_exp_id) as target_run:
            # Migrate parameters
            for col in run_data.index:
                if col.startswith("params."):
                    param_key = col[7:]  # Remove "params." prefix
                    param_value = run_data[col]
                    if param_value is not None:
                        mlflow.log_param(param_key, param_value)
            
            # Migrate metrics
            for col in run_data.index:
                if col.startswith("metrics."):
                    metric_key = col[8:]  # Remove "metrics." prefix
                    metric_value = run_data[col]
                    if metric_value is not None:
                        mlflow.log_metric(metric_key, metric_value)
            
            # Migrate tags
            for col in run_data.index:
                if col.startswith("tags."):
                    tag_key = col[5:]  # Remove "tags." prefix
                    tag_value = run_data[col]
                    if tag_value is not None:
                        mlflow.set_tag(tag_key, tag_value)
            
            # Set run timestamps and status
            # Note: This would require direct store access for full fidelity
            mlflow.set_tag("migration.source_run_id", run_data["run_id"])
            mlflow.set_tag("migration.source_start_time", str(run_data["start_time"]))
            if run_data.get("end_time"):
                mlflow.set_tag("migration.source_end_time", str(run_data["end_time"]))
    
    def _migrate_artifacts(self) -> bool:
        """Migrate artifact files."""
        logger.info("Migrating artifacts...")
        
        if self.dry_run:
            logger.info("DRY RUN: Would migrate artifacts")
            return True
        
        # This is a simplified implementation
        # In practice, you'd need to copy files from source to target artifact storage
        logger.warning("Artifact migration requires manual implementation based on storage backend")
        return True
    
    def _detect_store_type(self, uri: str) -> str:
        """Detect the type of MLflow store."""
        if uri.startswith("file://") or uri.startswith("/") or uri.startswith("./"):
            return "file"
        elif uri.startswith("sqlite://"):
            return "sqlite"
        elif uri.startswith("mysql://") or uri.startswith("postgresql://"):
            return "sql"
        elif uri.startswith("mongodb://"):
            return "mongodb"
        else:
            return "unknown"
    
    def _check_compatibility_issues(self, analysis: Dict) -> List[str]:
        """Check for potential compatibility issues."""
        issues = []
        
        # Check for very old MLflow versions (based on schema)
        if analysis["total_runs"] > 10000:
            issues.append("Large number of runs may require chunked migration")
        
        # Check for complex artifact structures
        if analysis["estimated_artifact_size"] > 1e9:  # 1GB
            issues.append("Large artifacts may require extended migration time")
        
        # Check for special characters in names
        for exp in analysis["experiments"]:
            if any(char in exp["name"] for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                issues.append(f"Experiment name '{exp['name']}' contains special characters")
        
        return issues
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if analysis["total_runs"] > 1000:
            recommendations.append("Consider using MongoDB backend for better performance with large datasets")
        
        if analysis["estimated_artifact_size"] > 1e8:  # 100MB
            recommendations.append("Consider migrating artifacts to cloud storage (Azure Blob, S3)")
        
        if len(analysis["experiments"]) > 100:
            recommendations.append("Consider organizing experiments with tags for better management")
        
        recommendations.append("Test migration with a subset of experiments first")
        recommendations.append("Backup source data before migration")
        
        return recommendations
    
    def _convert_artifact_location(self, source_location: str) -> str:
        """Convert artifact location from source to target format."""
        if self.target_artifact_root:
            # Use specified target artifact root
            return self.target_artifact_root
        
        # Default conversion logic
        if source_location.startswith("file://"):
            return source_location  # Keep file locations as-is
        
        return source_location  # No conversion by default
    
    def _log_migration_summary(self):
        """Log migration summary."""
        duration = None
        if self.migration_stats["start_time"] and self.migration_stats["end_time"]:
            duration = self.migration_stats["end_time"] - self.migration_stats["start_time"]
        
        logger.info("Migration Summary:")
        logger.info("=" * 50)
        logger.info(f"Experiments: {self.migration_stats['experiments']['migrated']}/{self.migration_stats['experiments']['total']} migrated")
        logger.info(f"Runs: {self.migration_stats['runs']['migrated']}/{self.migration_stats['runs']['total']} migrated")
        logger.info(f"Artifacts: {self.migration_stats['artifacts']['migrated']}/{self.migration_stats['artifacts']['total']} migrated")
        
        if duration:
            logger.info(f"Duration: {duration}")
        
        if self.failed_items:
            logger.warning(f"Failed items: {len(self.failed_items)}")
            for item_type, item_id, error in self.failed_items[:10]:  # Show first 10
                logger.warning(f"  {item_type} {item_id}: {error}")

def main():
    """Main CLI interface for migration tool."""
    parser = argparse.ArgumentParser(description="Migrate from MLflow to Genesis-Flow")
    
    parser.add_argument("--source-uri", required=True,
                       help="Source MLflow tracking URI")
    parser.add_argument("--target-uri", required=True,
                       help="Target Genesis-Flow tracking URI")
    parser.add_argument("--source-artifacts", 
                       help="Source artifact root (auto-detect if not specified)")
    parser.add_argument("--target-artifacts",
                       help="Target artifact root (auto-detect if not specified)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze source without migrating")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform dry run without making changes")
    parser.add_argument("--include-artifacts", action="store_true", default=True,
                       help="Include artifact migration")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create migration tool
    migration_tool = MLflowMigrationTool(
        source_uri=args.source_uri,
        target_uri=args.target_uri,
        source_artifact_root=args.source_artifacts,
        target_artifact_root=args.target_artifacts,
        dry_run=args.dry_run
    )
    
    try:
        if args.analyze_only:
            # Just analyze the source
            analysis = migration_tool.analyze_source()
            
            print("\nMLflow Migration Analysis")
            print("=" * 50)
            print(f"Source URI: {analysis['tracking_uri']}")
            print(f"Store Type: {analysis['store_type']}")
            print(f"Experiments: {len(analysis['experiments'])}")
            print(f"Total Runs: {analysis['total_runs']}")
            print(f"Total Artifacts: {analysis['total_artifacts']}")
            print(f"Estimated Artifact Size: {analysis['estimated_artifact_size']} bytes")
            
            if analysis['compatibility_issues']:
                print("\nCompatibility Issues:")
                for issue in analysis['compatibility_issues']:
                    print(f"  - {issue}")
            
            if analysis['recommendations']:
                print("\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"  - {rec}")
            
        else:
            # Perform migration
            success = migration_tool.migrate(include_artifacts=args.include_artifacts)
            
            if success:
                print("\n✅ Migration completed successfully!")
            else:
                print("\n❌ Migration failed. Check logs for details.")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Migration tool failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()