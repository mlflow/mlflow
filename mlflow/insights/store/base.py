"""
Base interface for MLflow Insights Store.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from mlflow.insights.models.dimensions import (
    DimensionValue,
    DimensionsDiscoveryResponse,
    NPMICalculationResponse,
)
from mlflow.insights.models.entities import Census, OperationalMetrics, QualityMetrics
from mlflow.insights.models.traffic_metrics import (
    ToolMetrics,
    TrafficLatency,
    TrafficVolume,
)
from mlflow.store.analytics.trace_correlation import NPMIResult


class InsightsStore(ABC):
    """Abstract base class for insights analytics operations."""

    @abstractmethod
    def get_traffic_volume(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = "1h",
    ) -> TrafficVolume:
        """
        Get traffic volume metrics with summary and time series.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            time_bucket_size: Time bucket size for aggregation ('1h', '6h', '1d')

        Returns:
            TrafficVolume object with summary statistics and time series
        """

    @abstractmethod
    def get_traffic_latency(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = "1h",
    ) -> TrafficLatency:
        """
        Get traffic latency metrics with summary and time series.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            time_bucket_size: Time bucket size for aggregation ('1h', '6h', '1d')

        Returns:
            TrafficLatency object with summary statistics and time series
        """

    @abstractmethod
    def get_tool_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = "1h",
    ) -> ToolMetrics:
        """
        Get tool usage metrics with summary and time series.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            time_bucket_size: Time bucket size for aggregation ('1h', '6h', '1d')

        Returns:
            ToolMetrics object with summary statistics and time series
        """

    @abstractmethod
    def calculate_npmi(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> NPMIResult:
        """
        Calculate NPMI correlation between two trace filter conditions.

        Note: This leverages the existing NPMI implementation in MLflow's
        tracking store backend.

        Args:
            experiment_ids: List of experiment IDs to analyze
            filter_string1: First filter condition (MLflow search filter syntax)
            filter_string2: Second filter condition (MLflow search filter syntax)
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            NPMIResult object with correlation values
        """

    @abstractmethod
    def get_operational_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_bucket_size: str = "1h",
    ) -> OperationalMetrics:
        """
        Get operational metrics (latency, errors, performance) for traces.

        Provides a comprehensive overview of system performance including trace counts,
        latency percentiles, error rates, and time-bucketed aggregations.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            time_bucket_size: Time bucket size for aggregation ('1h', '6h', '1d')

        Returns:
            OperationalMetrics object with performance and error statistics
        """

    @abstractmethod
    def get_quality_metrics(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sample_size: int = 100,
    ) -> QualityMetrics:
        """
        Analyze response quality metrics for traces.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            sample_size: Number of traces to sample for quality analysis

        Returns:
            QualityMetrics object with quality analysis results
        """

    @abstractmethod
    def get_dimensions_discovery(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> DimensionsDiscoveryResponse:
        """
        Discover all available dimensions for correlation analysis from actual data.

        This method analyzes the data to find:
        - Basic dimensions (status, span types, latency ranges)
        - Tool dimensions from spans
        - Tag dimensions from trace tags
        - Assessment dimensions from trace assessments

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DimensionsDiscoveryResponse with all discovered dimensions
        """

    @abstractmethod
    def calculate_dimensions_npmi(
        self,
        experiment_ids: list[str],
        dimension1: DimensionValue,
        dimension2: DimensionValue,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> NPMICalculationResponse:
        """
        Calculate NPMI correlation between two dimensions.

        This method converts dimensions to filter strings and uses the
        existing NPMI calculation infrastructure.

        Args:
            experiment_ids: List of experiment IDs to analyze
            dimension1: First dimension with parameter values
            dimension2: Second dimension with parameter values
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            NPMICalculationResponse with correlation results
        """

    @abstractmethod
    def get_error_analysis(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Analyze error patterns and potential root causes.

        Identifies common error patterns, error clustering, and potential
        root causes based on trace and span data.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of error patterns to return

        Returns:
            Dictionary containing:
            - error_patterns: Common error patterns with frequencies
            - error_clusters: Grouped errors by similarity
            - root_cause_hints: Potential root causes based on error analysis
            - affected_tools: Tools most associated with errors
        """

    @abstractmethod
    def compare_experiments(
        self,
        baseline_experiment_ids: list[str],
        comparison_experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Compare metrics between two sets of experiments.

        Useful for A/B testing, before/after analysis, or comparing
        different model versions.

        Args:
            baseline_experiment_ids: Baseline experiment IDs
            comparison_experiment_ids: Comparison experiment IDs
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary containing:
            - latency_change: Percentage change in latency metrics
            - error_rate_change: Change in error rates
            - throughput_change: Change in request throughput
            - statistical_significance: P-values for observed changes
        """

    @abstractmethod
    def get_performance_bottlenecks(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        percentile: int = 95,
    ) -> dict[str, Any]:
        """
        Identify performance bottlenecks in the system.

        Analyzes traces to find the slowest operations, tools, and
        patterns that contribute most to latency.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            percentile: Percentile to use for bottleneck identification (default: p95)

        Returns:
            Dictionary containing:
            - critical_path: Most common slow execution paths
            - slow_operations: Operations contributing most to latency
            - bottleneck_tools: Tools that are performance bottlenecks
            - optimization_opportunities: Suggested optimizations
        """

    @abstractmethod
    def generate_census(
        self,
        experiment_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        table_name: str | None = None,
    ) -> Census:
        """
        Generate a complete census of trace data.

        Args:
            experiment_ids: List of experiment IDs to analyze
            start_time: Optional start time filter
            end_time: Optional end time filter
            table_name: Optional source table name for metadata

        Returns:
            Census object with complete operational and quality metrics
        """