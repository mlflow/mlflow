import { useEffect, useMemo, useRef } from 'react';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { useMonitoringFilters, DEFAULT_START_TIME_LABEL } from '../../../hooks/useMonitoringFilters';
import { useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import {
  useSearchMlflowTraces,
  REQUEST_TIME_COLUMN_ID,
  TracesTableColumnType,
} from '@databricks/web-shared/genai-traces-table';
import { isDemoExperiment } from '../../../utils/isDemoExperiment';

const DEMO_TRACE_FETCH_LIMIT = 200;

/**
 * Hook that automatically sets the time range filter for demo experiments
 * based on the actual trace data timestamps.
 *
 * For demo experiments (identified by mlflow.demo.version.* tags), this hook:
 * 1. Fetches traces to determine the actual data time range
 * 2. Sets a CUSTOM time range from oldest to newest trace + 1 day buffer
 *
 * This ensures demo data is always visible regardless of when the demo was generated.
 * The hook respects explicit URL time range parameters and won't override them.
 */
export function useDemoExperimentTimeRange(experimentId: string) {
  const { data: experiment, loading: isExperimentLoading } = useGetExperimentQuery({ experimentId });

  const isDemo = useMemo(() => (experiment ? isDemoExperiment(experiment) : false), [experiment]);

  const [monitoringFilters, setMonitoringFilters, disableAutomaticInitialization] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  // Track whether we've already set the time range for demo experiments
  const hasSetDemoTimeRange = useRef(false);

  // For demo experiments, fetch traces to determine the actual data time range
  // Only fetch if:
  // - It's a demo experiment
  // - No explicit time range was set in URL (disableAutomaticInitialization is false)
  // - Current time range is the default (LAST_7_DAYS)
  // - We haven't already set the time range
  const shouldFetchDemoTraces =
    isDemo && !disableAutomaticInitialization && monitoringFilters.startTimeLabel === DEFAULT_START_TIME_LABEL;

  const { data: demoTraces, isLoading: isDemoTracesLoading } = useSearchMlflowTraces({
    locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
    tableSort: {
      key: REQUEST_TIME_COLUMN_ID,
      type: TracesTableColumnType.TRACE_INFO,
      asc: false,
    },
    disabled: !shouldFetchDemoTraces || hasSetDemoTimeRange.current,
    limit: DEMO_TRACE_FETCH_LIMIT,
    pageSize: DEMO_TRACE_FETCH_LIMIT,
  });

  // For demo experiments, set a custom time range based on actual trace timestamps
  useEffect(() => {
    if (
      shouldFetchDemoTraces &&
      !isDemoTracesLoading &&
      demoTraces &&
      demoTraces.length > 0 &&
      !hasSetDemoTimeRange.current
    ) {
      hasSetDemoTimeRange.current = true;

      // Traces are sorted in descending order (newest first), so oldest is last
      const oldestTrace = demoTraces[demoTraces.length - 1];
      const newestTrace = demoTraces[0];

      // Set custom time range from oldest trace to newest + 1 day buffer
      const startTime = oldestTrace.request_time;
      const endTime = newestTrace.request_time
        ? new Date(new Date(newestTrace.request_time).getTime() + 24 * 60 * 60 * 1000).toISOString()
        : monitoringConfig.dateNow.toISOString();

      setMonitoringFilters(
        {
          startTimeLabel: 'CUSTOM',
          startTime,
          endTime,
        },
        true,
      );
    }
  }, [shouldFetchDemoTraces, isDemoTracesLoading, demoTraces, setMonitoringFilters, monitoringConfig.dateNow]);

  return {
    isDemo,
    isLoading: isExperimentLoading || (shouldFetchDemoTraces && isDemoTracesLoading),
  };
}
