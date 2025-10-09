import { useEffect } from 'react';
import { useSearchParams } from '../../../../../../common/utils/RoutingUtils';
import { useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { REQUEST_TIME_COLUMN_ID, TracesTableColumnType } from '@databricks/web-shared/genai-traces-table';
import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { START_TIME_LABEL_QUERY_PARAM_KEY } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';

const DEFAULT_EMPTY_CHECK_PAGE_SIZE = 500;

/**
 * Hook for setting the default time filter when there are no traces using the default time filter.
 */
export const useSetInitialTimeFilter = ({
  experimentId,
  isTracesEmpty,
  isTraceMetadataLoading,
}: {
  experimentId: string;
  isTracesEmpty: boolean;
  isTraceMetadataLoading: boolean;
}) => {
  const [searchParams] = useSearchParams();
  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();

  // Additional hook for fetching traces when there is no time range filters set in the
  // url params and no traces.
  const shouldFetchForEmptyCheck =
    isTracesEmpty && !isTraceMetadataLoading && !searchParams.has(START_TIME_LABEL_QUERY_PARAM_KEY);

  const { data: emptyCheckTraces, isLoading: emptyCheckLoading } = useSearchMlflowTraces({
    experimentId,
    tableSort: {
      key: REQUEST_TIME_COLUMN_ID,
      type: TracesTableColumnType.TRACE_INFO,
      asc: false,
    },
    disabled: !shouldFetchForEmptyCheck,
    limit: DEFAULT_EMPTY_CHECK_PAGE_SIZE,
    pageSize: DEFAULT_EMPTY_CHECK_PAGE_SIZE,
  });

  // Set monitoring filters based on oldest trace from empty check
  if (shouldFetchForEmptyCheck && emptyCheckTraces && emptyCheckTraces.length > 0 && !emptyCheckLoading) {
    // Since traces are sorted in descending order (newest first), the oldest trace is the last one while newest is the first one
    const oldestTrace = emptyCheckTraces[emptyCheckTraces.length - 1];

    setMonitoringFilters({
      startTimeLabel: 'CUSTOM',
      startTime: oldestTrace.request_time,
      endTime: new Date().toISOString(),
    });
  }

  // Return loading state so component can show loading skeleton
  const isInitialTimeFilterLoading = shouldFetchForEmptyCheck && emptyCheckLoading;

  return {
    isInitialTimeFilterLoading,
  };
};
