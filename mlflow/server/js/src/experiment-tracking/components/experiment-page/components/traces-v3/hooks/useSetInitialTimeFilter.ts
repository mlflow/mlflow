import { useEffect } from 'react';
import { useSearchParams } from '../../../../../../common/utils/RoutingUtils';
import { useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { REQUEST_TIME_COLUMN_ID, TracesTableColumnType } from '@databricks/web-shared/genai-traces-table';
import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { START_TIME_LABEL_QUERY_PARAM_KEY } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import type { ModelTraceSearchLocation } from '@databricks/web-shared/model-trace-explorer';

const DEFAULT_EMPTY_CHECK_PAGE_SIZE = 500;

/**
 * Hook for setting the default time filter when there are no traces using the default time filter.
 */
export const useSetInitialTimeFilter = ({
  locations,
  isTracesEmpty,
  isTraceMetadataLoading,
  sqlWarehouseId,
  disabled = false,
}: {
  locations: ModelTraceSearchLocation[];
  isTracesEmpty: boolean;
  isTraceMetadataLoading: boolean;
  sqlWarehouseId?: string;
  disabled?: boolean;
}) => {
  const [searchParams] = useSearchParams();
  const [monitoringFilters, setMonitoringFilters, disableAutomaticInitialization] = useMonitoringFilters();

  // Additional hook for fetching traces when there is no time range filters set in the
  // url params and no traces.
  const shouldFetchForEmptyCheck = isTracesEmpty && !isTraceMetadataLoading && !disableAutomaticInitialization;

  const { data: emptyCheckTraces, isLoading: emptyCheckLoading } = useSearchMlflowTraces({
    locations,
    tableSort: {
      key: REQUEST_TIME_COLUMN_ID,
      type: TracesTableColumnType.TRACE_INFO,
      asc: false,
    },
    disabled: !shouldFetchForEmptyCheck || disabled,
    limit: DEFAULT_EMPTY_CHECK_PAGE_SIZE,
    pageSize: DEFAULT_EMPTY_CHECK_PAGE_SIZE,
    sqlWarehouseId,
  });

  // endTime must come from the response (stable across renders), not
  // `new Date()`, otherwise this effect's deps churn and refetch in a loop.
  useEffect(() => {
    if (!shouldFetchForEmptyCheck || emptyCheckLoading || !emptyCheckTraces || emptyCheckTraces.length === 0) {
      return;
    }
    const newestTrace = emptyCheckTraces[0];
    const oldestTrace = emptyCheckTraces[emptyCheckTraces.length - 1];
    setMonitoringFilters(
      {
        startTimeLabel: 'CUSTOM',
        startTime: oldestTrace.request_time,
        endTime: newestTrace.request_time,
      },
      true,
    );
  }, [shouldFetchForEmptyCheck, emptyCheckLoading, emptyCheckTraces, setMonitoringFilters]);

  // Return loading state so component can show loading skeleton
  const isInitialTimeFilterLoading = shouldFetchForEmptyCheck && emptyCheckLoading;

  return {
    isInitialTimeFilterLoading,
  };
};
