import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import {
  createTraceLocationForExperiment,
  GenAITracesTableBodySkeleton,
  invalidateMlflowSearchTracesCache,
  useSearchMlflowTraces,
  isSqlWarehouseTimeoutError,
} from '@databricks/web-shared/genai-traces-table';
import { FormattedMessage } from '@databricks/i18n';
import { Button, DangerIcon, Empty, ParagraphSkeleton, SearchIcon } from '@databricks/design-system';
import { getNamedDateFilters } from './utils/dateUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { useEffect, useMemo } from 'react';
import { useIntl } from '@databricks/i18n';
import { useQueryClient } from '@databricks/web-shared/query-client';
import {
  useExperimentKind,
  isGenAIExperimentKind,
} from '@mlflow/mlflow/src/experiment-tracking/utils/ExperimentKindUtils';
import { TracesViewTableNoTracesQuickstart } from '../../../traces/quickstart/TracesViewTableNoTracesQuickstart';
import {
  shouldEnableTracesTableStatePersistence,
  type ModelTraceSearchLocation,
} from '@databricks/web-shared/model-trace-explorer';

const EMPTY_STATE_POLL_INTERVAL_MS = 5000;

export const TracesV3EmptyState = (props: {
  traceSearchLocations: ModelTraceSearchLocation[];
  experimentIds: string[];
  loggedModelId?: string;
  isCallDisabled?: boolean;
}) => {
  const { experimentIds, traceSearchLocations, loggedModelId, isCallDisabled } = props;

  const intl = useIntl();
  const queryClient = useQueryClient();

  const {
    data: traces,
    isLoading,
    error,
  } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    pageSize: 1,
    limit: 1,
    ...(loggedModelId ? { filterByLoggedModelId: loggedModelId } : {}),
    disabled: isCallDisabled,
    // Poll while the empty state is visible so it auto-resolves once the
    // first trace is ingested. The component unmounts as soon as the parent
    // metadata query reports `isEmpty=false`, which stops the interval.
    refetchInterval: isCallDisabled ? false : EMPTY_STATE_POLL_INTERVAL_MS,
  });

  const hasAnyTraces = Boolean(traces && traces.length > 0);

  // When polling detects the first ingested trace, invalidate the shared search
  // traces cache so the parent metadata query re-fetches and swaps the empty
  // state for the populated table.
  useEffect(() => {
    if (hasAnyTraces) {
      invalidateMlflowSearchTracesCache({ queryClient });
    }
  }, [hasAnyTraces, queryClient]);

  // check experiment tags to see if it's genai or custom
  const { data: experimentEntity, loading: isExperimentLoading } = useGetExperimentQuery({
    experimentId: experimentIds[0],
  });
  const experiment = experimentEntity;
  const experimentKind = useExperimentKind(experiment?.tags);

  const isGenAIExperiment = experimentKind ? isGenAIExperimentKind(experimentKind) : false;

  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters({
    persist: shouldEnableTracesTableStatePersistence(),
  });

  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);

  const button = (
    <Button componentId="traces-v3-empty-state-button" onClick={() => setMonitoringFilters({ startTimeLabel: 'ALL' })}>
      <FormattedMessage defaultMessage="View All" description="View all traces button" />
    </Button>
  );

  if (isLoading || isExperimentLoading) {
    return <GenAITracesTableBodySkeleton />;
  }

  if (error) {
    const errorAsError = error instanceof Error ? error : new Error(String(error));
    return (
      <Empty
        image={<DangerIcon />}
        title={
          <FormattedMessage defaultMessage="Fetching traces failed" description="Fetching traces failed message" />
        }
        description={
          isSqlWarehouseTimeoutError(errorAsError)
            ? intl.formatMessage({
                defaultMessage:
                  'The SQL query timed out. Please retry, and if the problem persists, try selecting a larger SQL warehouse.',
                description:
                  'Traces empty state > SQL warehouse timeout error description with CTA to select larger warehouse',
              })
            : String(error)
        }
      />
    );
  }

  if (hasAnyTraces) {
    const image = <SearchIcon />;
    const description = (
      <FormattedMessage
        defaultMessage='Some traces are hidden by your time range filter: "{filterLabel}"'
        description="Message shown when traces are hidden by time filter"
        values={{
          filterLabel: (
            <strong>
              {namedDateFilters.find((namedDateFilter) => namedDateFilter.key === monitoringFilters.startTimeLabel)
                ?.label || ''}
            </strong>
          ),
        }}
      />
    );
    return (
      <Empty
        title={<FormattedMessage defaultMessage="No traces found" description="No traces found message" />}
        description={description}
        button={button}
        image={image}
      />
    );
  }
  return (
    <TracesViewTableNoTracesQuickstart
      baseComponentId="mlflow.traces"
      experimentName={experiment?.name ?? undefined}
      experimentId={experiment?.experimentId ?? undefined}
    />
  );
};
