import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import {
  createTraceLocationForExperiment,
  GenAITracesTableBodySkeleton,
  useSearchMlflowTraces,
  isSqlWarehouseTimeoutError,
} from '@databricks/web-shared/genai-traces-table';
import { FormattedMessage } from '@databricks/i18n';
import { Button, DangerIcon, Empty, ParagraphSkeleton, SearchIcon } from '@databricks/design-system';
import { getNamedDateFilters } from './utils/dateUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useIntl } from '@databricks/i18n';
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
  const { refresh: refreshMonitoringConfig } = useMonitoringConfig();

  // Latch once a trace appears so polling stops even if a filter still hides it.
  const [hasSeenTrace, setHasSeenTrace] = useState(false);

  const {
    data: traces,
    isLoading,
    isFetching,
    error,
  } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    pageSize: 1,
    limit: 1,
    ...(loggedModelId ? { filterByLoggedModelId: loggedModelId } : {}),
    disabled: isCallDisabled,
    refetchInterval: isCallDisabled || hasSeenTrace ? false : EMPTY_STATE_POLL_INTERVAL_MS,
  });

  // Gate `hasAnyTraces` on a fresh fetch so `keepPreviousData` cached values
  // from prior activity don't briefly drive the render or fire `refresh()` on
  // remount (e.g. after a delete-all). One-way latch — a ref is enough.
  const initialFetchDoneRef = useRef(false);
  if (!isFetching && !initialFetchDoneRef.current) {
    initialFetchDoneRef.current = true;
  }

  const hasAnyTraces = initialFetchDoneRef.current && Boolean(traces && traces.length > 0);

  useEffect(() => {
    if (hasAnyTraces && !hasSeenTrace) {
      setHasSeenTrace(true);
      refreshMonitoringConfig();
    }
  }, [hasAnyTraces, hasSeenTrace, refreshMonitoringConfig]);

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
