import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { FormattedMessage } from '@databricks/i18n';
import { Button, DangerIcon, Empty, ParagraphSkeleton, SearchIcon } from '@databricks/design-system';
import { getNamedDateFilters } from './utils/dateUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { useMemo } from 'react';
import { useIntl } from '@databricks/i18n';
import { getExperimentKindFromTags } from '@mlflow/mlflow/src/experiment-tracking/utils/ExperimentKindUtils';
import { ExperimentKind } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { TracesViewTableNoTracesQuickstart } from '../../../traces/quickstart/TracesViewTableNoTracesQuickstart';

export const TracesV3EmptyState = (props: { experimentIds: string[]; loggedModelId?: string }) => {
  const { experimentIds, loggedModelId } = props;

  const intl = useIntl();

  const {
    data: traces,
    isLoading,
    error,
  } = useSearchMlflowTraces({
    experimentId: experimentIds[0],
    pageSize: 1,
    limit: 1,
    ...(loggedModelId ? { filterByLoggedModelId: loggedModelId } : {}),
  });

  // check experiment tags to see if it's genai or custom
  const { data: experimentEntity, loading: isExperimentLoading } = useGetExperimentQuery({
    experimentId: experimentIds[0],
  });
  const experiment = experimentEntity;
  const experimentKind = getExperimentKindFromTags(experiment?.tags);

  const isGenAIExperiment =
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT || experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;

  const hasMoreTraces = traces && traces.length > 0;

  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();

  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);

  const button = (
    <Button componentId="traces-v3-empty-state-button" onClick={() => setMonitoringFilters({ startTimeLabel: 'ALL' })}>
      <FormattedMessage defaultMessage="View All" description="View all traces button" />
    </Button>
  );

  if (isLoading || isExperimentLoading) {
    return (
      <>
        {[...Array(10).keys()].map((i) => (
          <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
        ))}
      </>
    );
  }

  if (error) {
    return (
      <Empty
        image={<DangerIcon />}
        title={
          <FormattedMessage defaultMessage="Fetching traces failed" description="Fetching traces failed message" />
        }
        description={String(error)}
      />
    );
  }

  if (hasMoreTraces) {
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
  return <TracesViewTableNoTracesQuickstart baseComponentId="mlflow.traces" />;
};
