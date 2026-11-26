import { Fragment, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  CloseIcon,
  Empty,
  Input,
  LinkIcon,
  ListIcon,
  SearchIcon,
  SortAscendingIcon,
  SortDescendingIcon,
  Spinner,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { Link, useNavigate } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import { useExperimentEvaluationRunsData } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { ExperimentPageTabName } from '../../../constants';
import Routes from '../../../routes';
import { EvalRunsFilterOperator, quoteFilterValue } from '../../experiment-evaluation-runs/evalRunsFilterFields';
import { EvalRunsRecentRunsChart } from '../../experiment-evaluation-runs/charts/EvalRunsRecentRunsChart';
import { formatEvalRunsNumericValue } from '../../experiment-evaluation-runs/ExperimentEvaluationRunsNumberFormat';
import { EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM } from '../../experiment-evaluation-runs/hooks/useEvalRunsFilters';
import {
  EvalRunsDatePreset,
  START_TIME_LABEL_PARAM,
} from '../../experiment-evaluation-runs/hooks/usePersistedDatePreset';
import { RunStatusIcon } from '../../../components/RunStatusIcon';
import { useDebouncedSearchInput } from '../hooks/useDebouncedSearchInput';
import { SEARCH_DEBOUNCE_MS } from '../utils/constants';

export interface DatasetUsedByRunsPanelProps {
  datasetName: string;
  experimentId: string;
  open: boolean;
  onClose: () => void;
}

const RunStatusLabel = ({ status }: { status: string }) => (
  <FormattedMessage
    defaultMessage="{status, select, SCHEDULED {Scheduled} FAILED {Failed} FINISHED {Finished} RUNNING {Running} KILLED {Killed} other {Unknown}}"
    description="Human-readable status of an evaluation run"
    values={{ status }}
  />
);

export const DatasetUsedByRunsButton = ({ onClick }: { onClick: () => void }) => (
  <Button componentId="mlflow.eval-datasets-v2.used-by-runs.open" icon={<ListIcon />} onClick={onClick}>
    <FormattedMessage
      defaultMessage="Evaluation Runs"
      description="Button opening a panel of evaluation runs that use the current dataset"
    />
  </Button>
);

export const DatasetUsedByRunsPanel = ({ datasetName, experimentId, open, onClose }: DatasetUsedByRunsPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const [committedNameFilter, setCommittedNameFilter] = useState('');
  const [createdSortDirection, setCreatedSortDirection] = useState<'asc' | 'desc'>('desc');
  const [selectedChartMetricKeys, setSelectedChartMetricKeys] = useState<string[] | undefined>();
  const nameSearch = useDebouncedSearchInput({
    committedValue: committedNameFilter,
    onCommit: setCommittedNameFilter,
    debounceMs: SEARCH_DEBOUNCE_MS,
  });
  const datasetFilter = useMemo(() => {
    const filters = [`dataset.name = ${quoteFilterValue(datasetName)}`];
    const trimmedNameFilter = committedNameFilter.trim();
    if (trimmedNameFilter) {
      filters.push(`attributes.run_name ILIKE ${quoteFilterValue(`%${trimmedNameFilter}%`)}`);
    }
    return filters.join(' AND ');
  }, [committedNameFilter, datasetName]);
  const orderBy = useMemo(
    () => [`attributes.start_time ${createdSortDirection.toUpperCase()}`],
    [createdSortDirection],
  );
  const {
    data: runs,
    error,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isLoading,
  } = useExperimentEvaluationRunsData({
    experimentId,
    enabled: open,
    filter: datasetFilter,
    orderBy,
  });
  const chartMetricKeys = useMemo(
    () => Array.from(new Set(runs.flatMap((run) => run.data.metrics?.map((metric) => metric.key) ?? []))).sort(),
    [runs],
  );

  const evaluationRunsLocation = useMemo(() => {
    const searchParams = new URLSearchParams();
    searchParams.set(
      EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM,
      JSON.stringify([
        {
          field: 'dataset.name',
          operator: EvalRunsFilterOperator.EQUALS,
          value: datasetName,
        },
      ]),
    );
    // The panel lists runs from all time; without this, the experiment's saved date preset would apply.
    searchParams.set(START_TIME_LABEL_PARAM, EvalRunsDatePreset.ALL);
    return {
      pathname: Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns),
      search: searchParams.toString(),
    };
  }, [datasetName, experimentId]);

  if (!open) {
    return null;
  }

  return (
    <aside
      aria-label={intl.formatMessage({
        defaultMessage: 'Evaluation Runs',
        description: 'Accessible label for the evaluation runs side panel',
      })}
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      <div css={{ paddingLeft: theme.spacing.lg, paddingRight: theme.spacing.lg, flexShrink: 0 }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.sm,
            paddingBottom: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Evaluation Runs"
                description="Title of the evaluation runs side panel"
              />
            </Typography.Text>
            <Typography.Text size="sm" color="secondary" ellipsis title={datasetName}>
              <FormattedMessage
                defaultMessage="Runs that use {datasetName}"
                description="Subtitle identifying the dataset used by the listed evaluation runs"
                values={{ datasetName }}
              />
            </Typography.Text>
          </div>
          <Button
            componentId="mlflow.eval-datasets-v2.used-by-runs.close"
            icon={<CloseIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Close',
              description: 'Aria label for closing the evaluation runs side panel',
            })}
            onClick={onClose}
          />
        </div>
      </div>
      <div
        css={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          paddingLeft: theme.spacing.lg,
          paddingRight: theme.spacing.lg,
          paddingTop: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) auto',
            gap: theme.spacing.sm,
            marginBottom: theme.spacing.md,
          }}
        >
          <Input
            componentId="mlflow.eval-datasets-v2.used-by-runs.name-filter"
            prefix={<SearchIcon />}
            allowClear
            value={nameSearch.input}
            onChange={(event) => nameSearch.setInput(event.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Filter by name',
              description: 'Placeholder for filtering evaluation runs by name in the dataset side panel',
            })}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter by evaluation name',
              description: 'Aria label for filtering evaluation runs by name in the dataset side panel',
            })}
          />
          <Button
            componentId="mlflow.eval-datasets-v2.used-by-runs.created-sort"
            icon={createdSortDirection === 'asc' ? <SortAscendingIcon /> : <SortDescendingIcon />}
            onClick={() => setCreatedSortDirection((direction) => (direction === 'asc' ? 'desc' : 'asc'))}
            aria-label={intl.formatMessage(
              {
                defaultMessage: 'Sort by created, {direction}',
                description: 'Accessible label for sorting evaluation runs by creation date in the dataset side panel',
              },
              {
                direction:
                  createdSortDirection === 'asc'
                    ? intl.formatMessage({
                        defaultMessage: 'oldest first',
                        description: 'Ascending creation date sort direction',
                      })
                    : intl.formatMessage({
                        defaultMessage: 'newest first',
                        description: 'Descending creation date sort direction',
                      }),
              },
            )}
          >
            <FormattedMessage
              defaultMessage="Sort: Created"
              description="Button for sorting evaluation runs by creation date"
            />
          </Button>
        </div>
        {isLoading ? (
          <div
            role="status"
            aria-label={intl.formatMessage({
              defaultMessage: 'Loading evaluation runs',
              description: 'Accessible label shown while loading runs that use a dataset',
            })}
            css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}
          >
            <Spinner />
          </div>
        ) : error ? (
          <Alert
            componentId="mlflow.eval-datasets-v2.used-by-runs.error"
            type="error"
            message={intl.formatMessage({
              defaultMessage: "Couldn't load evaluation runs",
              description: 'Error shown when evaluation runs using a dataset cannot be loaded',
            })}
            closable={false}
          />
        ) : runs.length === 0 ? (
          <div
            css={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Empty
              image={
                committedNameFilter ? (
                  <SearchIcon css={{ fontSize: 24, color: theme.colors.textSecondary }} />
                ) : (
                  <ListIcon css={{ fontSize: 24, color: theme.colors.textSecondary }} />
                )
              }
              description={
                committedNameFilter ? (
                  <FormattedMessage
                    defaultMessage="No evaluation runs match this name"
                    description="Empty state when no evaluation runs match the panel's name filter"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="No evaluation runs use this dataset yet"
                    description="Empty state when no evaluation runs use the current dataset"
                  />
                )
              }
            />
          </div>
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <EvalRunsRecentRunsChart
              compact
              runs={runs}
              metricKeys={chartMetricKeys}
              selectedMetricKeys={selectedChartMetricKeys}
              onSelectedMetricKeysChange={setSelectedChartMetricKeys}
            />
            <div
              css={{
                border: `1px solid ${theme.colors.borderDecorative}`,
                borderRadius: theme.general.borderRadiusBase,
                overflow: 'hidden',
              }}
            >
              {runs.map((run, index) => {
                const metrics = run.data.metrics ?? [];
                return (
                  <div
                    key={run.info.runUuid}
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: theme.spacing.sm,
                      padding: theme.spacing.md,
                      borderBottom: index < runs.length - 1 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                    }}
                  >
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                      <Tooltip
                        componentId="mlflow.eval-datasets-v2.used-by-runs.status-tooltip"
                        content={<RunStatusLabel status={run.info.status} />}
                      >
                        <span css={{ display: 'inline-flex', flexShrink: 0 }}>
                          <RunStatusIcon status={run.info.status} />
                        </span>
                      </Tooltip>
                      <Link
                        componentId="mlflow.eval-datasets-v2.used-by-runs.run-link"
                        to={Routes.getEvaluationRunDetailsRoute(run.info.experimentId, run.info.runUuid)}
                        title={run.info.runName || run.info.runUuid}
                        css={{
                          flex: 1,
                          minWidth: 0,
                          display: 'flex',
                          alignItems: 'center',
                          gap: theme.spacing.xs,
                          fontWeight: theme.typography.typographyBoldFontWeight,
                        }}
                      >
                        <span
                          css={{
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {run.info.runName || run.info.runUuid}
                        </span>
                        <LinkIcon css={{ flexShrink: 0, fontSize: theme.typography.fontSizeSm }} aria-hidden="true" />
                      </Link>
                    </div>
                    <div css={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: theme.spacing.sm }}>
                      <Typography.Text size="sm" color="secondary">
                        <FormattedMessage
                          defaultMessage="Created {date}"
                          description="Creation date metadata for an evaluation run"
                          values={{ date: Utils.formatTimestamp(run.info.startTime, intl) }}
                        />
                      </Typography.Text>
                      <Typography.Text size="sm" color="secondary">
                        ·
                      </Typography.Text>
                      <Typography.Text size="sm" color="secondary">
                        <FormattedMessage
                          defaultMessage="Duration {duration}"
                          description="Duration metadata for an evaluation run"
                          values={{ duration: Utils.getDuration(run.info.startTime, run.info.endTime) ?? '—' }}
                        />
                      </Typography.Text>
                    </div>
                    <div
                      css={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: theme.spacing.sm,
                      }}
                    >
                      <Typography.Text size="sm" color="secondary" ellipsis title={run.info.runUuid}>
                        <FormattedMessage
                          defaultMessage="Run ID {runId}"
                          description="Identifier metadata for an evaluation run"
                          values={{ runId: run.info.runUuid.slice(0, 8) }}
                        />
                      </Typography.Text>
                      <Tooltip
                        componentId="mlflow.eval-datasets-v2.used-by-runs.results-tooltip"
                        maxWidth={360}
                        content={
                          metrics.length > 0 ? (
                            <div
                              css={{
                                display: 'grid',
                                gridTemplateColumns: 'minmax(0, 1fr) auto',
                                columnGap: theme.spacing.md,
                                rowGap: theme.spacing.xs,
                              }}
                            >
                              {metrics.map((metric, metricIndex) => (
                                <Fragment key={`${metric.key}-${metricIndex}`}>
                                  <span css={{ overflowWrap: 'anywhere' }}>{metric.key}</span>
                                  <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
                                    {formatEvalRunsNumericValue(metric.value)}
                                  </span>
                                </Fragment>
                              ))}
                            </div>
                          ) : (
                            <FormattedMessage
                              defaultMessage="No results recorded"
                              description="Tooltip shown when an evaluation run has no result metrics"
                            />
                          )
                        }
                      >
                        <Tag
                          componentId="mlflow.eval-datasets-v2.used-by-runs.results"
                          tabIndex={0}
                          css={{ flexShrink: 0, marginRight: 0 }}
                        >
                          <FormattedMessage
                            defaultMessage="Results ({count})"
                            description="Pill showing how many evaluation results an evaluation run has"
                            values={{ count: metrics.length }}
                          />
                        </Tag>
                      </Tooltip>
                    </div>
                  </div>
                );
              })}
            </div>
            {hasNextPage && (
              <Button
                componentId="mlflow.eval-datasets-v2.used-by-runs.load-more"
                type="tertiary"
                loading={isFetching}
                onClick={() => void fetchNextPage()}
                css={{ alignSelf: 'center', marginTop: theme.spacing.md }}
              >
                <FormattedMessage
                  defaultMessage="Load more"
                  description="Button loading more evaluation runs that use the current dataset"
                />
              </Button>
            )}
          </div>
        )}
      </div>
      <div css={{ paddingLeft: theme.spacing.lg, paddingRight: theme.spacing.lg, flexShrink: 0 }}>
        <div
          css={{
            borderTop: `1px solid ${theme.colors.border}`,
            paddingTop: theme.spacing.md,
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <Button
            componentId="mlflow.eval-datasets-v2.used-by-runs.view-all"
            type="primary"
            onClick={() => navigate(evaluationRunsLocation)}
          >
            <FormattedMessage
              defaultMessage="View in evaluation runs tab"
              description="Button navigating to the evaluation runs page filtered to the current dataset"
            />
          </Button>
        </div>
      </div>
    </aside>
  );
};
