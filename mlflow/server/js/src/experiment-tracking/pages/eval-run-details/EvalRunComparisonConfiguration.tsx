import { useMemo, useState } from 'react';
import { defineMessages, FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import {
  Button,
  InfoTooltip,
  ParagraphSkeleton,
  SortAscendingIcon,
  SortDescendingIcon,
  SortUnsortedIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import Utils from '../../../common/utils/Utils';
import { EvalRunComparisonDelta, type EvalRunComparisonTraceFilter } from './EvalRunComparisonDelta';
import { EvalRunComparisonRunHeader } from './EvalRunComparisonRunHeader';
import { EvalRunTraceComparisonLink } from './EvalRunTraceComparisonLink';
import {
  getEvalRunConfigurationMinimum,
  isEvalRunConfigurationMetricComplete,
  type EvalRunConfigurationMetric,
  type EvalRunConfigurationTraceStats,
} from './evalRunConfigurationData';
import type { EvalRunConfigurationTraceData } from './useEvalRunComparisonData';
import { useEvalRunConfigurationMetadata } from './useEvalRunConfigurationMetadata';

const ROW_LABELS = defineMessages({
  dataset: {
    defaultMessage: 'Dataset',
    description: 'Dataset row in evaluation run configuration comparison',
  },
  createdAt: {
    defaultMessage: 'Date created',
    description: 'Creation date row in evaluation run configuration comparison',
  },
  traceCount: {
    defaultMessage: 'Traces',
    description: 'Fetched trace count row in evaluation run configuration comparison',
  },
  totalTokens: {
    defaultMessage: 'Total tokens',
    description: 'Token usage row in evaluation run configuration comparison',
  },
  totalCost: {
    defaultMessage: 'Total cost',
    description: 'USD cost row in evaluation run configuration comparison',
  },
  errors: {
    defaultMessage: 'Errors',
    description: 'Trace execution errors row in evaluation run configuration comparison',
  },
});
// Same glyph and wording as the trace explorer's estimated cost (cost is computed from token usage and pricing).
const ESTIMATED_COST_PREFIX = '≈';
const ESTIMATED_COST_MESSAGES = defineMessages({
  tooltip: {
    defaultMessage: 'Cost is estimated from token usage and public model pricing.',
    description: 'Tooltip explaining that an eval run comparison cost is estimated, not reported by the provider',
  },
});
const METADATA_ROWS = ['dataset', 'createdAt'] as const;
const METRIC_ROWS: EvalRunConfigurationMetric[] = ['traceCount', 'totalTokens', 'totalCost', 'errors'];

export interface EvalRunComparisonConfigurationProps {
  experimentId: string;
  runs: EvalRunConfigurationTraceData[];
  noComparison: boolean;
  onRetryTraces: () => Promise<void>;
  onNavigateToTraceView?: (filter?: EvalRunComparisonTraceFilter) => void;
  enableOutcomeFilter?: boolean;
}

export interface EvalRunConfigurationMetricCellProps {
  metric: EvalRunConfigurationMetric;
  run: EvalRunConfigurationTraceData;
  baseline: EvalRunConfigurationTraceStats | undefined;
  isBaseline: boolean;
  minimum: number | undefined;
  onNavigateToTraceView?: (filter: EvalRunComparisonTraceFilter) => void;
}

type SortDirection = 'asc' | 'desc';
type SortValueKind = 'numeric' | 'date' | 'text';
type ConfigurationSortKey =
  | { kind: 'metadata'; field: (typeof METADATA_ROWS)[number] }
  | { kind: 'metric'; metric: EvalRunConfigurationMetric };

const SORT_DIRECTION_LABELS = defineMessages({
  dateDesc: {
    defaultMessage: 'Newest first',
    description: 'Active descending sort direction for a date in evaluation comparison',
  },
  dateAsc: {
    defaultMessage: 'Oldest first',
    description: 'Active ascending sort direction for a date in evaluation comparison',
  },
  textDesc: {
    defaultMessage: 'Z → A',
    description: 'Active descending sort direction for text in evaluation comparison',
  },
  textAsc: {
    defaultMessage: 'A → Z',
    description: 'Active ascending sort direction for text in evaluation comparison',
  },
  numericDesc: {
    defaultMessage: 'High → low',
    description: 'Active descending sort direction for a number in evaluation comparison',
  },
  numericAsc: {
    defaultMessage: 'Low → high',
    description: 'Active ascending sort direction for a number in evaluation comparison',
  },
});

interface EvalRunComparisonSortControlsProps {
  label: string;
  kind: SortValueKind;
  direction: SortDirection | undefined;
  onSort: (direction: SortDirection | undefined) => void;
}

const getSortDirectionLabel = (intl: IntlShape, kind: SortValueKind, direction: SortDirection) => {
  if (kind === 'date') {
    return intl.formatMessage(direction === 'desc' ? SORT_DIRECTION_LABELS.dateDesc : SORT_DIRECTION_LABELS.dateAsc);
  }
  if (kind === 'text') {
    return intl.formatMessage(direction === 'desc' ? SORT_DIRECTION_LABELS.textDesc : SORT_DIRECTION_LABELS.textAsc);
  }
  return intl.formatMessage(
    direction === 'desc' ? SORT_DIRECTION_LABELS.numericDesc : SORT_DIRECTION_LABELS.numericAsc,
  );
};

const EvalRunComparisonSortControls = ({ label, kind, direction, onSort }: EvalRunComparisonSortControlsProps) => {
  const intl = useIntl();
  const nextDirection: SortDirection | undefined =
    direction === 'desc' ? 'asc' : direction === 'asc' ? undefined : 'desc';
  const activeDirectionLabel = direction ? getSortDirectionLabel(intl, kind, direction) : undefined;
  const nextDirectionLabel = nextDirection
    ? getSortDirectionLabel(intl, kind, nextDirection)
    : intl.formatMessage({
        defaultMessage: 'Original order',
        description: 'Sort action that restores the original evaluation run order',
      });
  return (
    <Button
      componentId="mlflow.eval_run_details.compare.configuration.sort"
      size="small"
      type="tertiary"
      icon={
        <span className="eval-run-sort-icon" css={{ display: 'inline-flex', opacity: 0 }}>
          {direction === undefined ? (
            <SortUnsortedIcon aria-hidden="true" />
          ) : direction === 'desc' ? (
            <SortDescendingIcon aria-hidden="true" />
          ) : (
            <SortAscendingIcon aria-hidden="true" />
          )}
        </span>
      }
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'Sort {label} {direction}',
          description: 'Next sort direction for a comparison row',
        },
        { label, direction: nextDirectionLabel },
      )}
      aria-pressed={direction !== undefined}
      onClick={() => onSort(nextDirection)}
    >
      {activeDirectionLabel && <span css={{ whiteSpace: 'nowrap' }}>{activeDirectionLabel}</span>}
    </Button>
  );
};

const compareSortValues = (
  left: number | string | undefined,
  right: number | string | undefined,
  direction: SortDirection,
) => {
  if (left === undefined) return right === undefined ? 0 : 1;
  if (right === undefined) return -1;
  const comparison =
    typeof left === 'number' && typeof right === 'number'
      ? left - right
      : String(left).localeCompare(String(right), undefined, {
          numeric: true,
          sensitivity: 'base',
        });
  return direction === 'asc' ? comparison : -comparison;
};

const getMetadataValue = (
  query: ReturnType<typeof useEvalRunConfigurationMetadata>[number] | undefined,
  field: (typeof METADATA_ROWS)[number],
): string | number | undefined => {
  const value = query?.isError ? undefined : query?.data?.[field];
  return value === '' ? undefined : value;
};

const EvalRunConfigurationMetricCell = ({
  metric,
  run,
  baseline,
  isBaseline,
  minimum,
  onNavigateToTraceView,
}: EvalRunConfigurationMetricCellProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { stats } = run;
  const value = stats?.[metric];
  const isMinimum = value !== undefined && value === minimum;
  const baselineValue = baseline?.[metric];
  const absoluteDelta =
    !isBaseline &&
    value !== undefined &&
    baselineValue !== undefined &&
    isEvalRunConfigurationMetricComplete(stats, metric) &&
    isEvalRunConfigurationMetricComplete(baseline, metric)
      ? value - baselineValue
      : undefined;
  const coverage =
    metric === 'totalTokens' ? stats?.tracesWithTokens : metric === 'totalCost' ? stats?.tracesWithCost : undefined;
  const partial = value !== undefined && coverage !== undefined && stats !== undefined && coverage < stats.traceCount;
  const formatValue = (number: number, signed = false) => {
    const formatted = intl.formatNumber(number, {
      ...(metric === 'totalCost'
        ? {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 4,
            maximumFractionDigits: 4,
          }
        : {}),
      ...(signed ? { signDisplay: 'exceptZero' } : {}),
    });
    // Prefix cost values with ≈ to signal they are pricing-table estimates, not provider-reported.
    return metric === 'totalCost' && !signed ? `${ESTIMATED_COST_PREFIX}${formatted}` : formatted;
  };
  const traceColumnId =
    metric === 'totalTokens' ? 'tokens' : metric === 'totalCost' ? 'cost' : metric === 'errors' ? 'state' : undefined;
  const traceFilter =
    !isBaseline && traceColumnId && absoluteDelta !== undefined
      ? {
          columnId: traceColumnId,
          runUuid: run.runUuid,
          outcome: absoluteDelta < 0 ? ('improvement' as const) : ('regression' as const),
        }
      : undefined;

  return (
    <TableCell wrapContent={false} css={isMinimum ? { backgroundColor: theme.colors.backgroundSuccess } : undefined}>
      {run.status === 'loading' ? (
        <ParagraphSkeleton
          label={intl.formatMessage({
            defaultMessage: 'Loading trace statistics',
            description: 'Loading label for evaluation comparison trace statistics',
          })}
        />
      ) : (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-start',
            gap: theme.spacing.xs,
          }}
        >
          {metric === 'totalCost' && value !== undefined ? (
            <Tooltip
              componentId="mlflow.eval_run_details.compare.configuration.cost_estimate_tooltip"
              content={intl.formatMessage(ESTIMATED_COST_MESSAGES.tooltip)}
            >
              <span>
                <Typography.Text bold={isMinimum}>{formatValue(value)}</Typography.Text>
              </span>
            </Tooltip>
          ) : (
            <Typography.Text bold={isMinimum}>{value === undefined ? '—' : formatValue(value)}</Typography.Text>
          )}
          {isMinimum && (
            <span
              css={{
                position: 'absolute',
                width: 1,
                height: 1,
                overflow: 'hidden',
                clipPath: 'inset(50%)',
              }}
            >
              <FormattedMessage
                defaultMessage="Lowest value"
                description="Accessible label for tied minimum values in evaluation comparison"
              />
            </span>
          )}
          {metric !== 'traceCount' &&
            absoluteDelta !== undefined &&
            absoluteDelta !== 0 &&
            baselineValue !== undefined &&
            value !== undefined && (
              <EvalRunComparisonDelta
                currentValue={value}
                baselineValue={baselineValue}
                componentId="mlflow.eval_run_details.compare.configuration.delta"
                color={absoluteDelta < 0 ? 'lime' : 'coral'}
                onNavigateToTraceView={onNavigateToTraceView}
                filter={traceFilter}
                absoluteDifference={intl.formatMessage(
                  {
                    defaultMessage: 'Absolute difference: {delta}',
                    description: 'Tooltip for an evaluation run configuration comparison percentage change',
                  },
                  {
                    delta: formatValue(absoluteDelta, true),
                  },
                )}
              />
            )}
          {metric === 'traceCount' && stats?.isTruncated && (
            <InfoTooltip
              componentId="mlflow.eval_run_details.compare.configuration.trace_limit"
              iconTitle={intl.formatMessage({
                defaultMessage: 'Trace search limit reached',
                description: 'Accessible label for a potentially incomplete trace count',
              })}
              content={intl.formatMessage({
                defaultMessage:
                  'This run reached the trace search limit. Statistics cover only the loaded traces and may not represent the full run.',
                description: 'Per-run warning about potentially truncated configuration statistics',
              })}
            />
          )}
          {partial && (
            <>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage
                  defaultMessage="Partial"
                  description="Usage subtotal with incomplete trace metadata"
                />
              </Typography.Text>
              <InfoTooltip
                componentId="mlflow.eval_run_details.compare.configuration.coverage"
                iconTitle={intl.formatMessage({
                  defaultMessage: 'Usage coverage',
                  description: 'Accessible label for usage coverage information',
                })}
                content={intl.formatMessage(
                  {
                    defaultMessage: '{metric} available for {available, number} of {count, number} loaded traces.',
                    description: 'Coverage of recorded usage in the fetched traces for a comparison run',
                  },
                  {
                    metric: intl.formatMessage(ROW_LABELS[metric]),
                    available: coverage,
                    count: stats?.traceCount,
                  },
                )}
              />
            </>
          )}
        </div>
      )}
    </TableCell>
  );
};

export const EvalRunComparisonConfiguration = ({
  experimentId,
  runs,
  noComparison,
  onRetryTraces,
  onNavigateToTraceView,
  enableOutcomeFilter = true,
}: EvalRunComparisonConfigurationProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const metadata = useEvalRunConfigurationMetadata(
    experimentId,
    runs.map((run) => run.runUuid),
  );
  const metadataByRunUuid = useMemo(
    () => new Map(runs.map((run, index) => [run.runUuid, metadata[index]] as const)),
    [metadata, runs],
  );
  const [sortState, setSortState] = useState<{
    key: ConfigurationSortKey;
    direction: SortDirection;
  }>();
  const baselineRunUuid = runs[0]?.runUuid;
  const orderedRuns = useMemo(() => {
    if (!sortState) return runs;
    const getValue = (run: EvalRunConfigurationTraceData): number | string | undefined => {
      if (sortState.key.kind === 'metadata') {
        return getMetadataValue(metadataByRunUuid.get(run.runUuid), sortState.key.field);
      }
      return run.stats?.[sortState.key.metric];
    };
    const candidates = runs.map((run, index) => ({
      run,
      index,
      value: getValue(run),
    }));
    const baseline = candidates.shift();
    candidates.sort(
      (left, right) => compareSortValues(left.value, right.value, sortState.direction) || left.index - right.index,
    );
    return baseline ? [baseline.run, ...candidates.map(({ run }) => run)] : candidates.map(({ run }) => run);
  }, [metadataByRunUuid, runs, sortState]);
  const labelColumnStyle = { flex: `0 0 ${theme.spacing.lg * 10}px` };
  const loadingLabel = intl.formatMessage({
    defaultMessage: 'Loading configuration',
    description: 'Loading label for evaluation run configuration metadata',
  });
  const title = intl.formatMessage({
    defaultMessage: 'Run information',
    description: 'Title of the evaluation run information comparison section',
  });

  return (
    <section
      aria-label={title}
      css={{
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          flexWrap: 'wrap',
        }}
      >
        <Typography.Title level={3} withoutMargins>
          {title}
        </Typography.Title>
        {onNavigateToTraceView && !noComparison && (
          <EvalRunTraceComparisonLink onNavigateToTraceView={onNavigateToTraceView} />
        )}
      </div>
      <div
        css={{
          overflowX: 'auto',
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        <div
          css={{
            minWidth: theme.spacing.lg * 10 + runs.length * theme.spacing.lg * 12,
          }}
        >
          <Table noMinHeight>
            <TableRow isHeader>
              <TableHeader componentId="mlflow.eval_run_details.compare.configuration.label" style={labelColumnStyle}>
                <FormattedMessage
                  defaultMessage="Run"
                  description="Accessible header for run information comparison row labels"
                />
              </TableHeader>
              {orderedRuns.map((run) => {
                const query = metadataByRunUuid.get(run.runUuid);
                const name = query?.data?.runName || run.runName;
                return (
                  <TableHeader
                    key={run.runUuid}
                    componentId="mlflow.eval_run_details.compare.configuration.run"
                    wrapContent={false}
                  >
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs,
                        minWidth: 0,
                      }}
                    >
                      <EvalRunComparisonRunHeader
                        runUuid={run.runUuid}
                        runName={name}
                        isBaseline={run.runUuid === baselineRunUuid}
                        experimentId={experimentId}
                      />
                      {query?.isError && (
                        <div>
                          <Typography.Text color="secondary">
                            <FormattedMessage
                              defaultMessage="Configuration unavailable"
                              description="Run metadata could not be fetched"
                            />
                          </Typography.Text>
                          <Button
                            componentId="mlflow.eval_run_details.compare.configuration.retry"
                            type="link"
                            onClick={() => void query.refetch()}
                            aria-label={intl.formatMessage(
                              {
                                defaultMessage: 'Retry configuration for {runName}',
                                description: 'Retry fetching a run configuration',
                              },
                              { runName: name },
                            )}
                          >
                            <FormattedMessage
                              defaultMessage="Retry"
                              description="Retry failed run configuration lookup"
                            />
                          </Button>
                        </div>
                      )}
                    </div>
                  </TableHeader>
                );
              })}
            </TableRow>
            {METADATA_ROWS.map((field) => {
              const label = intl.formatMessage(ROW_LABELS[field]);
              const direction =
                sortState?.key.kind === 'metadata' && sortState.key.field === field ? sortState.direction : undefined;
              return (
                <TableRow
                  key={field}
                  css={{
                    '&:hover .eval-run-sort-icon, &:focus-within .eval-run-sort-icon': { opacity: 1 },
                  }}
                >
                  <TableCell style={labelColumnStyle}>
                    <span
                      css={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                      }}
                    >
                      <span>{label}</span>
                      <EvalRunComparisonSortControls
                        label={label}
                        kind={field === 'createdAt' ? 'date' : 'text'}
                        direction={direction}
                        onSort={(nextDirection) =>
                          setSortState(
                            nextDirection
                              ? {
                                  key: { kind: 'metadata', field },
                                  direction: nextDirection,
                                }
                              : undefined,
                          )
                        }
                      />
                    </span>
                  </TableCell>
                  {orderedRuns.map((run) => {
                    const query = metadataByRunUuid.get(run.runUuid);
                    const value = getMetadataValue(query, field);
                    return (
                      <TableCell key={run.runUuid}>
                        {query?.isLoading ? (
                          <ParagraphSkeleton label={loadingLabel} seed={`${run.runUuid}-${field}`} />
                        ) : value === undefined ? (
                          '—'
                        ) : typeof value === 'number' ? (
                          <span
                            title={intl.formatDate(value, {
                              dateStyle: 'full',
                              timeStyle: 'long',
                            })}
                          >
                            {Utils.timeSinceStr(value)}
                          </span>
                        ) : (
                          value
                        )}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
            {METRIC_ROWS.map((metric) => {
              const minimum = getEvalRunConfigurationMinimum(
                runs.map((run) => run.stats),
                metric,
              );
              const label = intl.formatMessage(ROW_LABELS[metric]);
              const direction =
                sortState?.key.kind === 'metric' && sortState.key.metric === metric ? sortState.direction : undefined;
              return (
                <TableRow
                  key={metric}
                  css={{
                    '&:hover .eval-run-sort-icon, &:focus-within .eval-run-sort-icon': { opacity: 1 },
                  }}
                >
                  <TableCell style={labelColumnStyle}>
                    <span
                      css={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                      }}
                    >
                      <span
                        css={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: theme.spacing.xs,
                        }}
                      >
                        <span>{label}</span>
                        {metric === 'totalCost' && (
                          <InfoTooltip
                            componentId="mlflow.eval_run_details.compare.configuration.cost_estimate"
                            iconTitle={intl.formatMessage({
                              defaultMessage: 'This is an estimate',
                              description: 'Accessible label for the total cost estimate tooltip',
                            })}
                            content={intl.formatMessage({
                              defaultMessage: 'This is an estimate',
                              description: 'Tooltip explaining that trace cost totals are estimates',
                            })}
                          />
                        )}
                      </span>
                      <EvalRunComparisonSortControls
                        label={label}
                        kind="numeric"
                        direction={direction}
                        onSort={(nextDirection) =>
                          setSortState(
                            nextDirection
                              ? {
                                  key: { kind: 'metric', metric },
                                  direction: nextDirection,
                                }
                              : undefined,
                          )
                        }
                      />
                    </span>
                  </TableCell>
                  {orderedRuns.map((run) => (
                    <EvalRunConfigurationMetricCell
                      key={run.runUuid}
                      metric={metric}
                      run={run}
                      baseline={runs[0]?.stats}
                      isBaseline={run.runUuid === baselineRunUuid}
                      minimum={minimum}
                      onNavigateToTraceView={enableOutcomeFilter ? onNavigateToTraceView : undefined}
                    />
                  ))}
                </TableRow>
              );
            })}
          </Table>
        </div>
      </div>
      {runs.some((run) => run.stats?.isTruncated) && (
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="At least one run reached the trace search limit. These values may not represent the full run; comparisons are hidden where coverage is incomplete."
            description="Warning about potentially truncated configuration statistics"
          />
        </Typography.Text>
      )}
      {noComparison && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Add a comparison run to load trace statistics."
            description="Trace statistics are not loaded until a comparison run is selected"
          />
        </Typography.Text>
      )}
      {runs.some((run) => run.status === 'error') && (
        <div>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Trace statistics unavailable for one or more runs."
              description="Failure loading traces for configuration comparison"
            />
          </Typography.Text>
          <Button
            componentId="mlflow.eval_run_details.compare.configuration.retry_traces"
            type="link"
            onClick={() => void onRetryTraces()}
          >
            <FormattedMessage
              defaultMessage="Retry trace data"
              description="Retry loading traces for configuration comparison"
            />
          </Button>
        </div>
      )}
    </section>
  );
};
