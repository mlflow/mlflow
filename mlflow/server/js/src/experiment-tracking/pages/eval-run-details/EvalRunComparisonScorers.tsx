import { useMemo, useState } from 'react';
import { defineMessages, FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import {
  Button,
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
import {
  EvalRunComparisonDelta,
  formatEvalRunComparisonNumericValue,
  formatEvalRunComparisonPercentChange,
  type EvalRunComparisonTraceFilter,
} from './EvalRunComparisonDelta';
import { EvalRunComparisonRunHeader } from './EvalRunComparisonRunHeader';
import { EvalRunTraceComparisonLink } from './EvalRunTraceComparisonLink';
import { ScorerScoreBreakdown } from './ScorersGrid';
import { assessmentColumnId } from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import {
  buildEvalRunComparisonScorerRows,
  type EvalRunComparisonScorerCell,
  type EvalRunComparisonScorerRun,
} from './evalRunComparisonScorerData';

type SortDirection = 'asc' | 'desc';

const SORT_DIRECTION_LABELS = defineMessages({
  desc: {
    defaultMessage: 'High → low',
    description: 'Active descending sort direction for a scorer in evaluation comparison',
  },
  asc: {
    defaultMessage: 'Low → high',
    description: 'Active ascending sort direction for a scorer in evaluation comparison',
  },
});

const getSortDirectionLabel = (intl: IntlShape, direction: SortDirection) =>
  intl.formatMessage(SORT_DIRECTION_LABELS[direction]);

const EvalRunComparisonSortButton = ({
  label,
  direction,
  onSort,
}: {
  label: string;
  direction: SortDirection | undefined;
  onSort: (direction: SortDirection | undefined) => void;
}) => {
  const intl = useIntl();
  const nextDirection: SortDirection | undefined =
    direction === 'desc' ? 'asc' : direction === 'asc' ? undefined : 'desc';
  const activeDirectionLabel = direction ? getSortDirectionLabel(intl, direction) : undefined;
  const nextDirectionLabel = nextDirection
    ? getSortDirectionLabel(intl, nextDirection)
    : intl.formatMessage({
        defaultMessage: 'Original order',
        description: 'Sort action that restores the original evaluation run order',
      });

  return (
    <Button
      componentId="mlflow.eval_run_details.compare.scorers.sort"
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
          description: 'Next sort direction for a comparison scorer row',
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

export interface EvalRunComparisonScorerValueProps {
  cell: EvalRunComparisonScorerCell;
  run: EvalRunComparisonScorerRun;
  onNavigateToTraceView?: (filter: EvalRunComparisonTraceFilter) => void;
  assessmentName: string;
}

const EvalRunComparisonScorerValue = ({
  cell,
  run,
  onNavigateToTraceView,
  assessmentName,
}: EvalRunComparisonScorerValueProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  if (run.status === 'loading') {
    return (
      <ParagraphSkeleton
        label={intl.formatMessage({
          defaultMessage: 'Loading scorer results',
          description: 'Loading label for a comparison scorer cell',
        })}
      />
    );
  }
  const { scorer, baselineValue, delta, isBest } = cell;
  const isRate = scorer?.kind === 'rate';
  const scoreContent = (
    <span
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: theme.spacing.xs,
      }}
    >
      <Typography.Text bold={isBest}>
        {scorer?.kind === 'numeric' && scorer.value !== undefined
          ? formatEvalRunComparisonNumericValue(intl, scorer.value)
          : (scorer?.displayScore ?? '—')}
      </Typography.Text>
      {isBest && (
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
            defaultMessage="Highest pass rate"
            description="Accessible label for tied best success rates in scorer comparison"
          />
        </span>
      )}
    </span>
  );
  const valueContent =
    scorer && scorer.valueCounts.length > 0 ? (
      <Tooltip
        componentId="mlflow.eval_run_details.compare.scorers.breakdown"
        content={<ScorerScoreBreakdown scorer={scorer} />}
      >
        <Button componentId="mlflow.eval_run_details.compare.scorers.breakdown_trigger" type="link" size="small">
          {scoreContent}
        </Button>
      </Tooltip>
    ) : (
      scoreContent
    );
  const absoluteDifference =
    delta !== undefined && scorer && baselineValue !== undefined
      ? intl.formatMessage(
          {
            defaultMessage: 'Absolute difference: {delta}. {directionNote}',
            description: 'Absolute difference tooltip for an evaluation comparison scorer',
          },
          {
            delta:
              scorer.kind === 'numeric'
                ? formatEvalRunComparisonNumericValue(intl, delta, 'exceptZero')
                : formatEvalRunComparisonPercentChange(intl, delta),
            directionNote: isRate
              ? ''
              : intl.formatMessage({
                  defaultMessage: 'Higher or lower is not necessarily better.',
                  description: 'Direction is not rated for a numeric scorer difference',
                }),
          },
        )
      : undefined;
  const traceFilter =
    isRate && delta !== undefined && scorer?.value !== undefined && baselineValue !== undefined
      ? {
          columnId: assessmentColumnId(assessmentName),
          runUuid: run.runUuid,
          outcome: delta > 0 ? ('improvement' as const) : ('regression' as const),
        }
      : undefined;

  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: theme.spacing.xs,
      }}
    >
      {valueContent}
      {isRate &&
        delta !== undefined &&
        delta !== 0 &&
        scorer?.value !== undefined &&
        baselineValue !== undefined &&
        absoluteDifference !== undefined && (
          <EvalRunComparisonDelta
            currentValue={scorer.value}
            baselineValue={baselineValue}
            percentagePoints={isRate}
            componentId="mlflow.eval_run_details.compare.scorers.delta"
            color={!isRate ? 'default' : delta > 0 ? 'lime' : 'coral'}
            onNavigateToTraceView={onNavigateToTraceView}
            filter={traceFilter}
            absoluteDifference={absoluteDifference}
          />
        )}
    </div>
  );
};

export interface EvalRunComparisonScorersProps {
  experimentId: string;
  runs: EvalRunComparisonScorerRun[];
  noComparison: boolean;
  onNavigateToTraceView?: (filter?: EvalRunComparisonTraceFilter) => void;
  enableOutcomeFilter?: boolean;
}

export const EvalRunComparisonScorers = ({
  experimentId,
  runs,
  noComparison,
  onNavigateToTraceView,
  enableOutcomeFilter = true,
}: EvalRunComparisonScorersProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const rows = useMemo(() => buildEvalRunComparisonScorerRows(runs), [runs]);
  const [sortState, setSortState] = useState<{
    rowId: string;
    direction: SortDirection;
  }>();
  const baselineRunUuid = runs[0]?.runUuid;
  const orderedRuns = useMemo(() => {
    if (!sortState) return runs;
    const row = rows.find((candidate) => candidate.id === sortState.rowId);
    const candidates = runs.slice(1).map((run, index) => {
      const runIndex = runs.findIndex((candidate) => candidate.runUuid === run.runUuid);
      return { run, index: index + 1, value: row?.cells[runIndex]?.scorer?.value };
    });
    candidates.sort((left, right) => {
      if (left.value === undefined) return right.value === undefined ? left.index - right.index : 1;
      if (right.value === undefined) return -1;
      const comparison = left.value - right.value;
      return (sortState.direction === 'asc' ? comparison : -comparison) || left.index - right.index;
    });
    return runs.length > 0 ? [runs[0], ...candidates.map(({ run }) => run)] : candidates.map(({ run }) => run);
  }, [rows, runs, sortState]);
  const isLoading = runs.some((run) => run.status === 'loading');
  const hasErrors = runs.some((run) => run.status === 'error');
  const title = intl.formatMessage({
    defaultMessage: 'Scorers',
    description: 'Heading for the side-by-side evaluation run scorer table',
  });
  const labelColumnStyle = { flex: `0 0 ${theme.spacing.lg * 10}px` };
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
              <TableHeader componentId="mlflow.eval_run_details.compare.scorers.label" style={labelColumnStyle}>
                <FormattedMessage
                  defaultMessage="Scorer"
                  description="Header for scorer names in the evaluation comparison table"
                />
              </TableHeader>
              {orderedRuns.map((run) => (
                <TableHeader
                  key={run.runUuid}
                  componentId="mlflow.eval_run_details.compare.scorers.run"
                  wrapContent={false}
                >
                  <EvalRunComparisonRunHeader
                    runUuid={run.runUuid}
                    runName={run.runName}
                    isBaseline={run.runUuid === baselineRunUuid}
                    experimentId={experimentId}
                  />
                </TableHeader>
              ))}
            </TableRow>
            {rows.map((row) => {
              const label = row.assessmentInfo.displayName;
              const direction = sortState?.rowId === row.id ? sortState.direction : undefined;
              return (
                <TableRow
                  key={row.id}
                  css={{
                    '&:hover .eval-run-sort-icon, &:focus-within .eval-run-sort-icon': { opacity: 1 },
                  }}
                >
                  <TableCell style={labelColumnStyle} multiline>
                    <span
                      css={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                      }}
                    >
                      <span title={row.assessmentInfo.name}>{label}</span>
                      <EvalRunComparisonSortButton
                        label={label}
                        direction={direction}
                        onSort={(nextDirection) =>
                          setSortState(nextDirection ? { rowId: row.id, direction: nextDirection } : undefined)
                        }
                      />
                    </span>
                  </TableCell>
                  {orderedRuns.map((run) => {
                    const index = runs.findIndex((candidate) => candidate.runUuid === run.runUuid);
                    const cell = row.cells[index];
                    return (
                      <TableCell
                        key={runs[index].runUuid}
                        wrapContent={false}
                        align="left"
                        css={
                          cell.isBest
                            ? {
                                backgroundColor: theme.colors.backgroundSuccess,
                              }
                            : undefined
                        }
                      >
                        <EvalRunComparisonScorerValue
                          cell={cell}
                          run={run}
                          assessmentName={row.assessmentInfo.name}
                          onNavigateToTraceView={enableOutcomeFilter ? onNavigateToTraceView : undefined}
                        />
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </Table>
        </div>
      </div>
      {noComparison ? (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Add a comparison run to load scorer results."
            description="Scorer comparison requires another selected run"
          />
        </Typography.Text>
      ) : isLoading && rows.length === 0 ? (
        <ParagraphSkeleton
          label={intl.formatMessage({
            defaultMessage: 'Loading scorer results',
            description: 'Loading label for a comparison scorer cell',
          })}
        />
      ) : !hasErrors && rows.length === 0 ? (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No scorers found in the loaded traces."
            description="Empty state for scorer comparison"
          />
        </Typography.Text>
      ) : null}
      {hasErrors && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Scorer results unavailable for one or more runs."
            description="A trace search failed for one of the compared scorer columns"
          />
        </Typography.Text>
      )}
      {runs.some((run) => run.isTruncated) && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="At least one run reached the trace search limit. Scores may not represent the full run; affected comparisons and highest-pass-rate highlights are hidden."
            description="Warning about incomplete fetched trace windows in scorer comparison"
          />
        </Typography.Text>
      )}
    </section>
  );
};
