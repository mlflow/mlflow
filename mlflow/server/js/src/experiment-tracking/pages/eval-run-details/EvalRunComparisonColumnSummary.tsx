import { useState } from 'react';
import { ArrowRightIcon, DropdownMenu, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { defineMessages, useIntl, type IntlShape } from 'react-intl';
import type { TraceComparisonRow } from '@databricks/web-shared/traces-table';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { formatEvalRunComparisonPercentChange } from './EvalRunComparisonDelta';
import type { EvalRunComparisonColumnAggregate, EvalRunComparisonOutcome } from './evalRunComparisonColumnAggregates';

const deltaSentenceMessages = defineMessages({
  rateDelta: {
    defaultMessage: 'The delta with baseline run is {delta} percentage points.',
    description: 'Percentage-point difference in a comparison run rate summary',
  },
  delta: {
    defaultMessage: 'The delta with baseline run is {delta}{percent}.',
    description: 'Delta and optional percent difference in a comparison run summary',
  },
  percent: {
    defaultMessage: 'The percent difference with baseline run is {percent}.',
    description: 'Relative percent change in a comparison run summary',
  },
});

export interface EvalRunComparisonColumnSummaryProps {
  aggregate: EvalRunComparisonColumnAggregate;
  runs: Pick<TraceComparisonRow['runs'][number], 'runUuid' | 'label' | 'color'>[];
  isGroupedBySession?: boolean;
  columnLabel?: string;
  higherIsBetter?: boolean;
  filterOptionCounts?: ReadonlyArray<EvalRunComparisonColumnAggregate['values'][number]['changeCounts']>;
  onOutcomeClick?: (runUuid: string, outcome: EvalRunComparisonOutcome) => void;
  activeOutcomes?: Readonly<Record<string, EvalRunComparisonOutcome>>;
}

export interface EvalRunComparisonChangeCountProps {
  count: number;
  improved: boolean;
  isGroupedBySession?: boolean;
}

const getChangeCountLabel = (intl: IntlShape, count: number, improved: boolean, isGroupedBySession?: boolean) =>
  isGroupedBySession
    ? improved
      ? intl.formatMessage(
          {
            defaultMessage:
              '{count, plural, one {# improved comparison group} other {# improved comparison groups}} compared with baseline',
            description: 'Count of matched session summaries or standalone traces that improved against baseline',
          },
          { count },
        )
      : intl.formatMessage(
          {
            defaultMessage:
              '{count, plural, one {# regressed comparison group} other {# regressed comparison groups}} compared with baseline',
            description: 'Count of matched session summaries or standalone traces that regressed against baseline',
          },
          { count },
        )
    : improved
      ? intl.formatMessage(
          {
            defaultMessage: '{count, plural, one {# improved trace} other {# improved traces}} compared with baseline',
            description:
              'Accessible label for the number of matched traces that improved compared with the baseline run',
          },
          { count },
        )
      : intl.formatMessage(
          {
            defaultMessage:
              '{count, plural, one {# regressed trace} other {# regressed traces}} compared with baseline',
            description:
              'Accessible label for the number of matched traces that regressed compared with the baseline run',
          },
          { count },
        );

const EvalRunComparisonChangeCount = ({ count, improved, isGroupedBySession }: EvalRunComparisonChangeCountProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const label = getChangeCountLabel(intl, count, improved, isGroupedBySession);
  return (
    <span role="img" aria-label={label} css={{ display: 'inline-flex', alignItems: 'center' }}>
      <Typography.Text size="sm" color={improved ? 'success' : 'error'}>
        <span aria-hidden css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <span
            data-testid={improved ? 'comparison-summary-improvement-icon' : 'comparison-summary-regression-icon'}
            css={{ display: 'inline-flex', transform: improved ? 'rotate(-45deg)' : 'rotate(45deg)' }}
          >
            <ArrowRightIcon size={theme.typography.fontSizeSm} />
          </span>
          {intl.formatNumber(count)}
        </span>
      </Typography.Text>
    </span>
  );
};

const getAggregateDescription = (intl: IntlShape, label: EvalRunComparisonColumnAggregate['label']): string => {
  switch (label) {
    case 'AVG':
      return intl.formatMessage({
        defaultMessage: 'average',
        description: 'Human-readable average aggregate label in a comparison summary sentence',
      });
    case 'MEDIAN':
      return intl.formatMessage({
        defaultMessage: 'median',
        description: 'Human-readable median aggregate label in a comparison summary sentence',
      });
    case 'MIN':
      return intl.formatMessage({
        defaultMessage: 'minimum',
        description: 'Human-readable minimum aggregate label in a comparison summary sentence',
      });
    case 'MAX':
      return intl.formatMessage({
        defaultMessage: 'maximum',
        description: 'Human-readable maximum aggregate label in a comparison summary sentence',
      });
    case 'PASS':
      return intl.formatMessage({
        defaultMessage: 'pass rate',
        description: 'Human-readable pass rate aggregate label in a comparison summary sentence',
      });
    case 'TRUE':
      return intl.formatMessage({
        defaultMessage: 'true rate',
        description: 'Human-readable true rate aggregate label in a comparison summary sentence',
      });
    case 'ERROR RATE':
      return intl.formatMessage({
        defaultMessage: 'error rate',
        description: 'Human-readable error rate aggregate label in a comparison summary sentence',
      });
    default:
      return intl.formatMessage({
        defaultMessage: 'summary',
        description: 'Fallback aggregate label in a comparison summary sentence',
      });
  }
};

const getDeltaImproved = (delta: number | undefined, higherIsBetter?: boolean): boolean | undefined => {
  if (delta === undefined || delta === 0) return undefined;
  const increased = delta > 0;
  return increased === (higherIsBetter ?? true);
};

const getDeltaColor = (delta: number | undefined, higherIsBetter?: boolean) => {
  const improved = getDeltaImproved(delta, higherIsBetter);
  return improved === undefined ? 'secondary' : improved ? 'success' : 'error';
};

const getDeltaTextColor = (delta: number | undefined, higherIsBetter: boolean | undefined, theme: Theme) => {
  const color = getDeltaColor(delta, higherIsBetter);
  return color === 'success'
    ? theme.colors.textValidationSuccess
    : color === 'error'
      ? theme.colors.textValidationDanger
      : theme.colors.textSecondary;
};

const getOutcomeFilterLabel = (intl: IntlShape, count: number, improved: boolean) =>
  improved
    ? intl.formatMessage(
        {
          defaultMessage: '{count, plural, one {# improvement} other {# improvements}}',
          description: 'Improvement count in a comparison summary filter menu',
        },
        { count },
      )
    : intl.formatMessage(
        {
          defaultMessage: '{count, plural, one {# regression} other {# regressions}}',
          description: 'Regression count in a comparison summary filter menu',
        },
        { count },
      );

const OutcomeFilterItem = ({
  outcome,
  count,
  checked,
  isGroupedBySession,
  onClick,
}: {
  outcome: EvalRunComparisonOutcome;
  count: number;
  checked: boolean;
  isGroupedBySession?: boolean;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const improved = outcome === 'improvement';
  return (
    <DropdownMenu.CheckboxItem
      data-testid={`comparison-summary-${outcome}-filter`}
      componentId={
        improved
          ? 'mlflow.eval-run-comparison.column-summary.improvements'
          : 'mlflow.eval-run-comparison.column-summary.regressions'
      }
      checked={checked}
      onCheckedChange={onClick}
      aria-label={getChangeCountLabel(intl, count, improved, isGroupedBySession)}
    >
      <DropdownMenu.ItemIndicator
        css={{
          position: 'absolute',
          left: 'auto',
          right: theme.spacing.sm,
          marginLeft: 0,
          fontSize: theme.general.iconFontSize,
        }}
      />
      <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
        <span
          aria-hidden="true"
          css={{
            color: improved ? theme.colors.textValidationSuccess : theme.colors.textValidationDanger,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: theme.spacing.md,
            flexShrink: 0,
          }}
        >
          <span
            data-testid={`comparison-summary-${outcome}-icon`}
            css={{ display: 'inline-flex', transform: improved ? 'rotate(-45deg)' : 'rotate(45deg)' }}
          >
            <ArrowRightIcon size={theme.typography.fontSizeSm} />
          </span>
        </span>
        <Typography.Text size="sm">{getOutcomeFilterLabel(intl, count, improved)}</Typography.Text>
      </span>
    </DropdownMenu.CheckboxItem>
  );
};

export const EvalRunComparisonColumnSummary = ({
  aggregate,
  runs,
  isGroupedBySession,
  columnLabel,
  higherIsBetter,
  filterOptionCounts,
  onOutcomeClick,
  activeOutcomes,
}: EvalRunComparisonColumnSummaryProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // Opens on click only: a hover-opened menu covers the run rows below it.
  const [openRunUuid, setOpenRunUuid] = useState<string>();
  const aggregateLabel =
    aggregate.label === 'AVG'
      ? intl.formatMessage({
          defaultMessage: 'AVG',
          description: 'Arithmetic mean label in a comparison column summary',
        })
      : aggregate.label === 'MEDIAN'
        ? intl.formatMessage({ defaultMessage: 'MEDIAN', description: 'Median label in a comparison column summary' })
        : aggregate.label === 'MIN'
          ? intl.formatMessage({ defaultMessage: 'MIN', description: 'Minimum label in a comparison column summary' })
          : aggregate.label === 'MAX'
            ? intl.formatMessage({
                defaultMessage: 'MAX',
                description: 'Maximum label in a comparison column summary',
              })
            : aggregate.label === 'WIN / LOSE'
              ? intl.formatMessage({
                  defaultMessage: 'WIN / LOSE',
                  description: 'Win and loss count order in a pairwise preference column summary',
                })
              : aggregate.label === 'PASS'
                ? intl.formatMessage({
                    defaultMessage: 'PASS',
                    description: 'Pass percentage label in a comparison column summary',
                  })
                : aggregate.label === 'TRUE'
                  ? intl.formatMessage({
                      defaultMessage: 'TRUE',
                      description: 'True percentage label in a comparison column summary',
                    })
                  : aggregate.label === 'ERROR RATE'
                    ? intl.formatMessage({
                        defaultMessage: 'ERROR RATE',
                        description: 'Percentage of completed traces that failed in a comparison column summary',
                      })
                    : undefined;
  const isRateAggregate = aggregate.label === 'PASS' || aggregate.label === 'TRUE' || aggregate.label === 'ERROR RATE';
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, paddingTop: theme.spacing.xs }}>
      {runs.map((run, index) => {
        const value = aggregate.values[index];
        const menuChangeCounts = filterOptionCounts?.[index] ?? value?.changeCounts;
        const baselineDisplayValue = aggregate.values[0]?.displayValue;
        const contextualDelta =
          index > 0 && value && value.displayValue !== '—' && baselineDisplayValue && baselineDisplayValue !== '—'
            ? (value.displayDelta ?? (value.delta === 0 ? '0' : undefined))
            : undefined;
        const contextualPercentDelta =
          index > 0 && !isRateAggregate && value?.percentDelta !== undefined
            ? formatEvalRunComparisonPercentChange(intl, value.percentDelta)
            : undefined;
        const runLabel = (
          <span
            data-testid="comparison-summary-run-label"
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              verticalAlign: 'middle',
              maxWidth: '100%',
            }}
          >
            <span
              aria-hidden="true"
              data-testid="comparison-summary-run-dot"
              style={{ backgroundColor: run.color }}
              css={{ width: theme.spacing.sm, height: theme.spacing.sm, borderRadius: '50%', flexShrink: 0 }}
            />
            <span
              css={{ minWidth: 0, maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
              {run.label}
            </span>
          </span>
        );
        const rowContents = (
          <>
            <span css={{ display: 'flex', alignSelf: 'center' }}>
              <RunColorPill aria-hidden color={run.color} data-testid="comparison-summary-run-color" />
            </span>
            <Typography.Text bold={index === 0} size={index === 0 ? 'md' : 'sm'}>
              {value?.displayValue ?? '—'}
            </Typography.Text>
            {index === 0 && aggregateLabel && (
              <Typography.Text size="sm" color="secondary">
                {aggregateLabel}
              </Typography.Text>
            )}
            {value?.displayDelta && (
              <Typography.Text size="sm" color={getDeltaColor(value.delta, higherIsBetter)}>
                {value.displayDelta}
              </Typography.Text>
            )}
            {index > 0 && value?.changeCounts && (
              <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, marginLeft: 'auto' }}>
                {value.changeCounts.improved > 0 && (
                  <EvalRunComparisonChangeCount
                    count={value.changeCounts.improved}
                    improved
                    isGroupedBySession={isGroupedBySession}
                  />
                )}
                {value.changeCounts.regressed > 0 && (
                  <EvalRunComparisonChangeCount
                    count={value.changeCounts.regressed}
                    improved={false}
                    isGroupedBySession={isGroupedBySession}
                  />
                )}
              </span>
            )}
          </>
        );

        const rowCss = {
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          whiteSpace: 'nowrap' as const,
          width: '100%',
          fontVariantNumeric: 'tabular-nums',
          boxSizing: 'border-box' as const,
        };

        const row = (
          <DropdownMenu.Root
            modal={false}
            open={openRunUuid === run.runUuid}
            onOpenChange={(open) => {
              if (open) {
                setOpenRunUuid(run.runUuid);
              } else {
                setOpenRunUuid((currentRunUuid) => (currentRunUuid === run.runUuid ? undefined : currentRunUuid));
              }
            }}
          >
            <DropdownMenu.Trigger asChild>
              <button
                type="button"
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => event.stopPropagation()}
                aria-label={intl.formatMessage(
                  {
                    defaultMessage: 'Open comparison summary for {run}',
                    description: 'Accessible label for opening a comparison run summary menu',
                  },
                  { run: run.label },
                )}
                css={{
                  ...rowCss,
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                  border: 0,
                  borderRadius: theme.borders.borderRadiusSm,
                  color: 'inherit',
                  background: 'transparent',
                  textAlign: 'left',
                  cursor: 'pointer',
                  '&:hover, &[data-state="open"]': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
                  '&:focus-visible': {
                    outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
                    outlineOffset: 1,
                  },
                }}
              >
                {rowContents}
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content
              align="start"
              minWidth={0}
              css={{
                width: `min(360px, calc(100vw - ${theme.spacing.lg}px))`,
                maxWidth: `calc(100vw - ${theme.spacing.lg}px)`,
                boxSizing: 'border-box',
              }}
            >
              <DropdownMenu.Label>
                <Typography.Text size="sm" color="secondary">
                  {index === 0
                    ? intl.formatMessage({
                        defaultMessage: 'Baseline run',
                        description: 'Header for a baseline run summary menu',
                      })
                    : intl.formatMessage({
                        defaultMessage: 'Comparison run',
                        description: 'Header for a comparison run summary menu',
                      })}
                </Typography.Text>
              </DropdownMenu.Label>
              <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.md}px` }}>
                <Typography.Text size="sm" data-testid="comparison-summary-value-sentence">
                  {intl.formatMessage(
                    {
                      defaultMessage: 'For {run}, the {aggregate} value for {column} is {value}.',
                      description: 'Summary of a comparison run aggregate value',
                    },
                    {
                      run: runLabel,
                      aggregate: getAggregateDescription(intl, aggregate.label),
                      column:
                        columnLabel ??
                        intl.formatMessage({
                          defaultMessage: 'this assessment',
                          description: 'Fallback assessment label in a comparison summary sentence',
                        }),
                      value: (
                        <span
                          data-testid="comparison-summary-aggregate-value"
                          css={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            padding: `0 ${theme.spacing.xs}px`,
                            borderRadius: theme.borders.borderRadiusSm,
                            backgroundColor: theme.colors.backgroundSecondary,
                            fontVariantNumeric: 'tabular-nums',
                          }}
                        >
                          {value?.displayValue ?? '—'}
                        </span>
                      ),
                    },
                  )}
                </Typography.Text>
                {(contextualDelta || contextualPercentDelta) && (
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <Typography.Text size="sm">
                      {intl.formatMessage(
                        contextualDelta
                          ? isRateAggregate
                            ? deltaSentenceMessages.rateDelta
                            : deltaSentenceMessages.delta
                          : deltaSentenceMessages.percent,
                        {
                          delta: contextualDelta ? (
                            <span
                              data-testid="comparison-summary-contextual-delta"
                              css={{ color: getDeltaTextColor(value?.delta, higherIsBetter, theme) }}
                            >
                              {contextualDelta}
                            </span>
                          ) : (
                            ''
                          ),
                          percent: contextualPercentDelta ? (
                            <span
                              data-testid="comparison-summary-contextual-percent-delta"
                              css={{
                                color: getDeltaTextColor(value?.percentDelta, higherIsBetter, theme),
                              }}
                            >
                              {contextualDelta ? ` (${contextualPercentDelta})` : contextualPercentDelta}
                            </span>
                          ) : (
                            ''
                          ),
                        },
                      )}
                    </Typography.Text>
                  </div>
                )}
              </div>
              {index > 0 &&
                onOutcomeClick &&
                menuChangeCounts &&
                (menuChangeCounts.improved > 0 || menuChangeCounts.regressed > 0) && (
                  <>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Label>
                      <Typography.Text size="sm" color="secondary">
                        {intl.formatMessage({
                          defaultMessage: 'Filter by',
                          description: 'Label for outcome filters in a comparison summary menu',
                        })}
                      </Typography.Text>
                    </DropdownMenu.Label>
                    <div
                      data-testid="comparison-summary-outcome-filters"
                      css={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 140px), 1fr))',
                        gap: theme.spacing.xs,
                        padding: `0 ${theme.spacing.xs}px`,
                        '& > [role="menuitemcheckbox"]': {
                          boxSizing: 'border-box',
                          width: '100%',
                          minWidth: 0,
                          paddingLeft: theme.spacing.sm + theme.spacing.xs,
                          paddingRight: theme.spacing.lg,
                          margin: `${theme.spacing.xs / 2}px 0`,
                          borderRadius: theme.borders.borderRadiusSm,
                          border: `1px solid ${theme.colors.borderDecorative}`,
                          backgroundColor: theme.colors.backgroundSecondary,
                          cursor: 'pointer',
                          transition: 'background-color 120ms ease-in-out',
                          '&:hover': { backgroundColor: theme.colors.actionTertiaryBackgroundHover },
                          '&:focus-visible': {
                            outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
                            outlineOffset: -1,
                          },
                          '& > span': {
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          },
                        },
                      }}
                    >
                      {menuChangeCounts.improved > 0 && (
                        <OutcomeFilterItem
                          outcome="improvement"
                          count={menuChangeCounts.improved}
                          checked={activeOutcomes?.[run.runUuid] === 'improvement'}
                          isGroupedBySession={isGroupedBySession}
                          onClick={() => onOutcomeClick(run.runUuid, 'improvement')}
                        />
                      )}
                      {menuChangeCounts.regressed > 0 && (
                        <OutcomeFilterItem
                          outcome="regression"
                          count={menuChangeCounts.regressed}
                          checked={activeOutcomes?.[run.runUuid] === 'regression'}
                          isGroupedBySession={isGroupedBySession}
                          onClick={() => onOutcomeClick(run.runUuid, 'regression')}
                        />
                      )}
                    </div>
                  </>
                )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        );

        return (
          <div key={run.runUuid} role="group" aria-label={run.label} css={{ width: '100%' }}>
            {row}
          </div>
        );
      })}
    </div>
  );
};
