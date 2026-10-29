import { Fragment } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { EvalRunComparisonDelta, getEvalRunComparisonRelativeChangePercent } from './EvalRunComparisonDelta';
import type { ComparisonRun, ComparisonScorerDelta, EvalRunComparisonSnapshot } from './evalRunComparisonData';

export interface ComparisonRunReferenceProps {
  run: ComparisonRun;
}

const ComparisonRunReference = ({ run }: ComparisonRunReferenceProps) => {
  const getRunColor = useGetExperimentRunColor();
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        display: 'inline-flex',
        alignItems: 'baseline',
        gap: theme.spacing.xs,
        maxWidth: '100%',
        overflowWrap: 'anywhere',
      }}
    >
      <span aria-hidden="true" css={{ display: 'inline-flex', flexShrink: 0 }}>
        <RunColorPill color={getRunColor(run.runUuid)} />
      </span>
      <Typography.Text bold>{run.runName}</Typography.Text>
    </span>
  );
};

export interface ComparisonDeltaPillProps {
  delta: ComparisonScorerDelta;
  run: ComparisonRun;
  baseline: ComparisonRun;
}

const ComparisonDeltaPill = ({ delta }: ComparisonDeltaPillProps) => {
  const intl = useIntl();
  const changePercent =
    delta.unit === 'percentagePoints'
      ? delta.delta
      : getEvalRunComparisonRelativeChangePercent(delta.value, delta.baselineValue);
  const formattedChange =
    changePercent === undefined
      ? intl.formatMessage({
          defaultMessage: 'N/A',
          description: 'Relative scorer change unavailable because the baseline score is zero',
        })
      : intl.formatNumber(changePercent / 100, {
          style: 'percent',
          maximumFractionDigits: 2,
          signDisplay: 'exceptZero',
        });
  const signedDelta = intl.formatNumber(delta.delta, {
    signDisplay: 'exceptZero',
    maximumFractionDigits: 4,
  });
  const description = intl.formatMessage(
    {
      defaultMessage: 'Absolute difference: {delta}{unit}. {directionNote}',
      description: 'Tooltip for the absolute difference in an evaluation comparison scorer',
    },
    {
      delta: signedDelta,
      unit:
        delta.unit === 'percentagePoints'
          ? ' ' +
            intl.formatMessage({
              defaultMessage: 'percentage points',
              description: 'Unit for a pass-rate difference',
            })
          : '',
      directionNote:
        delta.unit === 'percentagePoints'
          ? ''
          : intl.formatMessage({
              defaultMessage: 'Higher or lower is not necessarily better.',
              description: 'Direction is not rated for a numeric scorer difference',
            }),
    },
  );
  return (
    <EvalRunComparisonDelta
      currentValue={delta.value}
      baselineValue={delta.baselineValue}
      percentagePoints={delta.unit === 'percentagePoints'}
      absoluteDifference={description}
      componentId="mlflow.eval_run_details.compare.insight.scorer"
      color={delta.direction === 'improved' ? 'teal' : delta.direction === 'regressed' ? 'coral' : 'default'}
    >
      {delta.name} {formattedChange}
    </EvalRunComparisonDelta>
  );
};

export interface EvalRunComparisonInsightContentProps {
  snapshot: EvalRunComparisonSnapshot;
}

export const EvalRunComparisonInsightContent = ({ snapshot }: EvalRunComparisonInsightContentProps) => {
  const baseline = snapshot.runs[0];
  if (!baseline || snapshot.deltas.length === 0) {
    return (
      <FormattedMessage
        defaultMessage="There are not enough comparable scorer results to summarize these runs. Missing scores are not treated as zero."
        description="Comparison insight fallback when selected runs have no comparable results"
      />
    );
  }
  const leader = snapshot.runs.find((run) => run.runUuid === snapshot.leaderRunUuid);
  return (
    <>
      {snapshot.comparisons.map((comparison) => {
        const run = snapshot.runs.find((item) => item.runUuid === comparison.runUuid);
        if (!run) {
          return null;
        }
        const changes = snapshot.deltas.filter((delta) => delta.runUuid === run.runUuid);
        const successRateCount = comparison.improved + comparison.regressed + comparison.unchanged;
        const highlights = [...changes]
          .filter((delta) => delta.delta !== 0)
          // Pass-rate changes (percentage points) come before numeric changes (relative percent): the units differ.
          .sort((a, b) => {
            if (a.unit !== b.unit) return a.unit === 'percentagePoints' ? -1 : 1;
            const aChange =
              a.unit === 'percentagePoints'
                ? a.delta
                : getEvalRunComparisonRelativeChangePercent(a.value, a.baselineValue);
            const bChange =
              b.unit === 'percentagePoints'
                ? b.delta
                : getEvalRunComparisonRelativeChangePercent(b.value, b.baselineValue);
            if (aChange === undefined) return bChange === undefined ? 0 : 1;
            if (bChange === undefined) return -1;
            return Math.abs(bChange) - Math.abs(aChange);
          })
          .slice(0, 2);
        return (
          <Fragment key={run.runUuid}>
            {changes.length === 0 ? (
              <FormattedMessage
                defaultMessage="{run} has no comparable scorer results with baseline {baseline}."
                description="Comparison fallback sentence for a run missing common scorers"
                values={{
                  run: <ComparisonRunReference run={run} />,
                  baseline: <ComparisonRunReference run={baseline} />,
                }}
              />
            ) : (
              <>
                {successRateCount > 0 ? (
                  <FormattedMessage
                    defaultMessage="Compared with baseline {baseline}, {run} improves on {improved} and regresses on {regressed} of {count, plural, one {# shared success-rate scorer} other {# shared success-rate scorers}}, with {unchanged} unchanged."
                    description="Deterministic baseline-relative scorer comparison summary"
                    values={{
                      baseline: <ComparisonRunReference run={baseline} />,
                      run: <ComparisonRunReference run={run} />,
                      improved: comparison.improved,
                      regressed: comparison.regressed,
                      unchanged: comparison.unchanged,
                      count: successRateCount,
                    }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Compared with baseline {baseline}, {run} shares no success-rate scorers with it."
                    description="Baseline-relative comparison summary when the runs share only numeric scorers"
                    values={{
                      baseline: <ComparisonRunReference run={baseline} />,
                      run: <ComparisonRunReference run={run} />,
                    }}
                  />
                )}{' '}
                {comparison.numericChanges > 0 && (
                  <FormattedMessage
                    defaultMessage="{count, plural, one {# numeric scorer changed} other {# numeric scorers changed}}; direction is not rated as better or worse."
                    description="Comparison fallback explaining neutral numeric changes"
                    values={{ count: comparison.numericChanges }}
                  />
                )}
                {highlights.length > 0 && (
                  <>
                    {' '}
                    <FormattedMessage
                      defaultMessage="Largest changes: {scorers}."
                      description="Comparison fallback naming the largest scorer changes for a run"
                      values={{
                        scorers: highlights.map((delta, index) => (
                          <Fragment key={delta.id}>
                            {index > 0 ? ', ' : ''}
                            <ComparisonDeltaPill delta={delta} run={run} baseline={baseline} />
                          </Fragment>
                        )),
                      }}
                    />
                  </>
                )}
              </>
            )}{' '}
          </Fragment>
        );
      })}
      {leader ? (
        <FormattedMessage
          defaultMessage="{run} has the highest average passing rate across the {count, plural, one {# success-rate scorer} other {# success-rate scorers}} shared by all selected runs."
          description="Qualified comparison leader based on equally weighted common success-rate scorers"
          values={{ run: <ComparisonRunReference run={leader} />, count: snapshot.commonSuccessScorerCount }}
        />
      ) : (
        <FormattedMessage
          defaultMessage="There is no clear overall leader from the shared success-rate scorers."
          description="Comparison fallback for tied scores or insufficient common success-rate coverage"
        />
      )}
    </>
  );
};
