import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TraceAssessmentCell } from '../../components/experiment-page/components/traces-v4/components/TraceAssessmentCell';

export interface EvalRunComparisonAssessmentDelta {
  value: number;
  baselineValue: number;
  difference: number;
}

// Two fixed decimals match the score bar's value, so the row reads consistently.
const SCORE_DECIMALS = { minimumFractionDigits: 2, maximumFractionDigits: 2 } as const;

const formatEvalRunComparisonScore = (intl: IntlShape, value: number): string =>
  intl.formatNumber(value, SCORE_DECIMALS);

export const formatEvalRunComparisonAssessmentDelta = (intl: IntlShape, difference: number): string =>
  intl.formatNumber(difference, { ...SCORE_DECIMALS, signDisplay: 'exceptZero' });

export interface EvalRunComparisonAssessmentCellProps {
  trace: ModelTraceInfoV3;
  assessmentName: string;
  delta?: EvalRunComparisonAssessmentDelta;
}

/** A numeric scorer's score bar, followed by its difference from the baseline trace when one applies. */
export const EvalRunComparisonAssessmentCell = ({
  trace,
  assessmentName,
  delta,
}: EvalRunComparisonAssessmentCellProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const scoreBar = <TraceAssessmentCell trace={trace} assessmentName={assessmentName} columnType="numeric" />;
  if (!delta) {
    return scoreBar;
  }
  const deltaTag = (
    <Tag
      componentId="mlflow.eval-run-comparison.traces.assessment-delta-tag"
      color={delta.difference > 0 ? 'lime' : 'coral'}
    >
      {formatEvalRunComparisonAssessmentDelta(intl, delta.difference)}
    </Tag>
  );
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, maxWidth: '100%' }}>
      {scoreBar}
      <HoverCard
        trigger={deltaTag}
        content={
          // The card is portaled, so its clicks would otherwise also activate the comparison row.
          <div
            onClick={(event) => event.stopPropagation()}
            css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
          >
            <Typography.Text>
              <FormattedMessage
                defaultMessage="{baseline} → {current}"
                description="Baseline trace's numeric scorer value followed by this run's value, in the hover card of a comparison delta pill"
                values={{
                  baseline: formatEvalRunComparisonScore(intl, delta.baselineValue),
                  current: formatEvalRunComparisonScore(intl, delta.value),
                }}
              />
            </Typography.Text>
            {deltaTag}
          </div>
        }
      />
    </span>
  );
};
