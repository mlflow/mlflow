import type { ReactNode } from 'react';
import { FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import { Typography, useDesignSystemTheme, type ThemeType } from '@databricks/design-system';
// Not re-exported from the genai-traces-table barrel in OSS — import from their modules.
import { getAssessmentValueLabel } from '@databricks/web-shared/genai-traces-table/components/GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, AssessmentValueType } from '@databricks/web-shared/genai-traces-table';

/** Cap the hover breakdown so a numeric scorer with many distinct scores doesn't render a giant tooltip. */
const MAX_BREAKDOWN_ROWS = 8;

export interface ScorerValueCount {
  value: AssessmentValueType;
  count: number;
}

export interface ScorerScoreBreakdownProps {
  scorer: { dtype: AssessmentInfo['dtype']; assessmentInfo: AssessmentInfo; valueCounts: ScorerValueCount[] };
}

const renderValueBreakdown = (
  scorer: ScorerScoreBreakdownProps['scorer'],
  intl: IntlShape,
  theme: ThemeType,
): ReactNode => {
  const rows = scorer.valueCounts.slice(0, MAX_BREAKDOWN_ROWS);
  const remaining = scorer.valueCounts.length - rows.length;
  // Pass/fail (and boolean) labels are self-explanatory, so the check/X icon is just noise here.
  const showIcon = scorer.dtype !== 'pass-fail' && scorer.dtype !== 'boolean';

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      {rows.map(({ value, count }) => {
        // The shared label helper renders `undefined` as the literal string, so name the missing bucket here.
        const { content, icon } =
          value === undefined
            ? {
                content: intl.formatMessage({
                  defaultMessage: 'No value',
                  description: 'Label for the count of eval traces a scorer did not assess',
                }),
                icon: undefined,
              }
            : getAssessmentValueLabel(intl, theme, scorer.assessmentInfo, value);

        return (
          <div
            key={String(value)}
            css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.md }}
          >
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              {showIcon && icon}
              {content}
            </span>
            <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>{count}</span>
          </div>
        );
      })}
      {remaining > 0 && (
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="+{remaining} more"
            description="Suffix noting how many additional scorer values are omitted from the hover breakdown"
            values={{ remaining }}
          />
        </Typography.Text>
      )}
    </div>
  );
};

/**
 * For a numeric scorer with more distinct values than the breakdown can list (e.g. a continuous 0-1
 * similarity score), a per-value count is noise — most counts are 1 and the top-N is arbitrary. Show a
 * compact summary instead: how many traces were scored and the observed min-max (observed, not the
 * scorer's declared range, which the backend doesn't return yet).
 */
const renderNumericSummary = (
  scorer: ScorerScoreBreakdownProps['scorer'],
  intl: IntlShape,
  theme: ThemeType,
): ReactNode => {
  const numericEntries = scorer.valueCounts.filter(
    (entry): entry is { value: number; count: number } => typeof entry.value === 'number',
  );
  const tracesScored = numericEntries.reduce((sum, { count }) => sum + count, 0);
  const { min, max } = numericEntries.reduce(
    (acc, { value }) => ({ min: Math.min(acc.min, value), max: Math.max(acc.max, value) }),
    { min: Infinity, max: -Infinity },
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <span>
        <FormattedMessage
          defaultMessage="{count, number} traces scored"
          description="Hover summary of how many traces a high-cardinality numeric scorer evaluated"
          values={{ count: tracesScored }}
        />
      </span>
      <span>
        <FormattedMessage
          defaultMessage="Observed range {min}–{max}"
          description="Hover summary of the smallest and largest observed values for a high-cardinality numeric scorer"
          values={{
            min: getAssessmentValueLabel(intl, theme, scorer.assessmentInfo, min).content,
            max: getAssessmentValueLabel(intl, theme, scorer.assessmentInfo, max).content,
          }}
        />
      </span>
    </div>
  );
};

/**
 * Numeric scorers with more distinct values than the list can hold get the summary; every other scorer
 * (categorical, or low-cardinality numeric like a 1-5 rating) gets the per-value breakdown.
 */
export const ScorerScoreBreakdown = ({ scorer }: ScorerScoreBreakdownProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const distinctNumericValues =
    scorer.dtype === 'numeric' ? scorer.valueCounts.filter((entry) => typeof entry.value === 'number').length : 0;
  return (
    <>
      {distinctNumericValues > MAX_BREAKDOWN_ROWS
        ? renderNumericSummary(scorer, intl, theme)
        : renderValueBreakdown(scorer, intl, theme)}
    </>
  );
};
