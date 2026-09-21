import React from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { FeedbackAssessment } from '@databricks/web-shared/model-trace-explorer';

function parseMap(value: string | undefined): Record<string, unknown> {
  if (!value) return {};
  try {
    const result: unknown = JSON.parse(value);
    return result && typeof result === 'object' && !Array.isArray(result) ? (result as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

const JevAssessmentDetails: React.FC<{ assessment: FeedbackAssessment }> = ({ assessment }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const metadata = assessment.metadata;
  if (!metadata?.['jev.model']) return null;
  const probabilities = Object.entries(parseMap(metadata['jev.probabilities'])).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === 'number' && Number.isFinite(entry[1]) && entry[1] >= 0 && entry[1] <= 1,
  );
  const legend = parseMap(metadata['jev.legend']);
  const formatProbability = (value: string | undefined) => {
    if (value === undefined || value.trim() === '') return undefined;
    const numericValue = Number(value);
    return Number.isFinite(numericValue) && numericValue >= 0 && numericValue <= 1
      ? intl.formatNumber(numericValue, { style: 'percent', maximumFractionDigits: 2 })
      : undefined;
  };
  const probability = formatProbability(metadata['jev.probability']);
  const confidence = formatProbability(metadata['jev.confidence']);

  return (
    <div css={{ border: `1px solid ${theme.colors.border}`, padding: theme.spacing.md }}>
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Jev result: {value}"
          description="Decision returned by Jev"
          values={{ value: String(assessment.feedback.value ?? '') }}
        />
      </Typography.Text>
      <dl
        css={{
          margin: 0,
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: theme.spacing.xs,
          '& dd': { margin: 0 },
        }}
      >
        {probability && (
          <>
            <dt>
              <FormattedMessage defaultMessage="Probability of true" description="Jev noul probability label" />
            </dt>
            <dd>{probability}</dd>
          </>
        )}
        {confidence && (
          <>
            <dt>
              <FormattedMessage defaultMessage="Confidence" description="Jev confidence label" />
            </dt>
            <dd>{confidence}</dd>
          </>
        )}
        {probabilities.map(([label, value]) => (
          <React.Fragment key={label}>
            <dt>{typeof legend[label] === 'string' ? `${label}: ${legend[label]}` : label}</dt>
            <dd>{intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 2 })}</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
};

export default JevAssessmentDetails;
